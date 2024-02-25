# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer



import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

listdir = os.listdir(r"C:\Users\felhasznalo\Desktop\tracker\dataset\0000f77c-6257be58\frames")

def draw_boxes(image_path, boxes, classes, image_name):
    save_dir = r"C:\Users\felhasznalo\Desktop\tracker\saved_images"
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size

        # Move tensors to CPU and convert to NumPy arrays
        boxes = boxes.cpu().numpy()
        class_labels = classes[1].cpu().numpy()

        # Create a figure and axis to plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Convert normalized coordinates to absolute pixel values
        boxes[:, [0, 2]] *= width  # Convert x and w
        boxes[:, [1, 3]] *= height  # Convert y and h
        boxes[:, 0] -= boxes[:, 2] / 2  # Calculate left x coordinate
        boxes[:, 1] -= boxes[:, 3] / 2  # Calculate top y coordinate

        # Draw each box with its label
        for box, label in zip(boxes, class_labels):
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f'Class {label}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Save the plot
        save_path = os.path.join(save_dir, f'{image_name}.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the 



class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

matcher_dict = {}
last_indices = None
last_actual_frame_ids = None
id_counter = 0

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    """
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes = torch.stack([torch.cat((torch.ones((target_classes[i] == 1).sum().item(), dtype=target_classes.dtype),
                                  torch.full((target_classes[i].numel() - (target_classes[i] == 1).sum().item(),), 2, dtype=target_classes.dtype)))
                       for i in range(target_classes.size(0))]).to('cuda')
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    """
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pass
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        #print(src_boxes)
        #print(target_boxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    """
    def loss_boxes(self, outputs, targets, indices, num_boxes):

        assert 'pred_boxes' in outputs
        idx = self._get_tgt_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    """
    def loss_masks(self, outputs, targets, indices, num_boxes):
        pass

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    """
    def forward(self, outputs, targets):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    """

    def forward(self, outputs, targets):

        global matcher_dict
        global last_indices
        global last_actual_frame_ids
        global id_counter


        #batch predikt megcsinal

        #print(id_counter)
        if id_counter % 10 == 0: 
            matcher_dict = {}
            last_indices = None
            last_actual_frame_ids = None
            id_counter = 0


        

        


        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        #print(outputs_without_aux)
        #print()
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = None



        actual_frame_ids = targets[0]["classes"].cpu().tolist()

        if len(matcher_dict) == 0:
            #print("ALSO")
            indices = self.matcher(outputs_without_aux, targets)

            #indices = [(pair[0][torch.argsort(pair[1])], pair[1][torch.argsort(pair[1])]) for pair in indices]
            tensor_values_list = indices[0][0].tolist()

            for i, id in enumerate(actual_frame_ids):
                matcher_dict[id] = tensor_values_list[i]
            #print(len(matcher_dict))
        else:
            indices = last_indices.copy()

            missing_ids = [id for id in last_actual_frame_ids if id not in actual_frame_ids]

            if len(missing_ids) > 0: #3. eltűnik egy GT
                for key in missing_ids:
                    if key in matcher_dict:
                        del matcher_dict[key]

            print(matcher_dict)
            print(actual_frame_ids)
            gt_s_not_in_matcher = []
            for gt_id in actual_frame_ids:
                if gt_id not in matcher_dict:
                    gt_s_not_in_matcher.append(gt_id)





            if len(gt_s_not_in_matcher) == 0: #1. ugyan az azok a GT-k vannak
                indices = last_indices.copy()
                #print("KÖZÉP")
            print(gt_s_not_in_matcher)
            if len(gt_s_not_in_matcher) > 0: #2. új GT
                print("ÚJ")
                indices = last_indices.copy()
                filtered_targets = []
                for target in targets:
                    mask = torch.isin(target['classes'], torch.tensor(gt_s_not_in_matcher, device=target['classes'].device))
                    filtered_targets.append({
                        'labels': target['labels'][mask],
                        'boxes': target['boxes'][mask],
                        'classes': target['classes'][mask]
                    })

                indices_to_remove = indices[0][0].tolist()

                outputs_without_aux_copy = outputs_without_aux.copy()
                backup = []
                for index in indices_to_remove:
                    backup.append(outputs_without_aux_copy['pred_boxes'][0][index].clone())
                    outputs_without_aux_copy['pred_boxes'][0][index].data.fill_(11)
                    
                

                indices_for_news = self.matcher(outputs_without_aux_copy, filtered_targets)

                for index, element in zip(indices_to_remove, backup):
                    outputs_without_aux_copy['pred_boxes'][0][index].data.copy_(element.detach())



                #indices_for_news = [(pair[0][torch.argsort(pair[1])], pair[1][torch.argsort(pair[1])]) for pair in indices_for_news]
                combined_tensors = []
                
                for tensors in indices + indices_for_news:
                    if combined_tensors:
                        new_second_tensor = tensors[1] + combined_tensors[-1][1][-1] + 1
                    else:
                        new_second_tensor = tensors[1]
                    combined_tensors.append((tensors[0], new_second_tensor))

                res = [(torch.cat([t[0] for t in combined_tensors]), torch.cat([t[1] for t in combined_tensors]))]
                
                indices = res


                
                for i, id in enumerate(gt_s_not_in_matcher):
                    matcher_dict[id] = indices[0][0].tolist()[-len(gt_s_not_in_matcher)+i]



        last_actual_frame_ids = actual_frame_ids.copy()


        dict_values = set(matcher_dict.values())
 
        filtered_first_tensor = [value for value in indices[0][0] if value.item() in dict_values]

        filtered_second_tensor = torch.arange(len(filtered_first_tensor))

 
        result = [(torch.tensor(filtered_first_tensor), filtered_second_tensor)]

        indices = result.copy()

        last_indices = indices.copy()

        #print(outputs_without_aux)
        #print(indices)       
        #print(matcher_dict)
        #print(actual_frame_ids)
        print(id_counter)

        
        """image = os.path.join(r'ames', listdir[0])
        listdir.pop(0)
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs_without_aux['pred_boxes'][idx].detach()
        #print(src_boxes)

        draw_boxes(image, src_boxes, idx, listdir[0])
        """









        id_counter += 1


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        #print(num_boxes)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
               
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses
        
        

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    

    return model, criterion, postprocessors
