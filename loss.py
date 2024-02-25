import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from coordinate import CoordinateConverter
import numpy as np

converter = CoordinateConverter()

def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def ciou(bboxes1, bboxes2):
    """
    Calculate CIoU between two bounding boxes
    bboxes1, bboxes2: [x1, y1, x2, y2]
    """

    # Calculate intersection area
    x1_inter = max(bboxes1[0], bboxes2[0])
    y1_inter = max(bboxes1[1], bboxes2[1])
    x2_inter = min(bboxes1[2], bboxes2[2])
    y2_inter = min(bboxes1[3], bboxes2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union area
    area1 = bbox_area(bboxes1)
    area2 = bbox_area(bboxes2)
    
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    
    # Calculate the center distance
    c1_x, c1_y = (bboxes1[:2] + bboxes1[2:]) / 2
    c2_x, c2_y = (bboxes2[:2] + bboxes2[2:]) / 2
    c_dist = (c1_x - c2_x)**2 + (c1_y - c2_y)**2
    
    # Calculate aspect ratio term
    w1, h1 = bboxes1[2] - bboxes1[0], bboxes1[3] - bboxes1[1]
    w2, h2 = bboxes2[2] - bboxes2[0], bboxes2[3] - bboxes2[1]
    ar = 4 / (np.pi ** 2) * ((np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2)
    
    # Enclosing box
    x1_enc = min(bboxes1[0], bboxes2[0])
    y1_enc = min(bboxes1[1], bboxes2[1])
    x2_enc = max(bboxes1[2], bboxes2[2])
    y2_enc = max(bboxes1[3], bboxes2[3])
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
    
    # Calculate CIoU
    ciou = iou - (c_dist / enc_area + ar)

    return ciou

def giou(bboxes1, bboxes2):
    """
    Calculate GIoU between two bounding boxes
    bboxes1, bboxes2: [x1, y1, x2, y2]
    """

    # Calculate intersection area
    x1_inter = max(bboxes1[0], bboxes2[0])
    y1_inter = max(bboxes1[1], bboxes2[1])
    x2_inter = min(bboxes1[2], bboxes2[2])
    y2_inter = min(bboxes1[3], bboxes2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union area
    area1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    area2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0

    # Enclosing box
    x1_enc = min(bboxes1[0], bboxes2[0])
    y1_enc = min(bboxes1[1], bboxes2[1])
    x2_enc = max(bboxes1[2], bboxes2[2])
    y2_enc = max(bboxes1[3], bboxes2[3])
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
    
    # Calculate GIoU
    giou = iou - (enc_area - union_area) / enc_area if enc_area != 0 else iou

    return giou






def batch_ciou(bboxes1, bboxes2):
    ciou_values = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            ciou_values[i, j] = giou(bbox1, bbox2)
    return ciou_values

def optimal_assignment(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind




def bbox_area_batch_loss(bboxes):
    return (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])

def ciou_loss_batch(bboxes1, bboxes2):
    """
    Calculate batched CIoU loss between two sets of bounding boxes
    bboxes1, bboxes2: shape (batch_size, 4), where 4 is [x1, y1, x2, y2]
    """

    # Calculate intersection area
    x1_inter = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    y1_inter = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    x2_inter = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    y2_inter = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

    # Calculate union area
    area1 = bbox_area_batch_loss(bboxes1)
    area2 = bbox_area_batch_loss(bboxes2)
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    # Calculate the center distances
    c1 = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
    c2 = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
    c_dist = np.sum((c1 - c2) ** 2, axis=-1)

    # Calculate aspect ratio term
    w1, h1 = bboxes1[..., 2] - bboxes1[..., 0], bboxes1[..., 3] - bboxes1[..., 1]
    w2, h2 = bboxes2[..., 2] - bboxes2[..., 0], bboxes2[..., 3] - bboxes2[..., 1]
    ar = 4 / (np.pi ** 2) * ((np.arctan(w1 / (h1 + 1e-6)) - np.arctan(w2 / (h2 + 1e-6))) ** 2)

    # Enclosing box
    x1_enc = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    y1_enc = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    x2_enc = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    y2_enc = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    # Calculate CIoU
    ciou = iou - (c_dist / (enc_area + 1e-6) + ar)

    # Calculate CIoU loss
    ciou_loss = 1 - ciou
    aggregated_ciou_loss = np.mean(ciou_loss)
    return aggregated_ciou_loss


import cv2, os

def cxcywh_to_x1y1x2y2(boxes):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


# Calculate the area of the boxes
def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def gggiou_loss(boxes1, boxes2):
    boxes1 = cxcywh_to_x1y1x2y2(boxes1)
    boxes2 = cxcywh_to_x1y1x2y2(boxes2)


    area1 = area(boxes1)
    area2 = area(boxes2)

    # Find the intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Find the union
    union_area = area1 + area2 - inter_area

    # Calculate the Intersection over Union
    iou = inter_area / union_area

    # Find the smallest enclosing box
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

    # Calculate the Generalized Intersection over Union
    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)

    # GIoU loss is 1 - GIoU
    return 1 - giou



vvv = 0
def custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_len_of_preds, batch_z, max_one_hot):
    print(pred_boxes.shape)
    print(pred_labels.shape)

    
    image = cv2.imread(r"C:\Users\felhasznalo\Desktop\tracker\dataset\0000f77c-6257be58\frames\0000f77c-6257be58-0000001.jpg")
    image = cv2.resize(image, (1333,800))
    # Get image dimensions
    height, width, _ = image.shape
    for box, label in zip(pred_boxes.detach()[0], pred_labels.detach()[0]):
        print(label)
        x_center, y_center, box_width, box_height = box
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        # Convert to top-left x, y
        x1 = int(x_center - (box_width / 2))
        y1 = int(y_center - (box_height / 2))
        x2 = int(x_center + (box_width / 2))
        y2 = int(y_center + (box_height / 2))
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)
        font_thickness = 2
        cv2.putText(image, str(label), (x1, y1 - 10), font, font_scale, font_color, font_thickness)


    # Show image
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.show()
    exit()
    
    initial_tensor_pred = torch.empty((0, 4)).to(device)
    initial_tensor_gt = torch.empty((0, 4)).to(device)

    initial_tensor_pred_label = torch.empty((0, 2)).to(device)
    initial_tensor_gt_label = torch.empty((0, 1)).to(device)

    for gt_box, pred_box, pred_label in zip(gt_boxes, pred_boxes, pred_labels):

        gt_box = gt_box[torch.any(gt_box != 0, dim=1)]
        #pred_box = pred_box[torch.any(pred_box != 0, dim=1)]



        converter.to_corner_coordinates_batch(gt_box.cpu().detach().numpy())
        ciou_values = batch_ciou(converter.to_corner_coordinates_batch(gt_box.clone().cpu().detach().numpy()), converter.to_corner_coordinates_batch(pred_box.clone().cpu().detach().numpy()))
        #print(-ciou_values)
        row_ind, col_ind = optimal_assignment(-ciou_values)
 
        initial_tensor_gt = torch.cat((initial_tensor_gt, gt_box[row_ind]), 0)
        initial_tensor_pred = torch.cat((initial_tensor_pred, pred_box[col_ind]), 0)

        gt_label_helper = torch.zeros((max_len_of_preds, 1)).to(device)

        gt_label_helper[col_ind] = 1

        initial_tensor_gt_label = torch.cat((initial_tensor_gt_label, gt_label_helper), 0)

        initial_tensor_pred_label = torch.cat((initial_tensor_pred_label, pred_label), 0)


    box_loss = F.l1_loss(initial_tensor_pred, initial_tensor_gt, reduction='none').to(device)
    box_loss = box_loss.sum() / len(initial_tensor_gt)

    #initial_tensor_gt_label = torch.where(initial_tensor_gt_label == 0, torch.tensor(1, device=device), initial_tensor_gt_label)
    label_loss = F.cross_entropy(initial_tensor_pred_label.view(-1, 2), initial_tensor_gt_label.long().view(-1)).to(device)

    loss_giou = gggiou_loss(initial_tensor_pred.clone().cpu().detach(), initial_tensor_gt.clone().cpu().detach())
    loss_giou = loss_giou.sum() / len(initial_tensor_gt)

    loss = box_loss + label_loss + loss_giou

    return loss







"""
def custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_one_hot, batch_size):


    mask = gt_labels == 1
    #gt_classes = gt_classes.squeeze(2)

    #gt_classes = gt_classes.view(20, batch_size, 1).to(device)
    
    #print(gt_classes)

    mask_expanded = mask.expand(-1, -1, 4).to(device)
    #mask_expanded_classes = mask.clone().expand(-1, -1, max_one_hot).to(device)
    #mask_expanded_classes_gt = mask.clone().expand(-1, -1, 1).to(device)


    gt_boxes = gt_boxes[mask_expanded].view(-1, 4).to(device)

    #######pred_boxes = pred_boxes[mask_expanded].view(-1, 4).to(device)
    pred_boxes = pred_boxes.squeeze(dim=1)
    print(pred_boxes.shape)

    #print(gt_classes)
    # Compute the loss
    
    #gt_classes = gt_classes[mask_expanded_classes_gt].view(-1, 1).to(device)
    #gt_classes = gt_classes.view(-1).to(device)
    
    #non_zero_indices = gt_classes.nonzero(as_tuple=True)[0]
    #gt_classes = torch.index_select(gt_classes, 0, non_zero_indices)
    #print(gt_classes)
    #print(gt_classes)

    #pred_classes = pred_classes[mask_expanded_classes].view(-1, max_one_hot).to(device)

    #mask = gt_classes != 0
    #print(gt_classes.shape)
    #print(pred_classes.shape)
    #gt_classes = gt_classes[mask]
    #pred_classes = pred_classes[mask]
    #print(gt_classes.shape)
    #print(pred_classes.shape)

    
    box_loss = F.l1_loss(pred_boxes, gt_boxes).to(device)

    label_loss = F.cross_entropy(pred_labels.view(-1, 2), gt_labels.long().view(-1)).to(device)

    #class_loss = F.cross_entropy(pred_classes, gt_classes)

    loss = box_loss  + label_loss# + class_loss


    return loss

"""








import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def class_custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_one_hot, batch_size):

    """
    print(pred_boxes.shape)
    print(pred_labels.shape) 
    print(pred_classes.shape)
    print(gt_boxes.shape)
    print(gt_labels.shape)
    print(gt_classes.shape)
    
    print()
    """
    mask = gt_labels == 1


    gt_classes = gt_classes.view(10, batch_size, 1).to(device)


    mask_expanded = mask.expand(-1, -1, 4).to(device)
    mask_expanded_classes = mask.clone().expand(-1, -1, max_one_hot).to(device)
    mask_expanded_classes_gt = mask.clone().expand(-1, -1, 1).to(device)






    # Compute the loss
    gt_classes = gt_classes[mask_expanded_classes_gt].view(-1, 1).to(device)
    gt_classes = gt_classes.view(-1).to(device)

    #non_zero_indices = gt_classes.nonzero(as_tuple=True)[0]
    #gt_classes = torch.index_select(gt_classes, 0, non_zero_indices)
    #print(gt_classes)


    pred_classes = pred_classes[mask_expanded_classes].view(-1, max_one_hot).to(device)


    mask = gt_classes != 0
    gt_classes = gt_classes[mask]
    pred_classes = pred_classes[mask]



    class_loss = F.cross_entropy(pred_classes, gt_classes)

    loss = class_loss


    return loss





"""

def giou(boxA, boxB):
    cx1, cy1, w1, h1 = boxA
    cx2, cy2, w2, h2 = boxB
    
    x1A, y1A, x2A, y2A = cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2
    x1B, y1B, x2B, y2B = cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2
    
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (x2A - x1A) * (y2A - y1A)
    boxBArea = (x2B - x1B) * (y2B - y1B)

    unionArea = boxAArea + boxBArea - interArea

    iou = interArea / unionArea
    
    xC = min(x1A, x1B)
    yC = min(y1A, y1B)
    xD = max(x2A, x2B)
    yD = max(y2A, y2B)

    cArea = (xD - xC) * (yD - yC)
    giou = iou - (cArea - unionArea) / cArea
    
    return giou

"""


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):

    img_w, img_h = size

    b = box_cxcywh_to_xyxy(out_bbox)

    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)

    return b


def extract_elements_based_on_mask(tensor, mask):
    batch_size, seq_len, _ = tensor.shape
    extracted_elements = []

    for i in range(batch_size):
        for j in range(seq_len):
            if mask[i, j, 0] == 1.0:
                extracted_elements.append(tensor[i, j, :])

    return torch.stack(extracted_elements)

last_loss = None

def box_xyxy_to_cxcywh(x):
    x_min, y_min, x_max, y_max = x.unbind(1)
    b = [(x_min + x_max) / 2, (y_min + y_max) / 2,
         (x_max - x_min), (y_max - y_min)]
    return torch.stack(b, dim=1)

def box_cxcywh_to_xyxy(x):

    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def custom_lossAAA(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_len_of_preds, batch_z, max_one_hot):

    pred_boxes = extract_elements_based_on_mask(pred_boxes, gt_labels)
    gt_boxes = extract_elements_based_on_mask(gt_boxes, gt_labels)


    """
    pred_labels = extract_elements_based_on_mask(pred_labels, gt_labels).to(device)
    gt_labels = torch.full((pred_labels.shape[0],), 3, dtype=torch.long).to(device)

    """
    pred_labels = pred_labels.reshape(-1, 92).to(device)
    gt_labels = gt_labels.reshape(-1).long().to(device)
    gt_labels = torch.where(gt_labels == 1, torch.tensor(3, device='cuda:0'), gt_labels)
    print(pred_boxes)
    print(gt_boxes)

    box_loss = F.l1_loss(pred_boxes, gt_boxes).to(device)
    box_loss = box_loss.sum() / len(gt_boxes)

    label_loss = F.cross_entropy(pred_labels, gt_labels)
    

    loss_giou = 0
    for p, g in zip(pred_boxes, gt_boxes):
        loss_giou += (1 - giou(p.clone().cpu().detach().numpy(), g.clone().cpu().detach().numpy()))

    loss_giou = loss_giou / len(gt_boxes)

    loss = box_loss + label_loss# + loss_giou


    return loss


"""
def custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_len_of_preds, batch_z):

  

        
    #non_zero_indices = (gt_classes != 0).squeeze(-1).squeeze(-1).nonzero(as_tuple=False).squeeze(-1)


    #pred_boxes = pred_boxes[non_zero_indices]

    gt_boxes = gt_boxes.permute(1,0,2)
    #gt_labels = gt_labels.permute(1,0,2)
    pred_labels = pred_labels.permute(1,0,2)

    non_zero_rows = torch.any(gt_boxes != 0, dim=2)
    gt_boxes = gt_boxes[non_zero_rows]


    pred_boxes = pred_boxes.permute(1,0,2)
    gt_classes = gt_classes.permute(1,0,2)
    gt_labels = gt_labels.permute(1,0,2)






    
    #gt_labels = torch.zeros((100, 260, 1))
    gt_labels = torch.zeros((batch_z, max_len_of_preds, 1))


    non_zero_indices = torch.nonzero(gt_classes.squeeze(-1))
    #non_zero_indices = gt_classes[non_zero_indices]

    #print(non_zero_indices)
    #exit()
    # Üres lista az eredményeknek
    result_list = []

    label_list = []
    # Végigiterálok minden batch-en
    for batch_idx in range(pred_boxes.shape[0]):
        #indices_in_batch = non_zero_indices[non_zero_indices[:, 0] == batch_idx][:, 1]
        indices_in_batch = gt_classes.squeeze(-1)[batch_idx][gt_classes.squeeze(-1)[batch_idx] != 0]

        #print(indices_in_batch)
        selected = pred_boxes[batch_idx, indices_in_batch]

        result_list.append(selected)
        label_list.append(indices_in_batch.tolist())


    pred_boxes = torch.cat(result_list, dim=0)

    #print(gt_labels)
    #print(gt_labels.shape)

    #pred_boxes = pred_boxes.permute(1,0,2)

    

    for i, index_list in enumerate(label_list):
        gt_labels[i, index_list, 0] = 1


    #print(pred_boxes.shape)
    ######################################################################################mask = gt_labels == 1
    #mask_pred = gt_classes > 0

    #non_zero_indices = gt_classes.nonzero(as_tuple=True)[0]
    #non_zero_values = gt_classes[non_zero_indices].squeeze()


    # Kiválasztjuk a data_tensor megfelelő elemeit
    #pred_boxes = pred_boxes[non_zero_values, :, :].view(-1, 4).to(device)

    #gt_classes = gt_classes.squeeze(2)

    #gt_classes = gt_classes.view(20, batch_size, 1).to(device)
    
    #print(gt_classes)
    ######################################################################################mask_expanded = mask.expand(-1, -1, 4).to(device)
    #mask_expanded_pred = mask_pred.expand(-1, -1, 4).to(device)

    #mask_expanded_classes = mask.clone().expand(-1, -1, max_one_hot).to(device)
    #mask_expanded_classes_gt = mask.clone().expand(-1, -1, 1).to(device)

    ######################################################################################gt_boxes = gt_boxes[mask_expanded].view(-1, 4).to(device)
    #print(gt_boxes.shape)
    ##############pred_boxes = pred_boxes[mask_expanded].view(-1, 4).to(device)
    #pred_boxes = pred_boxes[mask_expanded_pred].view(-1, 4).to(device)

    #print(gt_classes)
    # Compute the loss
    
    #gt_classes = gt_classes[mask_expanded_classes_gt].view(-1, 1).to(device)
    #gt_classes = gt_classes.view(-1).to(device)
    
    #non_zero_indices = gt_classes.nonzero(as_tuple=True)[0]
    #gt_classes = torch.index_select(gt_classes, 0, non_zero_indices)
    #print(gt_classes)
    #print(gt_classes)

    #pred_classes = pred_classes[mask_expanded_classes].view(-1, max_one_hot).to(device)

    #mask = gt_classes != 0
    #print(gt_classes.shape)
    #print(pred_classes.shape)
    #gt_classes = gt_classes[mask]
    #pred_classes = pred_classes[mask]
    #print(gt_classes.shape)
    #print(pred_classes.shape)

   
    # Pozíciós kódolás hozzáadása
    #pred_boxes = add_positional_encoding(pred_boxes, scale_factor=0.01)

    box_loss = F.l1_loss(pred_boxes, gt_boxes).to(device)

    #label_loss = F.cross_entropy(pred_labels.view(-1, 2), gt_labels.long().view(-1)).to(device)

    label_loss = F.cross_entropy(pred_labels.reshape(-1, 2).to(device), gt_labels.long().reshape(-1).to(device)).to(device)


    #class_loss = F.cross_entropy(pred_classes, gt_classes)

    loss = box_loss  + label_loss# + class_loss


    return loss
"""