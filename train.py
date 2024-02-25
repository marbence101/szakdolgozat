import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
import os
import re
import torch
import warnings
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import utils
from detr import DETR
from loss import custom_loss, class_custom_loss
from preprocess import ImagesDataset
from models import build
#from src.trackformer.models import build_model
from torchvision.transforms import Compose, RandomRotation, GaussianBlur

warnings.filterwarnings("ignore")

#-------------------------Params-------------------------#

root_dir = r"C:\Users\felhasznalo\Desktop\tracker\dataset"
max_len_of_preds = 100
class_ids_num = max_len_of_preds
image_size = (480, 480)
train_with_grayscale = False
to_save_model = True
shuffle_data = False
batch_size = 30#30 #allitsd a masikat is a detr ben
video_frames = 30#10 # 10 és 32x3
epochs = 500
num_classes = 1#1
hidden_dim = 256
nheads = 8
num_encoder_layers = 6
num_decoder_layers = 6
freeze_backbone = False
lr=1e-4
dim_feedforward = 2048
dropout = 0.1
videos_num_to_train = 450#300#100 #1200
max_one_hot = 1
to_save_images = True
do_augmentation = False
augmentation_prob = 0.2
pretained_detr = True


num_classes += 1
class_ids_num += 1 





import argparse




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser




#-------------------------Params-------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #images_path, labels = utils.load_data(root_dir)





    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def read_video_data(video_folder_path):
        frames_folder_path = os.path.join(video_folder_path, 'frames')
        images_path = []
        labels = {}
        
        for filename in sorted(os.listdir(frames_folder_path), key=natural_keys):
            if filename.endswith('.jpg'):
                full_path = os.path.join(frames_folder_path, filename)
                images_path.append(full_path)

        gt_file_path = os.path.join(video_folder_path, 'gt.txt')
        with open(gt_file_path, 'r') as f:
            for line in f.readlines():
                frame_number, object_id, x, y, w, h, *_ = map(float, line.strip().split(","))
                frame_number = int(frame_number) - 1
                
                if frame_number not in labels:
                    labels[frame_number] = []
                labels[frame_number].append([[int(x), int(y), int(w), int(h)], int(object_id)])

        sorted_labels = [labels[i] if i in labels else [] for i in range(len(images_path))]
        return images_path, sorted_labels








    dataset_folder = 'dataset'
    videos = os.listdir(os.path.join(root_dir))
    videos = videos[:videos_num_to_train]

    all_images_path = []
    all_labels = []

    for video in videos:
        video_folder_path = os.path.join(dataset_folder, video)
        images_path, labels = read_video_data(video_folder_path)

        all_images_path.append(images_path)
        all_labels.append(labels)

    # Interleave the image paths and labels
    #interleaved_images_path = [img for sublist in zip(*all_images_path) for img in sublist]
    #interleaved_labels = [label for sublist in zip(*all_labels) for label in sublist]


    interleaved_images_path = []
    interleaved_labels = []

    num_videos = len(all_images_path) #900
    frames_per_video = len(all_images_path[0]) #30

    from_idx = 0
    to_idx = batch_size

    for i in range(int(videos_num_to_train/batch_size)):
        for j in range(video_frames):
            for k in range(from_idx, to_idx):
                interleaved_images_path.append(all_images_path[k][j])
                interleaved_labels.append(all_labels[k][j])
        if j == video_frames-1:
            from_idx += batch_size
            to_idx += batch_size 





    interleaved_images_path
    interleaved_labels

    images_path = []
    labels = {}

    #print("All image paths:")
    #print(interleaved_images_path)
    #print(len(interleaved_images_path))
    images_path = interleaved_images_path
    #print("All labels:")
    #print(interleaved_labels)
    #print(len(interleaved_labels))
    labels = interleaved_labels












    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def read_video_data(video_folder_path):
        frames_folder_path = os.path.join(video_folder_path, 'frames')
        images_path = []
        labels = {}
        
        # Collect image paths
        for filename in sorted(os.listdir(frames_folder_path), key=natural_keys):
            if filename.endswith('.jpg'):
                full_path = os.path.join(frames_folder_path, filename)
                images_path.append(full_path)

        # Read ground truth labels
        gt_file_path = os.path.join(video_folder_path, 'gt.txt')
        with open(gt_file_path, 'r') as f:

            for line in f.readlines():
                frame_number, object_id, x, y, w, h, *_ = map(float, line.strip().split(","))
                frame_number = int(frame_number) - 1
                
                if frame_number not in labels:
                    labels[frame_number] = []

                labels[frame_number].append([[int(x), int(y), int(w), int(h)], int(object_id)])


        # Sort labels by image order

        sorted_labels = [labels[i] if i in labels else [] for i in range(len(images_path))]
        #sorted_labels = [labels[label] for label in labels]
        #print(sorted_labels)
        return images_path, sorted_labels

    dataset_folder = 'dataset'
    videos = os.listdir(os.path.join(root_dir))
    videos = videos[:videos_num_to_train]
    all_images_path = []
    all_labels = []

    for video in videos:
        video_folder_path = os.path.join(dataset_folder, video)
        images_path, labels = read_video_data(video_folder_path)
        
        all_images_path.extend(images_path)
        all_labels.extend(labels)

    images_path = []
    labels = {}

    #print("All image paths:")
    #print(all_images_path)
    #print(len(all_images_path))
    images_path = all_images_path
    #print("All labels:")
    #print(all_labels[100:105])
    #print(len(all_labels))
    labels = all_labels
    """

    """


    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def read_video_data(video_folder_path):
        frames_folder_path = os.path.join(video_folder_path, 'frames')
        images_path = []
        labels = {}
        
        for filename in sorted(os.listdir(frames_folder_path), key=natural_keys):
            if filename.endswith('.jpg'):
                full_path = os.path.join(frames_folder_path, filename)
                images_path.append(full_path)

        gt_file_path = os.path.join(video_folder_path, 'gt.txt')
        with open(gt_file_path, 'r') as f:
            for line in f.readlines():
                frame_number, object_id, x, y, w, h, *_ = map(float, line.strip().split(","))
                frame_number = int(frame_number) - 1
                
                if frame_number not in labels:
                    labels[frame_number] = []
                labels[frame_number].append([[int(x), int(y), int(w), int(h)], int(object_id)])

        sorted_labels = [labels[i] if i in labels else [] for i in range(len(images_path))]
        return images_path, sorted_labels

    dataset_folder = 'dataset'
    videos = os.listdir(os.path.join(root_dir))
    videos = videos[:videos_num_to_train]

    all_images_path = []
    all_labels = []

    for video in videos:
        video_folder_path = os.path.join(dataset_folder, video)
        images_path, labels = read_video_data(video_folder_path)
        
        all_images_path.append(images_path)
        all_labels.append(labels)

    # Interleave the image paths and labels
    interleaved_images_path = [img for sublist in zip(*all_images_path) for img in sublist]
    interleaved_labels = [label for sublist in zip(*all_labels) for label in sublist]

    interleaved_images_path
    interleaved_labels

    images_path = []
    labels = {}

    #print("All image paths:")
    #print(interleaved_images_path)
    #print(len(interleaved_images_path))
    images_path = interleaved_images_path
    #print("All labels:")
    #print(interleaved_labels)
    #print(len(interleaved_labels))
    labels = interleaved_labels
    """

    with open('readme_path.txt', 'w') as f:
        f.write(str(images_path))
    #print(images_path)
    #print(labels)
    

    
    def custom_collate(batch):
        images, labels = zip(*batch)

        # Ensure all images have the same number of channels (convert to grayscale, if necessary)
        images = [img if img.shape[0] == 1 else torch.mean(img, dim=0, keepdim=True) for img in images]
        
        # Now stack into tensors
        images = torch.stack(images, 0)

        if all(isinstance(label, torch.Tensor) for label in labels):
            labels = torch.stack(labels, 0)
        elif all(isinstance(label, dict) for label in labels):
            # Handle the case where labels are dictionaries.
            keys = labels[0].keys()
            labels = {key: torch.stack([label[key] for label in labels]) for key in keys}
        else:
            raise TypeError("Inconsistent label types in batch.")

        return images, labels
    
    dataset = ImagesDataset(images_path, labels, max_len_of_preds, image_size, train_with_grayscale, do_augmentation, augmentation_prob, device=device)
    if train_with_grayscale:
        data_loader = DataLoader(dataset, batch_size, shuffle=shuffle_data, collate_fn=custom_collate)
    else:
        data_loader = DataLoader(dataset, batch_size, shuffle=shuffle_data)
    
    #utils.show_a_random_data(data_loader)

    #exit()



    def get_non_zero_masks(tensor, max_len):
            batch_size, seq_len, _ = tensor.shape
            masks = torch.zeros((batch_size, max_len, 1))
            
            non_zero_indices = (tensor.sum(dim=-1) != 0).nonzero()
            
            for i in range(len(non_zero_indices)):
                batch_idx, seq_idx = non_zero_indices[i]
                masks[batch_idx, seq_idx, 0] = 1.0

            return masks



    """
    model = DETR(num_classes=91).to(device)
    #model = DETR(num_classes, class_ids_num, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, max_len_of_preds, freeze_backbone, dim_feedforward, dropout, video_frames, batch_size, max_one_hot, train_with_grayscale)
    #model = torch.load('best_model.pth')
    model.to(device)
    
    state_dict = torch.load(r'h', map_location=device)
    #model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, strict = False)
    #model.eval()
    """
    

    """
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=pretained_detr).to(device)
    num_classes = 1  # Az új adathalmazban lévő osztályok száma
    model.class_embed = nn.Linear(in_features=256, out_features=num_classes + 1)
    model = model.to(device)
    """
    """


    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=1).to(device)
    checkpoint = torch.hub.load_state_dict_from_url(
                url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                map_location='cpu',
                check_hash=True)
    model.to(device)
    #del checkpoint["model"]["class_embed.weight"]
    #del checkpoint["model"]["class_embed.bias"]
    checkpoint["model"]["class_embed.weight"] = torch.randn((2, 256), device=device)
    checkpoint["model"]["class_embed.bias"] = torch.randn(2, device=device)


    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)


    """


    #model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
    """
    num_classes = 1  # You have only one class

    # Get pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
        map_location='cpu',
        check_hash=True)

    # Remove class weights
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # Modify the positional embedding size
    # If you're using a different number of queries, you also need to resize this
    num_queries = checkpoint["model"]["query_embed.weight"].shape[0]
    checkpoint["model"]["query_embed.weight"] = torch.zeros(num_queries, num_classes + 1)

    # Initialize the weights for the single class
    weight = torch.zeros(num_classes + 1, 256)  # Assuming the hidden dimension is 256
    bias = torch.zeros(num_classes + 1)
    torch.nn.init.normal_(weight, std=0.01)
    torch.nn.init.constant_(bias, 0)

    # Set the modified weights
    checkpoint["model"]["class_embed.weight"] = weight
    checkpoint["model"]["class_embed.bias"] = bias

    # Save the modified checkpoint
    torch.save(checkpoint, 'detr-r50_no-class-head.pth')
    """

    
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()



    model, criterion, _ = build(args)
    ####model, criterion, _ = build_model(args)
    model.to(device)
    
    checkpoint = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',map_location='cpu',check_hash=True)
    
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]
    model.load_state_dict(checkpoint["model"], strict=False)

    


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    best_loss = float('inf')
    losses = []
    


    #scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(epochs):

        epoch_losses = []

        model.train()





        if epoch % 1 == 0:
            print("{}. epoch".format(epoch+1))

            

        for i, (images, gts) in enumerate(data_loader):

            images = images.to(device)

            ########
            """for j in range(images.shape[0]):  # Az images.shape[0] a batch mérete (pl. 30)
                img = images[j].cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(img, (1, 2, 0)))  # Csatornák rendezése, ha szükséges
                plt.title(f"Image {j+1} in batch {i+1}")
                plt.show()"""
            #########



            #torch.Size([100, 1, 225, 224])

            
            gt_boxes = gts['bounding_boxes'].to(device)
            gt_class_ids = gts['class_ids'].to(device)

            #print(gt_class_ids.shape)
            #print(gt_class_ids[gt_class_ids > 0])
            gt_labels = get_non_zero_masks(gt_boxes, gt_boxes.shape[1])


            gt_classes = None

  
            ###preds = model(images)

            criterion.train()


            #############################################################################

         


            outputs = model(images)

            gt_labels = gt_labels.to(dtype=torch.long)




            targets = []
            
            for actual_index, gt_box in enumerate(gt_boxes):
                gt_box = gt_box[torch.any(gt_box != 0, dim=1)]
                targets.append({
                    "labels": torch.ones(len(gt_box), dtype=torch.long).to(device),
                    "boxes": gt_box.to(device),
                    "classes" : gt_class_ids[actual_index][gt_class_ids[actual_index] > 0].to(device)
                })

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            #pred_boxes = preds[0]
            #pred_labels = preds[1]
            pred_classes = None

            ###pred_boxes = preds["pred_boxes"]
            ###pred_labels = preds["pred_logits"]

            
            #loss = custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_one_hot, batch_size)
            ###loss = custom_loss(pred_boxes, pred_labels, pred_classes, gt_boxes, gt_labels, gt_classes, max_len_of_preds, batch_size, max_one_hot)

            #loss = torch.tensor(1., requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                print("Loss: (", i, "/", len(data_loader),"):   ", loss.item())
            
            epoch_losses.append(loss.item())
            """
            losses.append(loss.item())

            if to_save_model:
                if loss.item() < best_loss:
                    best_loss = loss.item()

                    #torch.save(model, 'best_model.pth')
                    torch.save(model.state_dict(), 'best_model.pth')
                    print('Model saved with Loss: {:.4f}'.format(best_loss))"""

        epoch_loss = sum(epoch_losses)/len(epoch_losses)
        epoch_losses.append(epoch_loss)
        print("EPOCH LOSS: ", epoch_loss)

        if to_save_model:
            if epoch_loss < best_loss:
                best_loss = epoch_loss

                #torch.save(model, 'best_model.pth')
                torch.save(model.state_dict(), 'best_model.pth')
                print('Model saved with Loss: {:.4f}'.format(best_loss))

        lr_scheduler.step()
    plt.plot(epoch_losses)   
    plt.show()     

if __name__ == "__main__":
    main()

