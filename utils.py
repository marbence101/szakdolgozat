import os
import cv2
import math
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from coordinate import CoordinateConverter

import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

from normalizer import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

converter = CoordinateConverter()

def positionalencoding2d(d_model, height, width):
    pe = torch.zeros(d_model, height, width, device=device)
    y_pos = torch.arange(0., height, device=device).unsqueeze(1).expand(height, width)
    x_pos = torch.arange(0., width, device=device).unsqueeze(0).expand(height, width)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=device) *-(math.log(10000.0) / d_model))

    pe[0::2, :, :] = torch.sin(y_pos * div_term.view(-1, 1, 1)).unsqueeze(0)
    pe[1::2, :, :] = torch.cos(x_pos * div_term.view(-1, 1, 1)).unsqueeze(0)

    return pe

def show_batch(image, bounding_boxes, class_ids):
    scaler = MinMaxScaler()
    labels = scaler.de_normalize(bounding_boxes)
    #image_np = image.numpy()
    image_np = image.cpu().numpy()

    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    #for label, id in zip(labels, class_ids):
    for label in labels:
        #label = label.detach().numpy()

        label = label.cpu().detach().numpy()





        cx, cy, w, h = map(int, label) #######
        #x1, y1, w, h = map(int,(label))
        x1, y1, w, h = converter.to_corner_coordinates(cx, cy, w, h) #####






        #cv2.putText(image_np, str(id.item()), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), (255, 0, 0), 1)

    plt.imshow(image_np)
    plt.show()
    plt.close()




def one_hot_encode(tensor, num_classes):
    # One-hot tensor létrehozása
    one_hot = torch.zeros(tensor.size(0), num_classes).to(tensor.device).long()
    
    # Az osztályokat jelző indexeket 1-re állítja
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)
    
    return one_hot





def tensor_to_image(tensor):
    tensor = tensor.clone().to(device)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).to(device) 
    tensor *= torch.tensor([0.229], device=device).view(1, 1, 1)
    tensor += torch.tensor([0.485], device=device).view(1, 1, 1)
    tensor = tensor.clip(0, 1)

    to_pil = ToPILImage()
    image = to_pil(tensor.cpu()) # Konvertálás vissza CPU-ra a PIL képhez

    return image


def show(tensor, box_coords, classes, ids, path):
    scaler = MinMaxScaler()
    probabilities = F.softmax(classes, dim=1)

    predicted_classes = torch.argmax(probabilities, dim=1)

    image = tensor_to_image(tensor)
    image_np = np.array(image)


    for i, (box, id) in enumerate(zip(box_coords, ids)):
    #for i, box in enumerate(box_coords):
        probabilities = F.softmax(id, dim=1)

        id = torch.argmax(probabilities, dim=1)

        box = scaler.de_normalize(box)





        cx, cy, w, h = map(int, box) #######
        #x1, y1, w, h = map(int, box)
        x1, y1, w, h = converter.to_corner_coordinates(cx, cy, w, h) #####









        #print(x1, y1, w, h)
        #image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), (255, 0, 0), 5)
        if predicted_classes[i] == 1:
            #cv2.putText(image_np, str(id.item()), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), (255, 0, 0), 1)

    file_name = os.path.join(path, '{}.png'.format(str(time.time())))
    cv2.imwrite(file_name, image_np)

def show_a_random_data(data_loader):
    for i, (image_batch, labels_batch) in enumerate(data_loader):
        #print(labels_batch)
        # Az első kép bounding boxai és osztályazonosítói
        bounding_boxes = labels_batch['bounding_boxes'][0]
        class_ids = labels_batch['class_ids'][0]
        
        # show_batch függvény hívása (feltételezem, hogy ez már megvan)
        show_batch(image_batch[0].clone(), bounding_boxes.clone(), class_ids.clone())

        break


def preprocess_a_custom_image(image_path, train_with_grayscale, image_size):
    test_image = None
    if train_with_grayscale:
        test_image = Image.open(image_path).convert('L')
    else:
        test_image = Image.open(image_path).convert('RGB')

    test_image = test_image.resize(image_size)

    test_image = ToTensor()(test_image).to(device)

    return test_image.unsqueeze(0)

def load_data(root_dir):
    images_path = []
    labels = []

    for annotation in ['train_gt.txt']:
        folder_path = annotation.split('_')[0]
        annotation_path = os.path.join(root_dir, 'annotations', annotation)
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            sub_labels = []

            i = 0
            while i < len(lines):
                
                if lines[i].__contains__('.jpg'):
                    lines[i] = lines[i].strip()
                    lines[i] = lines[i].replace('/', '_')
                    images_path.append(os.path.join(root_dir, folder_path, lines[i]))

                    i += 2

                while True:
                    numbers = lines[i].split()
                    x1, y1, w, h = map(int, numbers[:4])
                    sub_labels.append([x1, y1, w, h])
                    try:
                        if lines[i+1].__contains__('.jpg'):
                            break
                    except:
                        break

                    else:
                        i += 1
                i += 1

                labels.append(sub_labels)
                sub_labels = []
    labels = [sublist for sublist in labels if sublist]

    return images_path, labels



def ciou(b1, b2):
    """
    Calculate CIoU between two bounding boxes.
    Args:
        b1: tensor with first bounding box (x1, y1, w, h)
        b2: tensor with second bounding box (x1, y1, w, h)
    Returns:
        ciou: Complete Intersection over Union
    """
    
    # Transform from (x1, y1, w, h) to (x1, y1, x2, y2)
    b1_x2, b1_y2 = b1[..., 0] + b1[..., 2], b1[..., 1] + b1[..., 3]
    b2_x2, b2_y2 = b2[..., 0] + b2[..., 2], b2[..., 1] + b2[..., 3]
    
    # Intersection area
    intersect_x1 = torch.max(b1[..., 0], b2[..., 0])
    intersect_y1 = torch.max(b1[..., 1], b2[..., 1])
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)
    intersect_w = torch.max(intersect_x2 - intersect_x1, torch.tensor(0.0))
    intersect_h = torch.max(intersect_y2 - intersect_y1, torch.tensor(0.0))
    intersection = intersect_w * intersect_h
    
    # Union area
    b1_area = b1[..., 2] * b1[..., 3]
    b2_area = b2[..., 2] * b2[..., 3]
    union = b1_area + b2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    
    # Box centers
    b1_center = torch.tensor([(b1[..., 0] + b1_x2) / 2, (b1[..., 1] + b1_y2) / 2])
    b2_center = torch.tensor([(b2[..., 0] + b2_x2) / 2, (b2[..., 1] + b2_y2) / 2])
    
    # Center distance
    center_distance = torch.sum((b1_center - b2_center) ** 2)
    
    # Enclosing box
    enclose_x1 = torch.min(b1[..., 0], b2[..., 0])
    enclose_y1 = torch.min(b1[..., 1], b2[..., 1])
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    
    # Enclosing Diagonal - corrected from tuple to tensor sum
    enclose_diagonal = torch.sum(torch.tensor([enclose_w ** 2, enclose_h ** 2]))
    
    # Aspect ratio term
    v = 4 / (3.14159265 ** 2) * torch.pow((torch.atan(b1[..., 2] / (b1[..., 3] + 1e-6)) - torch.atan(b2[..., 2] / (b2[..., 3] + 1e-6))), 2)
    alpha = v / (1 - iou + v + 1e-6)
    
    # CIoU
    ciou_term = iou - (center_distance / (enclose_diagonal + 1e-6)) - alpha * v
    return ciou_term

def giou(b1, b2):
    """
    Calculate GIoU between two bounding boxes.
    Args:
        b1: tensor with first bounding box (x1, y1, w, h)
        b2: tensor with second bounding box (x1, y1, w, h)
    Returns:
        giou: Generalized Intersection over Union
    """
    
    # Transform from (x1, y1, w, h) to (x1, y1, x2, y2)
    b1_x2, b1_y2 = b1[..., 0] + b1[..., 2], b1[..., 1] + b1[..., 3]
    b2_x2, b2_y2 = b2[..., 0] + b2[..., 2], b2[..., 1] + b2[..., 3]
    
    # Intersection area
    intersect_x1 = torch.max(b1[..., 0], b2[..., 0])
    intersect_y1 = torch.max(b1[..., 1], b2[..., 1])
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)
    intersect_w = torch.max(intersect_x2 - intersect_x1, torch.tensor(0.0))
    intersect_h = torch.max(intersect_y2 - intersect_y1, torch.tensor(0.0))
    intersection = intersect_w * intersect_h
    
    # Union area
    b1_area = b1[..., 2] * b1[..., 3]
    b2_area = b2[..., 2] * b2[..., 3]
    union = b1_area + b2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    
    # Enclosing box
    enclose_x1 = torch.min(b1[..., 0], b2[..., 0])
    enclose_y1 = torch.min(b1[..., 1], b2[..., 1])
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    
    # Enclosing area
    enclose_area = enclose_w * enclose_h
    
    # GIoU
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
    
    return giou


