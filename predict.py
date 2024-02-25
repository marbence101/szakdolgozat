import torch
import torch.nn as nn
from detr import DETR
from torchvision.transforms import ToTensor
from PIL import Image
import os, re
import cv2
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

from sklearn.cluster import DBSCAN


import cv2
import torchvision
import warnings
import torch.nn as nn
import math
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment





import os
import glob


directory_path = r"C:\Users\felhasznalo\Desktop\tracker\saved_images"


jpg_files = glob.glob(os.path.join(directory_path, "*.jpg"))


for jpg_file in jpg_files:
    try:
        os.remove(jpg_file)
    except Exception as e:
        print(f"Hiba történt {jpg_file} törlése közben: {e}")








use_eval = True
CCC_threshold = 0.97#0.95
#CCC_threshold = 1 - CCC_threshold
only_car_class = False
#PP



file_path = r"C:\Users\felhasznalo\Desktop\tracker\evaluation\det.txt"  # A fájl elérési útja

# Ellenőrizzük, hogy a fájl létezik-e és van-e benne tartalom
try:
    with open(file_path, 'r') as file:
        # Ha a fájl nem üres, törljük a tartalmát
        if file.read().strip():
            # Újranyitjuk a fájlt írás módban, ami törli a tartalmát
            with open(file_path, 'w') as file:
                print("A fájl tartalma törölve.")
        else:
            print("A fájl már üres.")
except FileNotFoundError:
    print("A fájl nem található.")





from torchvision.transforms import ToPILImage
from normalizer import MinMaxScaler
from coordinate import CoordinateConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = (480, 480)

converter = CoordinateConverter()

img_counter = 1

def preprocess_a_custom_image(image_array):
    #test_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    test_image = Image.fromarray(image_array)

    test_image = test_image.resize(image_size)

    test_image = ToTensor()(test_image).to(device)

    return test_image.unsqueeze(0)




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

output_dir = "saved_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
counter = 1


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def read_and_resize_images(directory):
    image_files = os.listdir(directory)
    image_files.sort(key=natural_keys)
    
    for image_file in image_files:
        img_path = os.path.join(directory, image_file)
        image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #image_np = cv2.imread(img_path)
        
        if image_np is not None:
            resized_image = cv2.resize(image_np, image_size)
            yield resized_image

def get_color_by_id(id):
    # List of static colors
    color_list = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue

        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255),  # Deep Sky Blue
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (0, 255, 255), # Cyan
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Red
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Dark Blue
        (128, 128, 0), # Olive
        (128, 128, 128), # Grey
        (0, 128, 128), # Teal
        (128, 0, 128), # Purple
        (255, 165, 0), # Orange
        (255, 192, 203), # Pink
        (64, 224, 208), # Turquoise
        (75, 0, 130),  # Indigo
        (255, 69, 0),  # Red-Orange
        (34, 139, 34), # Forest Green
        (218, 165, 32), # Goldenrod
        (173, 216, 230), # Light Blue
        (240, 128, 128), # Light Coral
        (0, 100, 0),   # Dark Green
        (128, 0, 0),   # Maroon
        (100, 149, 237), # Cornflower Blue
        (189, 183, 107), # Dark Khaki
        (85, 107, 47), # Dark Olive Green
        (72, 61, 139), # Dark Slate Blue
        (139, 69, 19), # Saddle Brown
        (0, 191, 255)  # Deep Sky Blue
        
    ]

    # Return the color corresponding to the id
    return color_list[id]

import torchvision.transforms as T
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    
    img = transform(im).unsqueeze(0)
    img = img.to(device)


    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #print(outputs['pred_logits'].softmax(-1))
    #probas = outputs['pred_logits'].softmax(-1)[:, :, 1].reshape(-1, 1)
    #print(outputs['pred_logits'].softmax(-1)[:, :, 1].reshape(-1, 1))
    global CCC_threshold

    keep = probas.max(-1).values > CCC_threshold
    # convert boxes from [0; 1] to image scales
    true_indices = torch.nonzero(keep).squeeze().cpu().tolist()
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    
    return probas[keep], bboxes_scaled, true_indices

def find_missing_numbers(lists):
    missing_numbers = []
    previous_numbers = set()

    for current_numbers in lists:
        current_set = set(current_numbers)
        for num in previous_numbers:
            if num not in current_set:
                missing_numbers.append(num)
                
        previous_numbers = current_set

    return list(set(missing_numbers))

ids = list()
#archin belülre rakni
from torchvision.ops import nms
lines_to_write = []
def show(box_coords, classes, counter, image_gen):
    #class_ids = class_ids.squeeze(1)
    #class_predicted_classes = torch.argmax(class_ids, dim=1)

    inner_list = list()
    scaler = MinMaxScaler()
    probabilities = F.softmax(classes, dim=1)
    threshold = 0.9
    j = 0
    predicted_classes = torch.argmax(probabilities, dim=1)
    image_np = next(image_gen)
    j_list = set()

    exit()

    for i, box in enumerate(box_coords):


        box = scaler.de_normalize(box)
        
        if probabilities[i, predicted_classes[i]] >= threshold:

            cx, cy, w, h = map(int, box)
            x1, y1, w, h = converter.to_corner_coordinates(cx, cy, w, h)

            if predicted_classes[i] > -1:
                
                color = (0, 0, 0)
                """
                if i in find_missing_numbers(ids):
                    j = i
                    while j in find_missing_numbers(ids) or j in j_list or j == i+1:
                        j += 1
                    cv2.putText(image_np, str(j), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    inner_list.append(j)
                    j_list.add(j)
                else:
                    cv2.putText(image_np, str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    inner_list.append(i)
                image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), color, 2)

                global lines_to_write
                #1280x720 -> 224x224
                x_to_multiply = 1280 / 224
                y_to_multiply = 720 / 224
                lines_to_write.append([counter-1, inner_list[-1], x1*x_to_multiply, y1*y_to_multiply, w*x_to_multiply, h*y_to_multiply, 1, -1, -1, -1])


                """

                

                cv2.putText(image_np, str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                inner_list.append(i)
                image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), color, 2)


                global lines_to_write
                #1280x720 -> 224x224
                x_to_multiply = 1280 / 224
                y_to_multiply = 720 / 224
                lines_to_write.append([counter-1, inner_list[-1], x1*x_to_multiply, y1*y_to_multiply, w*x_to_multiply, h*y_to_multiply, 1, -1, -1, -1])

    ids.append(inner_list)
    #print(find_missing_numbers(ids))
    plt.imshow(image_np, cmap='gray')
    file_name = f"{output_dir}/image_epoch_loss_{counter}.png"
    plt.savefig(file_name)
    plt.close()





# COCO classes

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

#CLASSES = ['N/A', 'car']


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
from scipy.optimize import linear_sum_assignment
import numpy as np

# colors for visualization
COLORS = [[0.000, 0.0, 0.0]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
#model = torch.load('best_model.pth').to(device)

#model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False).to(device)

#model.load_state_dict(torch.load('best_model.pth'))







import torch.nn as nn
"""
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False).to(device)

num_classes = 1  # Az új adathalmazban lévő osztályok száma
model.class_embed = nn.Linear(in_features=256, out_features=num_classes + 1)
model = model.to(device)

model.load_state_dict(torch.load('best_model.pth'))
"""

"""

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=1).to(device)
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)
model.to(device)
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
"""
"""
# A modell újra létrehozása
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False).to(device)

# Súlyok betöltése
model.load_state_dict(torch.load('best_model.pth'))

# Modell áthelyezése a megfelelő eszközre
model.to(device)
"""


"""
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=1)

checkpoint = torch.load('best_model.pth',map_location='cpu')

model.load_state_dict(checkpoint,strict=False)
model.to(device)
"""

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2).to(device)
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)

del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]

checkpoint = torch.load(r'C:\Users\felhasznalo\Desktop\tracker\saved_mm\sima\best_model.pth',map_location='cpu')
model.load_state_dict(checkpoint,strict=False)
#model.load_state_dict(checkpoint["model"], strict=False)







if use_eval:
    model.eval()
else:
    model.train()

objects_to_track = ['motorcycle', 'car', 'bus', 'truck']
objects_to_track = ['car']

#nms

previous_boxes = torch.empty((0, 4)).to(device)
actual_ids = []

prev_ids = []


from torchvision.ops import generalized_box_iou
import torch




previous_boxes = torch.empty((0, 4)).to(device)
prev_ids = []
next_id = 1  # Start from 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mot_file = open("det.txt", "w")
previous_boxes = torch.empty((0, 4)).to(device)
prev_ids = []
next_id = 1  # Start from 1
fff = 1
def plot_results(pil_img, prob, boxes, save_path):
    global previous_boxes, prev_ids, next_id
    global fff
    actual_boxes = torch.empty((0, 4)).to(device)
    actual_ids = []

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        cl = p.argmax()
        #if CLASSES[cl] in objects_to_track:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        actual_boxes = torch.cat((actual_boxes, torch.tensor([xmin, ymin, xmax, ymax]).to(device).unsqueeze(0)), 0)

    if actual_boxes.shape[0] > 0:
        if previous_boxes.shape[0] > 0:
            iou_matrix = -generalized_box_iou(previous_boxes, actual_boxes)
            row_indices, col_indices = linear_sum_assignment(iou_matrix.cpu().numpy())

            for col in range(actual_boxes.shape[0]):
                if col in col_indices:
                    row = row_indices[list(col_indices).index(col)]
                    if -iou_matrix[row, col] > 0.1:
                        actual_ids.append(prev_ids[row])
                    else:
                        actual_ids.append(next_id)
                        next_id += 1
                else:
                    actual_ids.append(next_id)
                    next_id += 1
        else:
            actual_ids = list(range(next_id, next_id + len(actual_boxes)))
            next_id += len(actual_boxes)

        for idx, box in enumerate(actual_boxes):
            xmin, ymin, xmax, ymax = box
            ax.text(xmin, ymin, f"ID: {actual_ids[idx]}", fontsize=12, color='white')

        for idx, box in enumerate(actual_boxes):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            ax.text(xmin, ymin, f"ID: {actual_ids[idx]}", fontsize=12, color='white')

            # Save to MOTChallenge format
            mot_file.write(f"{fff},{actual_ids[idx]},{xmin},{ymin},{width},{height},-1,-1,-1,-1\n")

    previous_boxes = actual_boxes.clone()
    prev_ids = actual_ids.copy()
    fff += 1
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
































class UniqueNumberTracker:
    def __init__(self):
        self.number_ids = {}  # Szám és az hozzárendelt azonosító
        self.last_seen_cycle = {}  # Szám utolsó ciklusának nyilvántartása
        self.current_cycle = 0  # Jelenlegi ciklus számláló
        self.next_id = 1  # Következő szabad azonosító

    def start_new_cycle(self):
        self.current_cycle += 1
        self.current_cycle_numbers = set()  # Inicializálja a jelenlegi ciklusban észlelt számokat

    def add_number(self, number):
        if number not in self.number_ids or self.last_seen_cycle.get(number, 0) < self.current_cycle - 1:
            # Ha az adott szám új, vagy az előző ciklusban nem érkezett
            self.number_ids[number] = self.next_id
            self.next_id += 1
        self.last_seen_cycle[number] = self.current_cycle
        self.current_cycle_numbers.add(number)

    def get_number_id(self, number):
        return self.number_ids.get(number, None)

# Példa használata
assigner = UniqueNumberTracker()


def plot_results_attention(pil_img, prob, boxes, save_path, true_inc):
    assigner.start_new_cycle()
    global previous_boxes, prev_ids, next_id
    global fff
    actual_boxes = torch.empty((0, 4)).to(device)
    actual_ids = []

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    if type(true_inc) is not list:
        true_inc = [true_inc]
    for p, (xmin, ymin, xmax, ymax), c , ccc in zip(prob, boxes.tolist(), COLORS * 100, true_inc):

        cl = p.argmax()

        where_to_find = None
        if only_car_class:
            where_to_find = objects_to_track
        else:
            where_to_find = CLASSES

        if CLASSES[cl] in where_to_find:

            #ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=[szam / 255 for szam in get_color_by_id(ccc)], linewidth=3))
            actual_boxes = torch.cat((actual_boxes, torch.tensor([xmin, ymin, xmax, ymax]).to(device).unsqueeze(0)), 0)
            
            #1,1,611.833855836725,213.7717106624592,361.37881110475905,318.9490093663618,1,-1,-1,-1
            global img_counter

            
            x_to_multiply = 1280 / 480
            y_to_multiply = 720 / 480

            assigner.add_number(ccc)

            kiment = f"{img_counter},{assigner.get_number_id(ccc)},{xmin*x_to_multiply},{ymin*y_to_multiply},{(xmax-xmin)*x_to_multiply},{(ymax-ymin)*y_to_multiply},1,-1,-1,-1\n"

            with open(r"C:\Users\felhasznalo\Desktop\tracker\evaluation\det.txt", 'a') as file:
                file.write(kiment)

            

            
        


            






            #ax.text(xmin, ymin, ccc, fontsize=15, bbox=dict(alpha=0.5))


    previous_boxes = actual_boxes.clone()
    prev_ids = actual_ids.copy()
    fff += 1
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

    img_counter += 1




input_folder = r"C:\Users\felhasznalo\Desktop\dataset\00dc5030-237e7f71\frames"









import os
import shutil

# A kiindulási útvonal
source_path = input_folder

# Visszalépünk egy mappát
parent_directory = os.path.dirname(source_path)

# A gt.txt fájl útvonala a szülő mappában
gt_file_path = os.path.join(parent_directory, "gt.txt")

# A cél mappa útvonala (ide másoljuk a gt.txt-t)
# Itt adja meg a cél mappa útvonalát
destination_directory = r"C:\Users\felhasznalo\Desktop\tracker\evaluation"

# Ellenőrizzük, hogy létezik-e a gt.txt a szülő mappában
if os.path.exists(gt_file_path):
    # A teljes cél útvonal, beleértve a fájl nevét is
    destination_file_path = os.path.join(destination_directory, "gt_orig.txt")
    
    # Másoljuk a fájlt, felülírva ha már létezik
    shutil.copyfile(gt_file_path, destination_file_path)

else:
    print("A gt.txt fájl nem található a megadott helyen.")
















import cv2
import os, re
import glob
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]
# Képek olvasása a mappából
image_folder = input_folder
video_name = 'output_video.avi'  # Az output videó neve

images1 = [img for img in sorted(glob.glob(f"{image_folder}/*.jpg"), key=natural_keys)]

# Videó paraméterek beállítása
frame1 = cv2.imread(images1[0])
height1, width1, layers1 = frame1.shape
fps1 = 20  # Képkockák száma másodpercenként (frames per second)

video1 = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps1, (width1, height1))

# Képek hozzáadása a videóhoz
for image in images1:
    video1.write(cv2.imread(image))


video1.release()




    



output_folder = r"C:\Users\felhasznalo\Desktop\tracker\saved_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img_name in os.listdir(input_folder):
    if img_name.endswith(('.jpg', '.png')):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, "pred_" + img_name)

        #im = Image.open(input_path).resize((480, 480)) #(640, 480)
        im = Image.open(input_path).resize((480, 480)) #(640, 480)
        scores, boxes, true_inc = detect(im, model, transform)



        """max_values = torch.max(scores, 1).values

        k = torchvision.ops.nms(boxes, max_values, 0.0)

        plot_results_attention(im, scores[k], boxes[k], output_path, true_inc)"""
        #plot_results(im, scores, boxes, output_path)
        plot_results_attention(im, scores, boxes, output_path, true_inc)
        











with open(r'C:\Users\felhasznalo\Desktop\tracker\evaluation\gt_orig.txt', 'r') as file:
    lines = file.readlines()


filtered_lines = []

for line in lines:
    filtered_lines.append(line)



output_lines = []
min_area = 0.005 * 1280 * 720 #eredeti 0.005

for line in filtered_lines:
        parts = line.strip().split(',')
        w = float(parts[4])
        h = float(parts[5])
        area = w * h
        if area >= min_area:
            output_lines.append(line.strip())


with open(r"C:\Users\felhasznalo\Desktop\tracker\evaluation\gt.txt", "w") as f:
    f.write("\n".join(output_lines))

































import numpy as np
import motmetrics as mm
import os

#https://motchallenge.net/results/MOT15/
def calculate_iou(box1, box2):

    # Calculate coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate area of union rectangle
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

gt_path = r"C:\Users\felhasznalo\Desktop\tracker\evaluation\gt.txt"
det_path = r"C:\Users\felhasznalo\Desktop\tracker\evaluation\det.txt"



gt_data = np.loadtxt(r"C:\Users\felhasznalo\Desktop\tracker\evaluation\gt.txt", delimiter=',')
det_data = np.loadtxt(r"C:\Users\felhasznalo\Desktop\tracker\evaluation\det.txt", delimiter=',')

iou_thresh = 0.5

tp_count = 0
fp_count = 0
fn_count = 0

for frame in np.unique(gt_data[:, 0]):

    # ground truth and detections for this frame
    gt_boxes = gt_data[gt_data[:, 0] == frame, :]
    det_boxes = det_data[det_data[:, 0] == frame, :]

    # Match detections to ground truth by IoU
    det_matches = []
    det_used = []
    for det_box in det_boxes:
        ious = [calculate_iou(det_box[2:], gt_box[2:]) for gt_box in gt_boxes]
        max_iou = max(ious) if len(ious) > 0 else 0
        if max_iou >= iou_thresh:
            gt_match = gt_boxes[np.argmax(ious), :]
            if gt_match[1] not in det_used:
                det_matches.append(det_box)
                det_used.append(gt_match[1])

    # Count true positives, false positives, and false negatives for this frame
    tp_count += len(det_matches)
    fp_count += det_boxes.shape[0] - len(det_matches)
    fn_count += gt_boxes.shape[0] - len(det_used)

max_consecutive_frames = 5

tpa_count = 0
fpa_count = 0
fna_count = 0

for frame in np.unique(gt_data[:, 0]):

    #ground truth and detections for this frame
    gt_boxes = gt_data[gt_data[:, 0] == frame, :]
    det_boxes = det_data[det_data[:, 0] == frame, :]

    # Match detections to ground truth by IoU
    det_matches = []
    det_used = []
    for det_box in det_boxes:
        ious = [calculate_iou(det_box[2:], gt_box[2:]) for gt_box in gt_boxes]
        max_iou = max(ious) if len(ious) > 0 else 0
        if max_iou >= iou_thresh:
            gt_match = gt_boxes[np.argmax(ious), :]
            if gt_match[1] not in det_used:
                det_matches.append(det_box)
                det_used.append(gt_match[1])

    # Check matches for consecutive frames
    for i in range(1, max_consecutive_frames):
        next_frame = frame + i
        if next_frame in np.unique(gt_data[:, 0]):
            next_gt_boxes = gt_data[gt_data[:, 0] == next_frame, :]
            next_det_boxes = det_data[det_data[:, 0] == next_frame, :]
            next_det_matches = []
            for det_box in det_matches:
                ious = [calculate_iou(det_box[2:], next_gt_box[2:]) for next_gt_box in next_gt_boxes]
                max_iou = max(ious) if len(ious) > 0 else 0
                if max_iou >= iou_thresh:
                    next_det_matches.append(det_box)
            det_matches = next_det_matches

    # Count true positive associations, false positive associations, and false negatives for this frame
    gt_used = []
    for gt_box in gt_boxes:
        if gt_box[1] in det_used:
            tpa_count += 1
            gt_used.append(gt_box[1])
        else:
            fna_count += 1
    for det_box in det_boxes:
        if np.array([np.array_equal(det_box, det_match) for det_match in det_matches]).any():
            continue
        if det_box[1] not in gt_used:
            fpa_count += 1

detiou = tp_count / (tp_count + fn_count + fp_count)

assiou = tpa_count / (tpa_count + fna_count + fpa_count)

hota = np.sqrt(detiou * assiou)

import numpy as np

def motMetricsEnhancedCalculator(gtSource, tSource):

  gt = np.loadtxt(gtSource, delimiter=',')

  t = np.loadtxt(tSource, delimiter=',')

  acc = mm.MOTAccumulator(auto_id=True)

  for frame in range(int(gt[:,0].max())):
    frame += 1 

    
    gt_dets = gt[gt[:,0]==frame,1:6] 
    t_dets = t[t[:,0]==frame,1:6] 

    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=0.5) 

    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)

  mh = mm.metrics.create()

  summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], \
                      name='')

  strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
  )
  #return strsummary
  summary_lines = strsummary.splitlines()
  header_line = summary_lines[0]
  values_line = summary_lines[1]

  return f"{header_line}  DetIoU      AssIoU      HOTA\n{values_line}  {detiou:0.6f}  {assiou:0.6f}  {hota:0.6f}\n"


print()
results = motMetricsEnhancedCalculator(gt_path, det_path)
print(results)


























exit()










video_path = r"C:\Users\felhasznalo\Desktop\tracker\output_video.avi"
cap = cv2.VideoCapture(video_path)

model.eval()

with torch.no_grad():
    image_gen = read_and_resize_images(r"C:\Users\felhasznalo\Desktop\dataset\00a0f008-3c67908e\frames")
    while True:
        ret, frame = cap.read()  # Read a new frame from video
        if not ret:
            break
        
        # Convert the frame to the format your model expects (you might need to adjust this part)
        #test_image = preprocess_a_custom_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        test_image = preprocess_a_custom_image(frame)
        
        # Make predictions
        preds = model(test_image)
        preds = list(preds)
        preds[1] = preds[1].permute(1, 0, 2)
        preds[0] = preds[0].permute(1, 0, 2)

        # Show results (you might need to adjust this part according to your `show` function)
        counter += 1

        #class_ids = preds[2]

        #show(preds[0][0], preds[1][0], counter, class_ids, image_gen)
        for i in range(10):
            print("utofleldogozas CIOU-val belül")
        try:

            show(preds[0][0], preds[1][0], counter, image_gen)
            #show(preds[0][0][0:1], preds[1][0][0:1], counter, image_gen)
        except Exception as e:
            print(e)
            image_np = next(image_gen)
            plt.imshow(image_np, cmap='gray')
            file_name = f"{output_dir}/image_epoch_loss_{counter}.png"
            plt.savefig(file_name)
            plt.close()

        

        #show(preds[1][0], preds[0][0], counter, image_gen)
        #exit()
# Release video capture

with open(r'C:\Users\felhasznalo\Desktop\tracker\evaluation\det.txt', 'w') as output_file:
    for line in lines_to_write:
        output_file.write(','.join(map(str, line)) + '\n')
cap.release()
cv2.destroyAllWindows()

"""


counter = 0
image_gen = read_and_resize_images(r"")
def show_batch(all_box_coords, all_classes):
    global image_gen
    
    for box_coords, classes in zip(all_box_coords, all_classes):
        
        inner_list = list()
        scaler = MinMaxScaler()
        probabilities = F.softmax(classes, dim=1)
        threshold = 0.0
        j = 0
        predicted_classes = torch.argmax(probabilities, dim=1)
        image_np = next(image_gen)
        j_list = set()


        for i, box in enumerate(box_coords):


            box = scaler.de_normalize(box)
            if probabilities[i, predicted_classes[i]] >= threshold:

                cx, cy, w, h = map(int, box)
                x1, y1, w, h = converter.to_corner_coordinates(cx, cy, w, h)

                if predicted_classes[i] == 1:
                    color = (0, 0, 0)


                    cv2.putText(image_np, str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    inner_list.append(i)
                    image_np = cv2.rectangle(image_np, (x1, y1), (w+x1, h+y1), color, 2)
                    

        ids.append(inner_list)
        #print(find_missing_numbers(ids))
        plt.imshow(image_np, cmap='gray')
        global counter
        file_name = f"{output_dir}/image_epoch_loss_{counter}.png"
        counter += 1
        plt.savefig(file_name)
        plt.close()

model = torch.load('best_model.pth')
model.eval()

video_path = r""
cap = cv2.VideoCapture(video_path)

buffer = []

with torch.no_grad():
    counter = 0
    while True:
        ret, frame = cap.read()  # Read a new frame from video
        if not ret:
            break

        # Convert the frame to the format your model expects
        test_image = preprocess_a_custom_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Add the image tensor to buffer
        buffer.append(test_image)

        # Check if buffer size is 100
        if len(buffer) == 100:
            # Create a batch tensor of shape [100, 1, 255, 254]
            batch_tensor = torch.cat(buffer, dim=0)

            # Make predictions
            preds = model(batch_tensor)

            #preds[1] = preds[1].permute(1, 0, 2)
            #preds[0] = preds[0].permute(1, 0, 2)



            show_batch(preds[0].permute(1, 0, 2), preds[1].permute(1, 0, 2))


            buffer.clear()

            print("Processed a batch of 100 images")

# Release video capture
cap.release()
cv2.destroyAllWindows()




"""

