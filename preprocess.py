

import torch
import copy
import random
from PIL import Image
from utils import one_hot_encode
from normalizer import MinMaxScaler
from torch.utils.data import Dataset
from coordinate import CoordinateConverter
from torchvision.transforms import ToTensor
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

class ImagesDataset(Dataset):
    def __init__(self, images_path : list, labels : list, max_len_of_preds : int, image_size : tuple, train_with_grayscale : bool, do_augmentation, augmentation_prob, device = None):
        self.images_path = images_path
        self.labels = labels
        self.image_size = image_size

        self.max_len_of_preds = max_len_of_preds
        self.image_size = image_size

        self.train_with_grayscale = train_with_grayscale

        self.scaler = MinMaxScaler()
        self.converter = CoordinateConverter()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.do_augmentation = do_augmentation
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.images_path)

    def apply_random_transform(self, image):
        transforms = [
            ("grayscale", lambda img: ImageOps.grayscale(img).convert("RGB")),
            ("rotate", lambda img: img.rotate(random.uniform(-5, 5))),
            ("blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 3.0)))),
            ("color_jitter", lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))),
            ("brightness", lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))),
            ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))),
        ]
        
        if random.random() < self.augmentation_prob: 
            chosen_transforms = random.choices(transforms, k=random.randint(1, 4))
            for name, transform in chosen_transforms:
                image = transform(image)
            
        return image



    def __getitem__(self, idx):
        image = None

        if self.train_with_grayscale:
            image = Image.open(self.images_path[idx]).convert('L')
        else:
            image = Image.open(self.images_path[idx]).convert('RGB')

        owidth, oheight = image.size

        image = image.resize(self.image_size)

        if self.do_augmentation:
            image = self.apply_random_transform(image)

        nwidth, nheight = self.image_size
        
        labels = copy.deepcopy(self.labels[idx])
        for i in range(len(labels)):
            x1, y1, w, h = labels[i][0]
            x1 = x1 * nwidth / owidth
            y1 = y1 * nheight / oheight
            w = w * nwidth / owidth
            h = h * nheight / oheight
            cx, cy, w, h = self.converter.to_center_coordinates(x1, y1, w, h) #####
            labels[i][0] = [cx, cy, w, h]
            #labels[i][0] = [x1, y1, w, h]

        padding_needed = self.max_len_of_preds - len(labels)
        if padding_needed > 0:
            labels += [[[0, 0, 0, 0], 0]] * padding_needed
        elif padding_needed < 0:
            labels = labels[:self.max_len_of_preds]
        
        image = ToTensor()(image)

        image = image.to(self.device)

        bounding_boxes = [label[0] for label in labels]
        class_ids = [label[1] for label in labels]

        bounding_boxes_tensor = torch.Tensor(bounding_boxes).to(self.device)

        # Normalizálás
        normalized_bounding_boxes = self.scaler.normalize(bounding_boxes_tensor).to(self.device)

        # Osztályazonosítók (class IDs) tensorba konvertálása
        class_ids_tensor = torch.Tensor(class_ids).to(self.device).long()# long tensor, mert ezek indexek

        # Egy dictionary-be csomagoljuk az információt, hogy minden egyben maradjon
        labels = {
            "bounding_boxes": normalized_bounding_boxes,
            "class_ids": class_ids_tensor
        }

        return image, labels


