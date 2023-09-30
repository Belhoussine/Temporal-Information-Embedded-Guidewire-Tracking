# Load labels and images
# Parse labels to pass them in desired format, with image paths (see format_util.py or test_labels.py)
# Convert output format to desired format (from ocg)

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

from src.utils.io import read_image
from src.utils.dataset import mask_to_bbox, get_img_gt_paths



class TrackingDataset(Dataset):
    def __init__(self, root=None, is_train=True, transform=None, val_split=0.2, is_dhm=False):
        super().__init__()

        self.root = root
        self.image_paths = []
        self.label_paths = []
        self.is_train = is_train
        self.transform = transform
        self.is_dhm = is_dhm

        get_img_gt_paths(self.root, self.image_paths, self.label_paths, self.is_dhm)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get image and the bounding box
        image = read_image(self.image_paths[index])
        label = mask_to_bbox(self.label_paths[index], self.is_dhm)
        # Apply transforms on the generated crop
        if self.transform:
            image, label = self.transform(image, label)

        # Create a sample from image and its label
        sample = image, label
        return sample


# CoCo dataset
class TrackingCoCoDataset(CocoDetection):
    def __init__(self, root, processor, train=True):
        ann_file = f"{root}/annotations/{'custom_train.json' if train else 'custom_val.json'}"
        super().__init__(root, ann_file)
        self.processor = processor
        self.bbox_only=False

    def __getitem__(self, index):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(index)
        img = img.convert("RGB")

        if self.bbox_only:
            return target

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[index]
        target = {'image_id': image_id, 'annotations': target}

        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target