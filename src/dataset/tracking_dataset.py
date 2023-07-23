# Load labels and images
# Parse labels to pass them in desired format, with image paths (see format_util.py or test_labels.py)
# Convert output format to desired format (from ocg)

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils.io import read_image
from src.utils.dataset import mask_to_bbox



class TrackingDataset(Dataset):
    def __init__(self, root=None, is_train=True, transform=None, val_split=0.2, dhm_format=False):
        super().__init__()

        self.root = root
        self.image_paths = []
        self.label_paths = []
        self.is_train = is_train
        self.transform = transform

        for path, dirs, files in os.walk(self.root):
            dirs.sort()
            if len(dirs) == 0:
                for filename in sorted(files):
                    full_path = os.path.join(path,filename)
                    # relative_path = '/'.join(full_path.split('/')[-3:])
                    is_label = full_path.split('/')[-2] == 'gt'

                    if is_label:
                        self.label_paths.append(full_path)
                    else:
                        self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # Get image and the bounding box
        image = read_image(self.image_paths[index])
        label = mask_to_bbox(self.label_paths[index])

        # Apply transforms on the generated crop
        if self.transform:
            image, label = self.transform(image, label)

        # Create a sample from image and its label
        sample = image, label
        return sample

    def show_image(self, image, pil=False, tensor=False):
        if pil:
            image = np.array(image)
        elif tensor:
            image = image.permute((1, 2, 0)).numpy()
        image = image.astype(np.uint8)
        try:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            rgb_img = image
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)