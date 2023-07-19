# Load labels and images
# Parse labels to pass them in desired format, with image paths (see format_util.py or test_labels.py)
# Convert output format to desired format (from ocg)

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class TrackingDataset(Dataset):
    def __init__(self, root=None, is_train=True, transform=None):
        super().__init__()
        self.image_paths = []
        self.label_paths = []
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        # Get image and thebounding box
        image = read_image(img_path)
        label = self.label_paths[index]

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