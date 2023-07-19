# Load labels and images
# Parse labels to pass them in desired format, with image paths (see format_util.py or test_labels.py)
# Convert output format to desired format (from ocg)

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ocg.src.utils.file_util import load_JSON
from ocg.src.utils.image_util import load_image


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
        img_path = self.labels[index]["meta"]["image_path"]

        # Get image and the object's bounding box
        image = load_image(img_path)
        bbox, label = self.labels[index]['bbox'][:4], int(
            self.labels[index]['bbox'][4])

        # Get crop and class label
        crop = self.get_crop(image, bbox, size=self.crop_size, to_pil=True)

        # Apply transforms on the generated crop
        if self.transform:
            crop = self.transform(crop)

        # Create a sample from the crop and its class label
        # sample = crop, label
        sample = crop, self.label_to_index(label)
        return sample

    def get_crop(self, image, bbox, size=64, to_pil=False, to_tensor=False):
        image = image.astype(np.uint8)
        img_h, img_w, _ = image.shape
        l, t, r, b = map(int, map(np.ceil, bbox))

        crop_size = max(r - l, b - t)
        r, b = l + crop_size, t + crop_size

        assert r - l == b - \
            t, f"bbox is not square: width {r - l} != height {b - t}"

        crop = np.zeros([crop_size, crop_size, 3], dtype=np.uint8)

        # Top-left coordinates of where to put the image in the final crop
        start_l = -l if l < 0 else 0
        start_t = -t if t < 0 else 0

        object_crop = image[max(0, t):min(img_h, b), max(0, l):min(img_w, r)]
        crop[start_t: start_t + object_crop.shape[0],
             start_l: start_l + object_crop.shape[1]] = object_crop

        crop = cv2.resize(crop, dsize=(size, size),
                          interpolation=cv2.INTER_CUBIC)

        # self.show_image(image)
        if to_pil:
            crop = Image.fromarray(crop)
        elif to_tensor:
            crop = torch.from_numpy(crop) / 255.0
            crop = crop.permute((2, 0, 1))

        return crop

    def get_class_map(self):
        return {i: self.categories[i] for i in range(len(self.categories))}

    def ohe(self, label):
        label_index = self.categories.index(f"{label}")
        return torch.FloatTensor([1 if i == label_index else 0 for i in range(len(self.categories))])

    def label_to_index(self, label):
        return self.categories.index(label)

    def index_to_label(self, index):
        return self.categories[index]

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