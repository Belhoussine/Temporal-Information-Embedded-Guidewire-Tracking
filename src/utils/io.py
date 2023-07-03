"""
Utility functions for all input/output or file/image related functions 
>> read_image: read an image given its path
>> write_image: write an image to a path
"""

import os
import yaml
import numpy as np
from PIL import Image
from pathlib import Path


# Load image as a numpy array from a given path
def read_image(image_path, is_greyscale=False):
    img = Image.open(image_path)
    # Convert to greyscale if image is a mask (or greyscale image)
    if is_greyscale:
        img = img.convert('L')

    img.load()
    image_array = np.asarray(img, dtype="int32")
    return image_array

# Save numpy array as an image in given path


def write_image(image_array, image_path, mode="RGB"):
    c = np.clip(image_array, 0, 255)
    img = Image.fromarray(np.asarray(c, dtype="uint8"), mode)
    img.save(image_path)

# Delete Image from a given path


def remove_image(save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return yaml.safe_load(handle)


def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
