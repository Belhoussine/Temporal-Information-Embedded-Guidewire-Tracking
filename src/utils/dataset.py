"""
Utility functions for all dataset related functions 
>> mask_to_bbox: convert a mask label to a bounding box label
"""

import numpy as np

from src.utils.io import read_image


def mask_to_bbox(image_path):
    """
    Implement a function that reads a mask image and outputs bbox coordinates.
    >> Input: The path to the mask image (e.g path/to/mask.png)
    >> Output: The coordinates of the bounding box in the form: (x1, y1, x2, y2), where
        => (x1, y1): top-left point
        => (x2, y2): bottom-right point
    >> Algorithm:
        1. Read the image from the path and store it as a numpy array (matrix)
        2. Scan the image vertically
            2.a. Starting from the top -> get top white pixel (y1)
            2.b. Starting from the bottom -> get bottom white pixel (y2)
        3. Scan the image horizontally
            3.a. Starting from the left -> get left most white pixel (x1)
            3.b. Starting from the right -> get right most white pixel (x2)
        4. Offset all coordinates by 1 (to include border pixels)
            4.a. x1 = x1 - 1; y1 = y1 -1; 
            4.b. x2 = x2 + 1; y2 = y2 + 1; 
            4.c. Make sure coordinates are valid after this update (e.g not negative or > img size)
        5. Return (x1, y1, x2, y2)
    """
    # TODO
    pass
