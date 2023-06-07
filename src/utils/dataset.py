"""
Utility functions for all dataset related functions 
>> mask_to_bbox: convert a mask label to a bounding box label
"""

import numpy as np
from utils.io import read_image


def mask_to_bbox(image_path):
    # Read image (greyscale)
    seg_img = read_image(image_path, is_greyscale=True)

    # Get coordinates where pixel is not black (white)
    y_coords, x_coords = np.where(seg_img != 0)

    # If these coordinates exist
    if len(x_coords) and len(y_coords):
        w, h = seg_img.shape
        # Get top left & bottom right points
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        # Adjust bbox by 1 pixel
        x1, y1 = max(x1 - 1, 0), max(y1 - 1, 0)
        x2, y2 = min(x2 + 1, w), min(y2 + 1, h)
        bbox = (x1, y1, x2, y2)

        # Draw and display bbox
        # img = cv2.rectangle(seg_img, (x1,y1), (x2,y2), (255, 0, 0), 1)
        # plt.imshow(img, cmap='gray')
        # plt.show()

        return bbox
