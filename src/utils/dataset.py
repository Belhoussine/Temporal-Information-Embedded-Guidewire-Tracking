"""
Utility functions for all dataset related functions 
>> mask_to_bbox: convert a mask label to a bounding box label
"""

import numpy as np
from utils.io import read_image

# Convert mask image to bounding box


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

# Flatten an image matrix to a vector


def flatten(image):
    img_vec = image.flatten()
    return img_vec

# Stack sequence frames as columns


def stack_frames(frame_paths, resolution=(512, 512), n_frames=10):
    n_pixels = resolution[0] * resolution[1]
    stacked_matrix = np.empty(shape=(n_pixels, n_frames), dtype='float')
    for i, frame_path in enumerate(frame_paths):
        frame = read_image(frame_path, is_greyscale=True)
        flat_frame = flatten(frame)
        stacked_matrix[:, i] = flat_frame
    return stacked_matrix


def unstack_frames(stacked_matrix, resolution=(512, 512)):
    S_frames = []
    for col in range(stacked_matrix.shape[1]):
        S_frame = normalize(np.reshape(
            stacked_matrix[:, col], (-1, resolution[1])))*255
        S_frames.append(S_frame)
    return S_frames


def normalize(matrix):
    return 1-(matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
