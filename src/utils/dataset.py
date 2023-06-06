"""
Utility functions for all dataset related functions 
>> mask_to_bbox: convert a mask label to a bounding box label
"""

import numpy as np
from PIL import Image
from numpy import asarray
"from src.utils.io import read_image"


def mask_to_bbox(image_path):
    
    "Store image as a numpy array (matrix)"
    data = asarray(image_path)
    print(type(data))
    print(data.shape)
    
    "Scan the image vertically"
    "Starting from the top -> get top white pixel (y1)"
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            if np.all(data[row, column] == [255, 255, 255]):
                y1 = row
                print("y1 = ", y1)
            else:
                continue
            break
        
    "Starting from the bottom --> get bottom white pixel (y2)"
    for row in reversed(range(data.shape[0])):
        for column in range(data.shape[1]):
            if np.all(data[row, column] == [255, 255, 255]):
                y2 = row
                print("y2 = ", y2)
            else:
                continue
            break
    
    
    "Scan the image horizontally"
    "Starting from the left --> get left most white pixel (x1)"

    for column in range (data.shape[1]):
        for row in range(data.shape[0]):
            if np.all(data[row, column] == [255, 255, 255]):
                x1 = column
                print("x1 = ", x1)
            else:
                continue
            break

    "Starting from the right --> get right most white pixel (x2)"
    for column in reversed(range(data.shape[1])):
        for row in range(data.shape[0]):
            if np.all(data[row, column] == [255, 255, 255]):
                x2 = column
                print("x2 = ", x2)
            else:
                continue
            break

    
    "Offset all coordinates by 1 (to include border pixels)"
    "x1 = x1 - 1; y1 = y1 - 1"
    x1 = x1 - 1
    y1 = y1 - 1
    
    "x2 = x2 + 1; y2 = y2 + 1"
    x2 = x2 + 1
    y2 = y2 + 1
    
    print(x1, y1, x2, y2)
    
    "Check validity of coordinates after update (e.g. no negative or >img size)"
    image_width, image_height = data.shape[:2]
    if 0 <= x1 <= image_width:
        pass
    else: 
        print("x1 is not valid")
        return
        
    if 0 <= x2 <= image_width:
        pass
    else:
        print("x2 is not valid")
        return
        
    if 0 <= y1 <= image_height:
        pass
    else:
        print("y1 is not valid")
        return
        
    if 0 <= y2 <= image_height:
        pass
    else:
        print("y2 is not valid")
        return
    
    "Return cooridnates ((x1, y1), (x2, y2))"
    coordinates = [(x1, y1), (x2, y2)]
    return coordinates

    pass

"Input: The path of the mask image (e.g path/to/mask.pgn)"
inp = Image.open("C:/Users/ninam/OneDrive/Desktop/navi00000017.png")

mask_to_bbox(inp)
