"""
Utility functions for all dataset related functions 
>> mask_to_bbox: convert a mask label to a bounding box label
"""

import os
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
from src.utils.io import read_image, write_json

# Convert mask image to bounding box


def mask_to_bbox(image_path, is_dhm):
    # Read image (greyscale)
    if is_dhm:
        seg_img = read_image(image_path, is_greyscale=False)
        r,g,b = seg_img[:, :, 0], seg_img[:, :, 1], seg_img[:, :, 2]  
        seg_img = (r > 200) & (g < 150) & (b < 150)
        y_coords, x_coords = np.where(seg_img != 0)
        if len(x_coords) == 0 and len(y_coords) == 0:
            seg_img = (r <= 150) & (g <= 150) & (b > 200)
    else:
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

    return [-1, -1, -1, -1]

# Flatten an image matrix to a vector


def flatten(image):
    img_vec = image.flatten()
    return img_vec

# Stack sequence frames as columns


def stack_frames(frame_paths, resolution=(512, 512), n_frames=10, frames_loaded = False):
    n_pixels = resolution[0] * resolution[1]
    stacked_matrix = np.empty(shape=(n_pixels, n_frames), dtype='float')
    for i, frame_path in enumerate(frame_paths):
        if frames_loaded:
            frame = frame_path
        else:
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


def get_img_gt_paths(root, image_paths, label_paths, is_dhm):
    if is_dhm:
        dhm_img_path = f'{root}/DHM_guidewiredataset'
        dhm_gt_path = f'{root}/DHM_guidewiremask'
        for path, dirs, files in os.walk(dhm_img_path):
            dirs.sort()
            if len(dirs) == 0:
                for filename in sorted(files):
                    full_path = os.path.join(path,filename)
                    # relative_path = '/'.join(full_path.split('/')[-3:])
                    image_paths.append(full_path)
        for path, dirs, files in os.walk(dhm_gt_path):
            dirs.sort()
            if len(dirs) == 0:
                for filename in sorted(files):
                    full_path = os.path.join(path,filename)
                    # relative_path = '/'.join(full_path.split('/')[-3:])
                    label_paths.append(full_path)

    else:
        for path, dirs, files in os.walk(root):
            dirs.sort()
            if len(dirs) == 0:
                for filename in sorted(files):
                    full_path = os.path.join(path,filename)
                    # relative_path = '/'.join(full_path.split('/')[-3:])
                    is_label = full_path.split('/')[-2] == 'gt'

                    if is_label:
                        label_paths.append(full_path)
                    else:
                        image_paths.append(full_path)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results
     

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, threshold=0.7):
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  return probas_to_keep, bboxes_scaled

def create_coco_dataset(root = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets", dataset_name= "Guide_wire_sythetic_data", is_dhm=False, save_path="annotations/custom_train.json"):
    dataset = {
        "info":{
            "year": 2023,
            "description": "Custom coco dataset for fine-tuning DETR on guidewire detection dataset.",
            "contributor": "Belhoussine"
        },
        "licenses": None,
        "categories":[
            {
                "id": 0,
                "name": "guidewire",
                "supercategory": None,
            }
        ],
        "images":[],
        "annotations":[], 
    }

    img_paths, label_paths = [], []
    get_img_gt_paths(f"{root}/{dataset_name}", img_paths, label_paths, is_dhm)
    
    for i, (img_path, label_path) in tqdm(enumerate(zip(img_paths, label_paths)), total=len(img_paths)):
        img_info = {
            "id": i,
            "license": None,
            "file_name": '/'.join(img_path.split('/')[-5:]),
            "height": 512,
            "width": 512,
            "date_captured": None
        }
        dataset["images"].append(img_info)
        
        # print(img_path, label_path, mask_to_bbox(label_path, is_dhm))
        x1, y1, x2, y2 = mask_to_bbox(label_path, is_dhm)
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        annotation = {
            "id": i,
            "image_id": i,
            "category_id": 0,
            "bbox": bbox,
            "segmentation": None,
            "area": None,
            "iscrowd": 0
        }
        dataset["annotations"].append(annotation)
    write_json(dataset, f"{root}/{save_path}")


def visualize_random_image(train_dataset):
    import numpy as np
    import os
    from PIL import Image, ImageDraw

    image_ids = train_dataset.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image n°{}'.format(image_id))
    image = train_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(datasets_folder, image['file_name'])).convert("RGB")

    annotations = train_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')

    write_image(image, "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/results/img.png", is_pil=True)