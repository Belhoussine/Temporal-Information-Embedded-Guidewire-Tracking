import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from coco_eval import CocoEvaluator
from src.utils.io import plot_results
from src.models.transformers import Detr
from src.dataset import TrackingCoCoDataset, TrackingDataset
from src.dataloader import TrackingDataloader
from src.utils.dataset import prepare_for_coco_detection
from transformers import DetrImageProcessor, DetrForObjectDetection

def get_iou(bb1, bb2):
  # Adjustements for bbox from (x,y,w,h) to (x1,y1, x2,y2)
  bb1 = {
    'x1': bb1[0],
    'y1': bb1[1],
    'x2': bb1[0] + bb1[2],
    'y2': bb1[1] + bb1[3],
  }

  bb2 = {
    'x1': bb2[0],
    'y1': bb2[1],
    'x2': bb2[0] + bb2[2],
    'y2': bb2[1] + bb2[3],
  }

  """
  Calculate the Intersection over Union (IoU) of two bounding boxes.

  Parameters
  ----------
  bb1 : dict
      Keys: {'x1', 'x2', 'y1', 'y2'}
      The (x1, y1) position is at the top left corner,
      the (x2, y2) position is at the bottom right corner
  bb2 : dict
      Keys: {'x1', 'x2', 'y1', 'y2'}
      The (x, y) position is at the top left corner,
      the (x2, y2) position is at the bottom right corner

  Returns
  -------
  float
      in [0, 1]
  """
  assert bb1['x1'] < bb1['x2']
  assert bb1['y1'] < bb1['y2']
  assert bb2['x1'] < bb2['x2']
  assert bb2['y1'] < bb2['y2']

  # determine the coordinates of the intersection rectangle
  x_left = max(bb1['x1'], bb2['x1'])
  y_top = max(bb1['y1'], bb2['y1'])
  x_right = min(bb1['x2'], bb2['x2'])
  y_bottom = min(bb1['y2'], bb2['y2'])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
  bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou


def get_batch_iou(pred, labels):
  avg_iou = 0
  for p, l in zip(pred, labels):
    if l['bbox'][0] > 0:
      iou = get_iou(p['bbox'], l['bbox'])
    else:
      iou = 1
    avg_iou += iou
  return avg_iou / len(pred)


datasets_folder = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets"

# Loading image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Loading validation dataset and dataloader
val_dataset = TrackingCoCoDataset(root=datasets_folder, processor=processor, train=False)
val_dataloader = TrackingDataloader(val_dataset, batch_size=4, is_train=False, processor=processor, num_workers=0)

# Loading model from checkpoint
checkpoint = torch.load("/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/checkpoints/lightning_logs/version_1/checkpoints/epoch=99-step=93900.ckpt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=None, val_dataloader=None)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to(device)


# initialize evaluator with ground truth (gt)
# evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])
avg_iou = 0
total_batches_considered = 0

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

    # provide to metric
    # metric expects a list of dictionaries, each item 
    # containing image_id, category_id, bbox and score keys 
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    # print(f"Batch: {batch}")
    # print()
    # print(f"Predictions: {predictions}")
    # print()
    # print(f"Labels: {labels}")
    # print()
    # print(f"Dataset: {val_dataset[idx*4: (idx+1)*4]}")
    val_dataset.bbox_only=True
    print(val_dataset[idx*4: (idx+1)*4])
    batch_iou =get_batch_iou(predictions, val_dataset[idx*4: (idx+1)*4]) 
    if batch_iou > 0.1:
      total_batches_considered += 1
    avg_iou += batch_iou
    print(batch_iou)
    print(avg_iou / total_batches_considered)
    val_dataset.bbox_only=False

    # evaluator.update(predictions)

avg_iou /= total_batches_considered
print(f"Average IOU: {avg_iou}")
# evaluator.synchronize_between_processes()
# evaluator.accumulate()
# evaluator.summarize()