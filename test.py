import cv2
import time
import torch
import requests
import numpy as np
from PIL import Image
from random import randint
from src.utils.io import plot_results
from src.models.transformers import Detr
from src.dataset import TrackingCoCoDataset
from src.dataloader import TrackingDataloader
from transformers import DetrImageProcessor, DetrForObjectDetection


datasets_folder = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets"

# Loading image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Loading validation dataset and dataloader
val_dataset = TrackingCoCoDataset(root=datasets_folder, processor=processor, train=False)
val_dataloader = TrackingDataloader(val_dataset, batch_size=16, is_train=False, processor=processor, num_workers=0)

# Loading model from checkpoint
checkpoint = torch.load("/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/checkpoints/lightning_logs/version_1/checkpoints/epoch=99-step=93900.ckpt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=None, val_dataloader=None)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to(device)

# Detection on n random images in val dataset
n = 700
for i in range(n):
    # idx = randint(0, len(val_dataset))
    idx = i
    pixel_values, target = val_dataset[idx]
    pixel_values = pixel_values.unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
    end = time.time()
    image_id = target['image_id'].item()
    img = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(f"{datasets_folder}/{img['file_name']}")

    print(f"Image #{idx} in val_dataset: {img['file_name']} - {end-start}s")

    # postprocess model outputs
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=0.9)
    results = postprocessed_outputs[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'], img['file_name'], patch_only=True)