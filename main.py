from src.dataset import TrackingDataset, get_train_val_split
from src.dataloader import TrackingDataloader

synthetic_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data"
DHM_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA"
DHMv2_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA_v2"

datasets = {
    "DHM": {
        "path": "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA", 
        "is_dhm": True
    },
    "DHMv2": {
        "path": "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA_v2", 
        "is_dhm": True
    },
    "Synth": {
        "path": "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data", 
        "is_dhm": False
    },
}

if __name__ == "__main__":
    train_dataset = TrackingDataset(root=datasets["Synth"]["path"], is_train=True, is_dhm=datasets["Synth"]["is_dhm"])
    val_dataset = TrackingDataset(root=datasets["DHM"]["path"], is_train=False, is_dhm=datasets["DHM"]["is_dhm"])

    import cv2
    import torch
    import requests
    import numpy as np
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor

    # Online image
    url = "http://images.cocodataset.org/test-stuff2017/000000025775.jpg" #"https://placekitten.com/200/140"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size, len(image.split()))

    # Dataset image
    # path = "datasets/Guide_wire_sythetic_data/sequence01_frame00000_sequence25_frame00150/png/navi00000005.png"
    # image = Image.open(path).convert("RGB")
    # print(image.size, len(image.split()))

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )