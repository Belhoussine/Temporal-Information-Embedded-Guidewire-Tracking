from src.utils.io import write_image
from src.models.transformers import Detr
from src.dataset import TrackingCoCoDataset
from src.dataloader import TrackingDataloader
# from src.utils.dataset import create_coco_dataset

from pytorch_lightning import Trainer
from transformers import DetrImageProcessor


# Dataset & Dataloader Definition
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

datasets_folder = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets"

train_dataset = TrackingCoCoDataset(root=datasets_folder, processor=processor)
val_dataset = TrackingCoCoDataset(root=datasets_folder, processor=processor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

train_dataloader = TrackingDataloader(train_dataset, batch_size=16, shuffle=True, processor=processor)
val_dataloader = TrackingDataloader(val_dataset, batch_size=16, is_train=False, processor=processor, num_workers=0)

# Model definition
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

# Training
epochs = 100
trainer = Trainer(max_epochs=epochs, gradient_clip_val=0.1, accelerator="gpu", devices = 1, default_root_dir="/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/checkpoints/")
trainer.fit(model)

trainer.save_checkpoint(f"/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/checkpoints/detr_model_ep{epochs}.ckpt")