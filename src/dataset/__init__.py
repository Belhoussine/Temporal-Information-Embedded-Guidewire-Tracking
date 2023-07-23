from src.dataset.tracking_dataset import TrackingDataset
from torch.utils.data import random_split

def get_train_val_split(dataset, val_split=0.2):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset