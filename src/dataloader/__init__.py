import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class TrackingDataloader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
        is_train=True,
        shuffle=False,
        num_workers=8,
        collate_fn=default_collate,
        processor=None
    ):
        self.processor = processor

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(**self.init_kwargs)


    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch