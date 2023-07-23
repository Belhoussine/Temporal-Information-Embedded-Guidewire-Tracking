import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class TrackingDataloader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
        is_train=True,
        shuffle=True,
        num_workers=1,
        collate_fn=default_collate,
    ):

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(**self.init_kwargs)