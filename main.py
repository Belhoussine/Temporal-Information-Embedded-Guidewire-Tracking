from src.dataset import TrackingDataset, get_train_val_split
from src.dataloader import TrackingDataloader

synthetic_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data"
DHM_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA"
DHMv2_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/DHM-DATA_v2"

if __name__ == "__main__":
    dataset = TrackingDataset(root=DHM_path, is_dhm=True)