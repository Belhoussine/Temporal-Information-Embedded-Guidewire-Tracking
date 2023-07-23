from src.dataset import TrackingDataset, get_train_val_split
from src.dataloader import TrackingDataloader

synthetic_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data"

if __name__ == "__main__":
    dataset = TrackingDataset(root=synthetic_path)
    train_dataset, val_dataset = get_train_val_split(dataset, val_split=0.2)

    train_dataloader = TrackingDataloader(train_dataset, batch_size=32, is_train=True)
    val_dataloader = TrackingDataloader(val_dataset, batch_size=32, is_train=False)

    print(len(train_dataset), len(val_dataset))
    print(len(train_dataloader), len(val_dataloader))