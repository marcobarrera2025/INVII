from torch.utils.data import DataLoader

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS
from dataset import HandbagDataset
from transforms_pipeline import train_transform, eval_transform

def build_dataloaders():
    train_dataset = HandbagDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = HandbagDataset(
        root_dir=VAL_DIR,
        transform=eval_transform
    )

    test_dataset = HandbagDataset(
        root_dir=TEST_DIR,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
