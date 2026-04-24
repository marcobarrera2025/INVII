import os
import random
import shutil
from math import floor

from config import (
    GENUINE_DIR,
    SPLIT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

def split_by_bag():
    random.seed(RANDOM_SEED)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    if not GENUINE_DIR.exists():
        print(f"No existe GENUINE_DIR: {GENUINE_DIR}")
        return False

    for model in sorted(os.listdir(GENUINE_DIR)):
        model_path = GENUINE_DIR / model

        if not model_path.is_dir():
            continue

        bag_ids = [
            d for d in os.listdir(model_path)
            if (model_path / d).is_dir()
        ]

        random.shuffle(bag_ids)

        n_total = len(bag_ids)
        n_train = floor(n_total * TRAIN_RATIO)
        n_val = floor(n_total * VAL_RATIO)

        train_ids = bag_ids[:n_train]
        val_ids = bag_ids[n_train:n_train + n_val]
        test_ids = bag_ids[n_train + n_val:]

        print(
            f"{model}: total={n_total} | "
            f"train={len(train_ids)} | "
            f"val={len(val_ids)} | "
            f"test={len(test_ids)}"
        )

        for split_name, ids in {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }.items():
            for bag_id in ids:
                src = model_path / bag_id
                dst = SPLIT_DIR / split_name / model / bag_id
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst)

    return True

def print_split_counts():
    for split in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split
        print(f"\nSplit: {split}")

        if not split_path.exists():
            print("  no existe")
            continue

        for model in os.listdir(split_path):
            model_path = split_path / model
            if model_path.is_dir():
                n = len([
                    d for d in os.listdir(model_path)
                    if (model_path / d).is_dir()
                ])
                print(f"  {model}: {n} carteras")

if __name__ == "__main__":
    split_by_bag()
    print_split_counts()
