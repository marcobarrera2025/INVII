import os
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_EXTS

class HandbagDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        root_dir:
          data/split/train
          data/split/val
          data/split/test
        '''
        self.samples = []
        self.transform = transform
        root_dir = str(root_dir)

        if not os.path.exists(root_dir):
            print(f"No existe: {root_dir}")
            return

        for model in sorted(os.listdir(root_dir)):
            model_path = os.path.join(root_dir, model)

            if not os.path.isdir(model_path):
                continue

            for bag_id in os.listdir(model_path):
                bag_path = os.path.join(model_path, bag_id)

                if not os.path.isdir(bag_path):
                    continue

                for img_name in os.listdir(bag_path):
                    if img_name.lower().endswith(IMAGE_EXTS):
                        self.samples.append(
                            os.path.join(bag_path, img_name)
                        )

        print(f"{len(self.samples)} imágenes encontradas en {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
