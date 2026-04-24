import os
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from config import TRAIN_DIR, FAKE_HARD_DIR, IMAGE_EXTS, FAKES_PER_BAG, RANDOM_SEED

random.seed(RANDOM_SEED)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def occlusion_tensor(img):
    _, h, w = img.shape
    occ_w = int(w * random.uniform(0.2, 0.4))
    occ_h = int(h * random.uniform(0.2, 0.4))
    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)
    return F.erase(img, y, x, occ_h, occ_w, v=0)

def warp_tensor(img):
    return F.affine(
        img,
        angle=random.uniform(-40, 40),
        translate=(0, 0),
        scale=random.uniform(0.7, 1.1),
        shear=random.uniform(-25, 25)
    )

def generate_fake_hard():
    src_dir = TRAIN_DIR
    dst_dir = FAKE_HARD_DIR

    for model_name in os.listdir(src_dir):
        mpath = src_dir / model_name
        if not mpath.is_dir():
            continue

        for bag in os.listdir(mpath):
            bpath = mpath / bag
            if not bpath.is_dir():
                continue

            imgs = [f for f in os.listdir(bpath) if f.lower().endswith(IMAGE_EXTS)]
            if not imgs:
                continue

            for k in range(FAKES_PER_BAG):
                fake_bag = f"{bag}_fake_hard{k}"
                dst = dst_dir / model_name / fake_bag
                dst.mkdir(parents=True, exist_ok=True)

                for img_name in imgs:
                    img = Image.open(bpath / img_name).convert("RGB")

                    x = to_tensor(img)
                    x = occlusion_tensor(x)
                    x = warp_tensor(x)

                    x = transforms.ColorJitter(
                        brightness=0.8,
                        contrast=0.8,
                        saturation=0.8,
                        hue=0.15
                    )(x)

                    x = transforms.GaussianBlur(kernel_size=9)(x)
                    img_fake = to_pil(x)

                    img_fake.save(dst / img_name)

    print("Carteras fake HARD creadas correctamente")

if __name__ == "__main__":
    generate_fake_hard()
