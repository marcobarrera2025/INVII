import argparse
import json
import os
import numpy as np
import torch
from PIL import Image

from config import (
    BAG_DIR,
    IMAGE_EXTS,
    MODEL_PATH,
    CENTER_PATH,
    THRESHOLD_PATH,
)
from model import create_model
from transforms_pipeline import eval_transform

def predict_bag(bag_dir=BAG_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    center = torch.load(CENTER_PATH, map_location=device)

    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
        threshold = float(thr_data["threshold"])
    else:
        threshold = 169.76677

    assert os.path.isdir(bag_dir), "Carpeta no existe"

    imgs = [f for f in os.listdir(bag_dir) if f.lower().endswith(IMAGE_EXTS)]
    assert len(imgs) > 0, "No hay imágenes válidas"

    print(f"Imágenes encontradas: {len(imgs)}")

    distances = []

    with torch.no_grad():
        for img_name in imgs:
            img = Image.open(os.path.join(bag_dir, img_name)).convert("RGB")
            x = eval_transform(img).unsqueeze(0).to(device)
            emb = model(x)
            dist = torch.sum((emb - center) ** 2, dim=1)
            distances.append(dist.item())

    mean_score = float(np.mean(distances))
    p95_score = float(np.percentile(distances, 95))
    max_score = float(np.max(distances))
    std_score = float(np.std(distances))

    print("\nRESULTADOS POR CARTERA")
    print(f"Mean : {mean_score:.4f}")
    print(f"P95  : {p95_score:.4f}")
    print(f"Max  : {max_score:.4f}")
    print(f"Std  : {std_score:.4f}")
    print(f"N    : {len(distances)}")

    if max_score <= threshold:
        decision = "AUTÉNTICO"
    else:
        decision = "RECHAZADO"

    print("\nDECISIÓN FINAL")
    print("Resultado :", decision)
    print("Threshold :", threshold)

    print("\nDETALLE POR IMAGEN")
    for img, d in zip(imgs, distances):
        print(f"{img:30s} dist={d:.4f}")

    return {
        "mean": mean_score,
        "p95": p95_score,
        "max": max_score,
        "std": std_score,
        "n": len(distances),
        "threshold": threshold,
        "decision": decision,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", default=str(BAG_DIR))
    args = parser.parse_args()

    predict_bag(args.bag_dir)
