import os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from config import (
    VAL_DIR,
    FAKE_HARD_DIR,
    IMAGE_EXTS,
    MODEL_PATH,
    CENTER_PATH,
    ROC_PATH,
    OUTPUTS_DIR,
)
from model import create_model
from transforms_pipeline import eval_transform

def score_bags(root_dir, label, model, center, device):
    scores, labels = [], []

    with torch.no_grad():
        for model_name in os.listdir(root_dir):
            mpath = root_dir / model_name
            if not mpath.is_dir():
                continue

            for bag in os.listdir(mpath):
                bpath = mpath / bag
                if not bpath.is_dir():
                    continue

                dists = []

                for img_name in os.listdir(bpath):
                    if img_name.lower().endswith(IMAGE_EXTS):
                        img = Image.open(bpath / img_name).convert("RGB")
                        x = eval_transform(img).unsqueeze(0).to(device)
                        emb = model(x)
                        d = torch.sum((emb - center) ** 2, dim=1).item()
                        dists.append(d)

                if dists:
                    scores.append(max(dists))
                    labels.append(label)

    return scores, labels

def evaluate_roc_auc():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    center = torch.load(CENTER_PATH, map_location=device)

    g_scores, g_labels = score_bags(VAL_DIR, 1, model, center, device)
    f_scores, f_labels = score_bags(FAKE_HARD_DIR, 0, model, center, device)

    scores = np.array(g_scores + f_scores)
    y_true = np.array(g_labels + f_labels)

    print("Total carteras:", len(scores))
    print("Genuinas:", len(g_scores), "Fake:", len(f_scores))

    scores_inv = -scores

    roc_auc = roc_auc_score(y_true, scores_inv)
    print("ROC-AUC:", roc_auc)

    fpr, tpr, _ = roc_curve(y_true, scores_inv)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-Class Handbag Authentication)")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_PATH, dpi=150)
    print("ROC guardado en:", ROC_PATH)

    return roc_auc

if __name__ == "__main__":
    evaluate_roc_auc()
