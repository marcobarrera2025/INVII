import json
import numpy as np
import torch
from tqdm import tqdm

from config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_PATH,
    CENTER_PATH,
    METRICS_PATH,
    THRESHOLD_PATH,
    PERCENTILE,
    MODELS_DIR,
    METRICS_DIR,
)
from dataloaders import build_dataloaders
from model import create_model
from oneclass import compute_center, OneClassLoss, compute_distances

def reject_rate(dist, thr):
    return float((dist > thr).mean())

def train_oneclass():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders()

    if len(train_loader.dataset) == 0:
        print("No hay imágenes de entrenamiento.")
        return False

    model = create_model(device)

    # Igual que notebook: primero calcular centro con el modelo inicial
    center = compute_center(model, train_loader, device)

    # Guardar centro inicial
    torch.save(center, CENTER_PATH)
    print("Centro guardado en:", CENTER_PATH)

    loss_fn = OneClassLoss(center).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            batch = batch.to(device)

            optimizer.zero_grad()
            embeddings = model(batch)
            loss = loss_fn(embeddings)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(train_loss)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                loss = loss_fn(embeddings)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.6f} | "
            f"Val Loss = {val_loss:.6f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print("Modelo guardado en:", MODEL_PATH)

    train_dist = compute_distances(model, train_loader, center, device)
    val_dist = compute_distances(model, val_loader, center, device)
    test_dist = compute_distances(model, test_loader, center, device)

    print("Distancias:")
    print("Train:", train_dist.mean(), "+/-", train_dist.std())
    print("Val  :", val_dist.mean(), "+/-", val_dist.std())
    print("Test :", test_dist.mean(), "+/-", test_dist.std())

    np.savez(
        METRICS_PATH,
        train_dist=train_dist,
        val_dist=val_dist,
        test_dist=test_dist,
        train_loss=history["train_loss"],
        val_loss=history["val_loss"]
    )

    threshold = float(np.percentile(val_dist, PERCENTILE))

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"percentile": PERCENTILE, "threshold": threshold},
            f,
            indent=2
        )

    print("Umbral =", threshold)
    print("Rechazo genuinos (Train):", reject_rate(train_dist, threshold))
    print("Rechazo genuinos (Val)  :", reject_rate(val_dist, threshold))
    print("Rechazo genuinos (Test) :", reject_rate(test_dist, threshold))

    return True

if __name__ == "__main__":
    train_oneclass()
