import torch
import torch.nn as nn
from torchvision import models

from config import EMBEDDING_DIM

class EfficientNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()

        # Backbone preentrenado
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Dimensión de salida del backbone
        in_features = self.backbone.classifier[1].in_features

        # Quitamos el clasificador original
        self.backbone.classifier = nn.Identity()

        # Proyección a embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        emb = self.embedding(features)
        return emb

def create_model(device):
    model = EfficientNetEmbedding(embedding_dim=EMBEDDING_DIM).to(device)
    return model
