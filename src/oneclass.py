import torch

@torch.no_grad()
def compute_center(model, dataloader, device):
    model.eval()
    embeddings = []

    for batch in dataloader:
        batch = batch.to(device)
        emb = model(batch)
        embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    center = embeddings.mean(dim=0).to(device)

    return center

class OneClassLoss(torch.nn.Module):
    def __init__(self, center):
        super().__init__()
        self.register_buffer("center", center)

    def forward(self, embeddings):
        distances = torch.sum((embeddings - self.center) ** 2, dim=1)
        return distances.mean()

@torch.no_grad()
def compute_distances(model, dataloader, center, device):
    model.eval()
    all_distances = []

    for batch in dataloader:
        batch = batch.to(device)
        emb = model(batch)
        distances = torch.sum((emb - center) ** 2, dim=1)
        all_distances.append(distances.cpu())

    all_distances = torch.cat(all_distances).numpy()
    return all_distances
