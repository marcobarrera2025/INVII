from torchvision import transforms
from config import IMG_SIZE

# FIEL AL NOTEBOOK:
# train_transform = RandomResizedCrop + ColorJitter + RandomApply GaussianBlur
# + ToTensor + RandomErasing + Normalize

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.75, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3)],
        p=0.3
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.06)
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# FIEL AL NOTEBOOK:
# eval_transform = Resize(256) + CenterCrop(224) + ToTensor + Normalize

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
