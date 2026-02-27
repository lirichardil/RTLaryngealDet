"""Data augmentation transforms for SRE-YOLO."""
from typing import Callable

import torchvision.transforms as T


def get_transforms(train: bool = True, lr_size: int = 640) -> Callable:
    """Return a torchvision transform pipeline.

    Args:
        train: If True, include random augmentations.
        lr_size: LR image size used for normalisation reference.

    Returns:
        A callable transform (applied to PIL images).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
