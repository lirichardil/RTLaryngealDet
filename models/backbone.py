"""Backbone builder for SRE-YOLO."""
import torch.nn as nn


def build_backbone(name: str = "cspdarknet", pretrained: bool = True) -> nn.Module:
    """Return a backbone network by name.

    Args:
        name: One of 'cspdarknet', 'resnet50', 'efficientnet'.
        pretrained: Load ImageNet / COCO pre-trained weights when available.

    Returns:
        nn.Module with a ``forward`` that yields a list of feature maps.
    """
    raise NotImplementedError(f"Backbone '{name}' not yet implemented.")
