"""SRE-YOLO: Super-Resolution Enhanced YOLO for laryngeal lesion detection."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import build_backbone
from .sr_branch import SRBranch


class SREYolo(nn.Module):
    """Joint SR + detection model.

    Args:
        cfg: Parsed model.yaml dict.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(
            name=cfg.get("backbone", "cspdarknet"),
            pretrained=cfg.get("backbone_pretrained", True),
        )
        if cfg.get("sr_enabled", True):
            # SR branch is attached to the first tapped layer
            self.sr_branch = SRBranch(
                in_channels=256,  # placeholder — set after backbone is implemented
                upscale=cfg.get("sr_upscale", 2),
                n_resblocks=cfg.get("sr_resblocks", 16),
                decoder=cfg.get("sr_decoder", "deep"),
            )
        else:
            self.sr_branch = None

        # Detection head placeholder
        self.det_head = None  # implement after backbone

    def forward(
        self, lr_img: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """Forward pass.

        Returns:
            sr_out: Super-resolved image tensor, or None if SR disabled.
            det_preds: List of detection prediction tensors per scale.
        """
        features = self.backbone(lr_img)
        sr_out = self.sr_branch(features[self.cfg["sr_layers"][0]]) if self.sr_branch else None
        det_preds = []  # self.det_head(features)
        return sr_out, det_preds
