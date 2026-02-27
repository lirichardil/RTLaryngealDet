"""Joint SR + detection loss for SRE-YOLO."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SREYoloLoss(nn.Module):
    """Weighted combination of SR pixel loss and YOLO detection losses.

    Loss = c1*L_sr + c2*L_box + c3*L_obj + c4*L_cls

    Args:
        c1: SR loss weight.
        c2: Box regression loss weight.
        c3: Objectness loss weight.
        c4: Classification loss weight.
    """

    def __init__(self, c1: float = 0.1, c2: float = 7.5, c3: float = 1.5, c4: float = 0.5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.bce = nn.BCEWithLogitsLoss()

    def sr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    def forward(
        self,
        sr_pred: torch.Tensor,
        hr_target: torch.Tensor,
        det_preds: list,
        det_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total weighted loss.

        Args:
            sr_pred: SR branch output (B, 3, H_hr, W_hr).
            hr_target: Ground-truth HR image (B, 3, H_hr, W_hr).
            det_preds: List of detection head outputs per scale.
            det_targets: YOLO-format targets (N, 6) [batch_idx, cls, cx, cy, w, h].

        Returns:
            Scalar total loss.
        """
        l_sr = self.sr_loss(sr_pred, hr_target) if sr_pred is not None else torch.tensor(0.0)

        # Placeholder detection losses — replace with full YOLO loss implementation
        l_box = torch.tensor(0.0, requires_grad=True)
        l_obj = torch.tensor(0.0, requires_grad=True)
        l_cls = torch.tensor(0.0, requires_grad=True)

        total = self.c1 * l_sr + self.c2 * l_box + self.c3 * l_obj + self.c4 * l_cls
        return total
