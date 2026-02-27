"""Training loop for SRE-YOLO."""
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import SREYolo
from training.loss import SREYoloLoss


class Trainer:
    """Manages training, validation, checkpointing, and early stopping.

    Args:
        model: Instantiated SREYolo model.
        cfg: Parsed train.yaml dict.
        device: torch device string (e.g. 'cuda' or 'cpu').
        weights_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        model: SREYolo,
        cfg: Dict,
        device: str = "cuda",
        weights_dir: Path = Path("weights"),
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = SREYoloLoss(
            c1=cfg["c1"], c2=cfg["c2"], c3=cfg["c3"], c4=cfg["c4"]
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg["lr0"] * cfg["lrf"],
        )

    def _build_optimizer(self) -> optim.Optimizer:
        name = self.cfg.get("optimizer", "AdamW")
        lr = self.cfg["lr0"]
        if name == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=lr)
        if name == "SGD":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.937, nesterov=True)
        raise ValueError(f"Unknown optimizer: {name}")

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for lr_imgs, hr_imgs, labels in loader:
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            sr_out, det_preds = self.model(lr_imgs)
            loss = self.criterion(sr_out, hr_imgs, det_preds, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_loss = float("inf")
        patience_counter = 0
        epochs = self.cfg["epochs"]
        patience = self.cfg["patience"]

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)
            self.scheduler.step()

            print(f"Epoch {epoch}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self._save_checkpoint("last.pt")

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs, labels in loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                labels = labels.to(self.device)
                sr_out, det_preds = self.model(lr_imgs)
                loss = self.criterion(sr_out, hr_imgs, det_preds, labels)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _save_checkpoint(self, name: str) -> None:
        path = self.weights_dir / name
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)
        print(f"Saved checkpoint → {path}")
