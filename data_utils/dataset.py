"""PyTorch Dataset for SRE-YOLO joint SR + detection training."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class SREYoloDataset(Dataset):
    """Loads (LR image, HR image, detection labels) triplets.

    Args:
        split_file: Path to a .txt file listing image stems (one per line).
        lr_dir: Directory containing 640x640 LR images.
        hr_dir: Directory containing 1280x1280 HR images.
        label_dir: Directory containing YOLO-format .txt label files.
        transform: Optional transform applied to both LR and HR images.
    """

    def __init__(
        self,
        split_file: Path,
        lr_dir: Path,
        hr_dir: Path,
        label_dir: Path,
        transform: Optional[Callable] = None,
    ):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.samples: List[str] = Path(split_file).read_text().splitlines()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.samples[idx]
        lr_img = Image.open(self.lr_dir / f"{stem}.jpg").convert("RGB")
        hr_img = Image.open(self.hr_dir / f"{stem}.jpg").convert("RGB")

        label_path = self.label_dir / f"{stem}.txt"
        labels = self._load_labels(label_path)

        if self.transform:
            lr_img, hr_img, labels = self.transform(lr_img, hr_img, labels)

        return lr_img, hr_img, labels

    @staticmethod
    def _load_labels(path: Path) -> torch.Tensor:
        """Read YOLO-format labels → (N, 5) tensor [cls, cx, cy, w, h]."""
        if not path.exists():
            return torch.zeros((0, 5))
        rows = []
        for line in path.read_text().splitlines():
            parts = list(map(float, line.strip().split()))
            if parts:
                rows.append(parts)
        return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5))
