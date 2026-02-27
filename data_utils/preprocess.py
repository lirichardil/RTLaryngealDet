"""Preprocessing pipeline: raw images → LR (640) and HR (1280) pairs."""
from pathlib import Path

import cv2
import numpy as np


def preprocess(
    src: Path,
    lr_out: Path,
    hr_out: Path,
    lr_size: int = 640,
    hr_size: int = 1280,
) -> None:
    """Resize a single image to LR and HR and save both.

    Args:
        src: Path to the original image.
        lr_out: Destination path for the LR image.
        hr_out: Destination path for the HR image.
        lr_size: Target LR resolution (square).
        hr_size: Target HR resolution (square).
    """
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {src}")

    lr_img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
    hr_img = cv2.resize(img, (hr_size, hr_size), interpolation=cv2.INTER_LANCZOS4)

    lr_out.parent.mkdir(parents=True, exist_ok=True)
    hr_out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(lr_out), lr_img)
    cv2.imwrite(str(hr_out), hr_img)


def preprocess_dataset(
    raw_dir: Path,
    lr_dir: Path,
    hr_dir: Path,
    lr_size: int = 640,
    hr_size: int = 1280,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
) -> None:
    """Batch-preprocess all images in raw_dir."""
    images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in extensions]
    for img_path in images:
        rel = img_path.relative_to(raw_dir)
        preprocess(
            src=img_path,
            lr_out=lr_dir / rel,
            hr_out=hr_dir / rel,
            lr_size=lr_size,
            hr_size=hr_size,
        )
    print(f"Preprocessed {len(images)} images.")
