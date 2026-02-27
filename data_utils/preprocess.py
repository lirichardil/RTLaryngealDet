"""Preprocessing pipeline: raw images → LR (640) and HR (1280) pairs."""
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

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


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def filter_frames(
    src_dir: Path,
    dst_dir: Path,
    blur_thresh: float = 100.0,
    bright_min: int = 30,
    bright_max: int = 240,
) -> Tuple[int, int]:
    """Copy quality frames from src_dir to dst_dir, excluding poor-quality images.

    Exclusion rules (first match wins):
      - Tiny:  file size < 5 KB
      - Blur:  Laplacian variance of grayscale < blur_thresh
      - Dark:  mean pixel value < bright_min
      - Blown: mean pixel value > bright_max

    Excluded files are logged to results/filter_log.csv.

    Args:
        src_dir: Directory containing raw frames.
        dst_dir: Directory where kept frames are written.
        blur_thresh: Minimum acceptable Laplacian variance.
        bright_min: Minimum acceptable mean pixel value.
        bright_max: Maximum acceptable mean pixel value.

    Returns:
        (n_kept, n_excluded) counts.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    log_path = Path("results/filter_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in src_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]

    n_kept = 0
    n_excluded = 0

    with log_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "reason"])

        for img_path in candidates:
            reason = _exclusion_reason(img_path, blur_thresh, bright_min, bright_max)

            if reason:
                writer.writerow([img_path.name, reason])
                n_excluded += 1
            else:
                rel = img_path.relative_to(src_dir)
                out_path = dst_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_path)
                n_kept += 1

    return n_kept, n_excluded


def _exclusion_reason(
    img_path: Path,
    blur_thresh: float,
    bright_min: int,
    bright_max: int,
) -> str:
    """Return a non-empty reason string if the image should be excluded, else ''."""
    if img_path.stat().st_size < 5 * 1024:
        return "tiny"

    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thresh:
        return "blur"

    mean_val = float(np.mean(gray))
    if mean_val < bright_min:
        return "dark"
    if mean_val > bright_max:
        return "blown"

    return ""


def create_dual_resolution(
    src_dir: Path,
    dst_lr: Path,
    dst_hr: Path,
) -> int:
    """Create LR (640×640) and HR (1280×1280) image pairs with scaled labels.

    Expects src_dir layout::

        src_dir/images/<stem>.<ext>
        src_dir/labels/<stem>.txt   # YOLO: class cx cy w h (pixel coords)

    Outputs::

        dst_lr/images/<stem>.jpg    # INTER_LINEAR resize
        dst_lr/labels/<stem>.txt    # coords scaled to 640×640 pixel space
        dst_hr/images/<stem>.jpg    # INTER_CUBIC resize — NO labels (SR target)

    LR and HR share an identical <stem> so they pair by name.

    Args:
        src_dir: Root containing images/ and labels/ subdirectories.
        dst_lr:  Root for LR outputs.
        dst_hr:  Root for HR outputs.

    Returns:
        Number of (image, label) pairs successfully processed.
    """
    src_dir = Path(src_dir)
    dst_lr = Path(dst_lr)
    dst_hr = Path(dst_hr)

    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"

    (dst_lr / "images").mkdir(parents=True, exist_ok=True)
    (dst_lr / "labels").mkdir(parents=True, exist_ok=True)
    (dst_hr / "images").mkdir(parents=True, exist_ok=True)

    candidates = [p for p in img_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]

    n_pairs = 0
    for img_path in candidates:
        lbl_path = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        stem = img_path.stem

        # LR: resize with INTER_LINEAR + write scaled labels
        lr_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_lr / "images" / f"{stem}.jpg"), lr_img)
        _write_scaled_labels(
            lbl_path,
            dst_lr / "labels" / f"{stem}.txt",
            sx=640 / orig_w,
            sy=640 / orig_h,
        )

        # HR: resize with INTER_CUBIC — no labels (SR reconstruction target only)
        hr_img = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(dst_hr / "images" / f"{stem}.jpg"), hr_img)

        n_pairs += 1

    return n_pairs


def _write_scaled_labels(src: Path, dst: Path, sx: float, sy: float) -> None:
    """Read YOLO label file, scale bbox coords, write result to dst.

    Each source line: ``class_id cx cy w h``
    cx and w are multiplied by sx; cy and h by sy.
    """
    lines_out = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx *= sx
        cy *= sy
        w  *= sx
        h  *= sy
        lines_out.append(f"{class_id} {cx} {cy} {w} {h}")
    dst.write_text("\n".join(lines_out))


def make_splits(
    metadata_csv: Path,
    n_splits: int = 3,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """Create patient-level train/val/test splits and write split .txt files.

    Reads ``metadata_csv`` (columns: filename, patient_id, dataset_source) and
    produces ``n_splits`` independent splits, each seeded differently so patient
    ordering varies.  For each split the patients are divided::

        first 89 % → train  |  next 7 % → val  |  last 4 % → test

    Each .txt file contains one absolute LR image path per line::

        data/splits/split{i}/train.txt
        data/splits/split{i}/val.txt
        data/splits/split{i}/test.txt

    Args:
        metadata_csv: Path to CSV with columns [filename, patient_id, dataset_source].
        n_splits:     Number of independent splits to generate.
        seed:         Base random seed; split i uses seed + i.

    Returns:
        ``{"split1": {"train": N, "val": N, "test": N}, ...}``

    Raises:
        AssertionError: if any train patient also appears in the test set.
    """
    metadata_csv = Path(metadata_csv)
    lr_img_dir = Path("data/processed/lr/images")

    # ── load CSV ──────────────────────────────────────────────────────────────
    patient_files: Dict[str, list] = defaultdict(list)
    with metadata_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            patient_files[row["patient_id"]].append(row["filename"])

    # ── header ────────────────────────────────────────────────────────────────
    print(
        f"{'split':>6} | {'train_patients':>14} | {'val_patients':>12} | "
        f"{'test_patients':>13} | {'train_frames':>12}"
    )
    print("-" * 70)

    results: Dict[str, Dict[str, int]] = {}

    for split_idx in range(n_splits):
        rng = random.Random(seed + split_idx)

        patients = sorted(patient_files.keys())   # deterministic base order
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * 0.89)
        n_val = int(n * 0.07)
        # test gets all remaining patients (absorbs rounding remainder)

        train_pats = patients[:n_train]
        val_pats   = patients[n_train : n_train + n_val]
        test_pats  = patients[n_train + n_val :]

        # ── hard assertion: zero patient overlap between train and test ───────
        assert not (set(train_pats) & set(test_pats)), (
            f"Split {split_idx + 1}: train/test patient overlap detected: "
            f"{set(train_pats) & set(test_pats)}"
        )

        def _paths(pat_list: list) -> list:
            return [
                str((lr_img_dir / fname).resolve())
                for pid in pat_list
                for fname in patient_files[pid]
            ]

        train_paths = _paths(train_pats)
        val_paths   = _paths(val_pats)
        test_paths  = _paths(test_pats)

        split_name = f"split{split_idx + 1}"
        split_dir = Path(f"data/splits/{split_name}")
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "train.txt").write_text("\n".join(train_paths))
        (split_dir / "val.txt").write_text("\n".join(val_paths))
        (split_dir / "test.txt").write_text("\n".join(test_paths))

        results[split_name] = {
            "train": len(train_paths),
            "val":   len(val_paths),
            "test":  len(test_paths),
        }

        print(
            f"{split_idx + 1:>6} | {len(train_pats):>14} | {len(val_pats):>12} | "
            f"{len(test_pats):>13} | {len(train_paths):>12}"
        )

    return results
