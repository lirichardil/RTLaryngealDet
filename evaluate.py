"""Entry point: evaluate a trained SRE-YOLO checkpoint."""
import argparse
import json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from models import SREYolo
from data_utils import SREYoloDataset, get_transforms
from evaluation import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SRE-YOLO")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p.add_argument("--train-cfg",   default="configs/train.yaml")
    p.add_argument("--split",       default="split1", choices=["split1", "split2", "split3"])
    p.add_argument("--subset",      default="test",   choices=["val", "test"])
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",         default="results/eval.json")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_cfg = yaml.safe_load(Path(args.dataset_cfg).read_text())
    train_cfg   = yaml.safe_load(Path(args.train_cfg).read_text())

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = SREYolo(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    split_files = dataset_cfg["splits"][args.split]
    ds = SREYoloDataset(
        split_file=Path(split_files[args.subset]),
        lr_dir=Path(dataset_cfg["lr_dir"]),
        hr_dir=Path(dataset_cfg["hr_dir"]),
        label_dir=Path(dataset_cfg["raw_dir"]),
        transform=get_transforms(train=False, lr_size=train_cfg["imgsz_lr"]),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    predictions, ground_truths = [], []
    with torch.no_grad():
        for lr_imgs, hr_imgs, labels in loader:
            lr_imgs = lr_imgs.to(args.device)
            _, det_preds = model(lr_imgs)
            # TODO: decode det_preds → boxes/scores/labels dicts
            predictions.append({})
            ground_truths.append({})

    nc = dataset_cfg.get("nc", 4)
    results = compute_metrics(
        predictions, ground_truths,
        iou_thresh=train_cfg["iou_thresh"],
        conf_thresh=train_cfg["conf_thresh"],
        nc=nc,
    )
    print(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
