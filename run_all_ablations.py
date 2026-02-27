"""Run the full ablation study grid defined in training/ablation.py."""
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from models import SREYolo
from data_utils import SREYoloDataset, get_transforms
from training import Trainer, run_ablation
from training.ablation import ABLATION_GRID


def parse_args():
    p = argparse.ArgumentParser(description="SRE-YOLO ablation runner")
    p.add_argument("--train-cfg",   default="configs/train.yaml")
    p.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p.add_argument("--model-cfg",   default="configs/model.yaml")
    p.add_argument("--split",       default="split1", choices=["split1", "split2", "split3"])
    p.add_argument("--results-dir", default="results/ablations")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    train_cfg   = yaml.safe_load(Path(args.train_cfg).read_text())
    dataset_cfg = yaml.safe_load(Path(args.dataset_cfg).read_text())
    base_model_cfg = yaml.safe_load(Path(args.model_cfg).read_text())

    torch.manual_seed(train_cfg["seed"])

    split = dataset_cfg["splits"][args.split]

    def make_loader(split_file, train: bool) -> DataLoader:
        ds = SREYoloDataset(
            split_file=Path(split_file),
            lr_dir=Path(dataset_cfg["lr_dir"]),
            hr_dir=Path(dataset_cfg["hr_dir"]),
            label_dir=Path(dataset_cfg["raw_dir"]),
            transform=get_transforms(train=train, lr_size=train_cfg["imgsz_lr"]),
        )
        return DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=train, num_workers=4)

    train_loader = make_loader(split["train"], train=True)
    val_loader   = make_loader(split["val"],   train=False)

    results_dir = Path(args.results_dir)

    for variant in ABLATION_GRID:
        tag = variant.get("tag", "unnamed")
        overrides = {k: v for k, v in variant.items() if k != "tag"}
        model_cfg = {**base_model_cfg, **overrides}

        out_dir = results_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\nAblation: {tag}\n{'='*60}")

        model   = SREYolo(model_cfg)
        trainer = Trainer(
            model, train_cfg,
            device=args.device,
            weights_dir=out_dir / "weights",
        )
        trainer.fit(train_loader, val_loader)
        print(f"[{tag}] Training complete. Weights in {out_dir / 'weights'}")


if __name__ == "__main__":
    main()
