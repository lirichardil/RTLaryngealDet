"""Entry point: train SRE-YOLO on a single split."""
import argparse
from pathlib import Path

import yaml
import torch

from models import SREYolo
from data_utils import SREYoloDataset, get_transforms
from training import Trainer
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train SRE-YOLO")
    p.add_argument("--train-cfg",   default="configs/train.yaml")
    p.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p.add_argument("--model-cfg",   default="configs/model.yaml")
    p.add_argument("--split",       default="split1", choices=["split1", "split2", "split3"])
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights-dir", default="weights")
    return p.parse_args()


def main():
    args = parse_args()

    train_cfg   = yaml.safe_load(Path(args.train_cfg).read_text())
    dataset_cfg = yaml.safe_load(Path(args.dataset_cfg).read_text())
    model_cfg   = yaml.safe_load(Path(args.model_cfg).read_text())

    torch.manual_seed(train_cfg["seed"])

    split = dataset_cfg["splits"][args.split]
    train_ds = SREYoloDataset(
        split_file=Path(split["train"]),
        lr_dir=Path(dataset_cfg["lr_dir"]),
        hr_dir=Path(dataset_cfg["hr_dir"]),
        label_dir=Path(dataset_cfg["raw_dir"]),
        transform=get_transforms(train=True, lr_size=train_cfg["imgsz_lr"]),
    )
    val_ds = SREYoloDataset(
        split_file=Path(split["val"]),
        lr_dir=Path(dataset_cfg["lr_dir"]),
        hr_dir=Path(dataset_cfg["hr_dir"]),
        label_dir=Path(dataset_cfg["raw_dir"]),
        transform=get_transforms(train=False, lr_size=train_cfg["imgsz_lr"]),
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4)

    model   = SREYolo(model_cfg)
    trainer = Trainer(model, train_cfg, device=args.device, weights_dir=Path(args.weights_dir))
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
