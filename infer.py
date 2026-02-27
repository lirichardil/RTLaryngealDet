"""Entry point: run inference on a single image or directory."""
import argparse
from pathlib import Path

import torch
import cv2
import yaml

from models import SREYolo
from data_utils import get_transforms
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="SRE-YOLO inference")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--source",      required=True, help="Image path or directory")
    p.add_argument("--train-cfg",   default="configs/train.yaml")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",     default="results/infer")
    return p.parse_args()


def infer_image(model, img_path: Path, transform, device: str, out_dir: Path):
    img_pil = Image.open(img_path).convert("RGB")
    tensor  = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_out, det_preds = model(tensor)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save SR output
    if sr_out is not None:
        sr_np = sr_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_np = (sr_np * 255).clip(0, 255).astype("uint8")
        cv2.imwrite(str(out_dir / f"{img_path.stem}_sr.jpg"), cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))

    print(f"Processed {img_path.name} → {out_dir}")


def main():
    args = parse_args()
    train_cfg = yaml.safe_load(Path(args.train_cfg).read_text())

    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    model = SREYolo(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])

    transform = get_transforms(train=False, lr_size=train_cfg["imgsz_lr"])
    out_dir   = Path(args.out_dir)
    source    = Path(args.source)

    images = list(source.rglob("*.[jp][pn]g")) if source.is_dir() else [source]
    for img_path in images:
        infer_image(model, img_path, transform, args.device, out_dir)


if __name__ == "__main__":
    main()
