"""Export a trained SRE-YOLO checkpoint to ONNX or TorchScript."""
import argparse
from pathlib import Path

import torch
import yaml

from models import SREYolo


def parse_args():
    p = argparse.ArgumentParser(description="Export SRE-YOLO")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format",     default="onnx", choices=["onnx", "torchscript"])
    p.add_argument("--train-cfg",  default="configs/train.yaml")
    p.add_argument("--out",        default="weights/sre_yolo_export")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    train_cfg = yaml.safe_load(Path(args.train_cfg).read_text())

    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    model = SREYolo(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])

    imgsz = train_cfg["imgsz_lr"]
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=args.device)

    if args.format == "onnx":
        out_path = Path(args.out).with_suffix(".onnx")
        torch.onnx.export(
            model, dummy, str(out_path),
            input_names=["lr_image"],
            output_names=["sr_out", "det_preds"],
            opset_version=17,
            dynamic_axes={"lr_image": {0: "batch"}},
        )
        print(f"ONNX model saved → {out_path}")

    elif args.format == "torchscript":
        out_path = Path(args.out).with_suffix(".pt")
        traced  = torch.jit.trace(model, dummy)
        traced.save(str(out_path))
        print(f"TorchScript model saved → {out_path}")


if __name__ == "__main__":
    main()
