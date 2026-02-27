# SRE-YOLO — Super-Resolution Enhanced YOLO for Laryngeal Lesion Detection

A joint super-resolution + object detection framework built on YOLOv8, designed for small-lesion detection in laryngoscopy imagery.

---

## Project Structure

```
sre_yolo/
├── configs/
│   ├── dataset.yaml      # dataset paths + class names
│   ├── train.yaml        # all hyperparameters
│   └── model.yaml        # architecture flags
├── data/
│   ├── raw/              # original downloaded images
│   └── processed/
│       ├── lr/           # 640×640 LR inputs
│       └── hr/           # 1280×1280 HR SR targets
├── data/splits/          # split1/ split2/ split3/ — train/val/test.txt
├── models/               # backbone, SR branch, full SRE-YOLO model
├── data_utils/           # preprocessing, dataset, augmentation
├── training/             # loss, trainer, ablation runner
├── evaluation/           # mAP + SR metrics
├── tests/
├── weights/              # saved checkpoints
├── results/              # evaluation outputs, ablation results
├── train.py
├── evaluate.py
├── infer.py
├── export.py
└── run_all_ablations.py
```

---

## Quick Start

### 1. Preprocess raw data
```bash
python -c "
from data_utils.preprocess import preprocess_dataset
from pathlib import Path
preprocess_dataset(Path('data/raw'), Path('data/processed/lr'), Path('data/processed/hr'))
"
```

### 2. Train
```bash
python train.py --split split1
```

### 3. Evaluate
```bash
python evaluate.py --checkpoint weights/best.pt --split split1 --subset test
```

### 4. Inference
```bash
python infer.py --checkpoint weights/best.pt --source data/raw/sample.jpg
```

### 5. Export
```bash
python export.py --checkpoint weights/best.pt --format onnx
```

### 6. Run ablation study
```bash
python run_all_ablations.py --split split1 --results-dir results/ablations
```

---

## Key Hyperparameters (`configs/train.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr0` | 0.05 | Initial learning rate |
| `lrf` | 0.2 | Final LR multiplier |
| `optimizer` | AdamW | Optimiser |
| `epochs` | 100 | Max training epochs |
| `patience` | 50 | Early stopping patience |
| `batch_size` | 16 | Batch size |
| `imgsz_lr` | 640 | LR input resolution |
| `imgsz_hr` | 1280 | HR SR target resolution |
| `c1–c4` | 0.1 / 7.5 / 1.5 / 0.5 | SR / box / obj / cls loss weights |
| `sr_resblocks` | 16 | Residual blocks in SR decoder |
| `sr_decoder` | deep | Decoder variant |

---

## Classes (`configs/dataset.yaml`)

| ID | Name |
|----|------|
| 0 | normal |
| 1 | leukoplakia |
| 2 | erythroplakia |
| 3 | carcinoma |
