# SRE-YOLO ML Learning Companion

**Key Concepts, Explanations & Study Resources for Every Section**

> Baldini et al. (2025) · Comput. Methods Programs Biomed. 260:108539

---

## 📖 How to Use This Document

- Before starting each section in the implementation spec, read the matching section here first.
- Watch the linked YouTube videos for any concept you are not yet confident with.
- Concepts are rated 🟢 Beginner · 🟡 Intermediate · 🔵 Advanced.
- You do not need to master every concept before starting — learn as you build!

---

## 🛠️ Section 1 — Environment Setup

> Before you can write a single line of ML code, you need a properly configured Python environment. These concepts underpin every step that follows.

---

### Conda & Virtual Environments · 🟢 Beginner

A virtual environment is an isolated Python installation that keeps your project's dependencies separate from other projects. Conda is a popular environment manager that handles both Python versions and packages. Without it, installing one project's libraries can break another's.

**🔗 In this project:** You'll create a `sre_yolo` environment with Python 3.10 and every library pinned to an exact version, ensuring your training runs are reproducible.

- ▶ [Conda Tutorial for Beginners — Anaconda](https://www.youtube.com/watch?v=sDCtY68QM4s) — 15 min intro to conda environments
- ▶ [Python Virtual Environments Full Guide](https://www.youtube.com/watch?v=IAvAlS0CuxI) — covers venv, pip, and conda differences

---

### pip & Package Management · 🟢 Beginner

pip is Python's package installer. It downloads libraries from PyPI (the Python Package Index). "Pinning" a version (e.g. `numpy==1.26.4`) means everyone gets the exact same code, preventing the classic "works on my machine" bug.

**🔗 In this project:** `environment.yml` pins every library (PyTorch, Ultralytics, OpenCV, etc.) so the training results from this paper can be exactly reproduced.

- ▶ [pip Tutorial — How to Install Python Packages](https://www.youtube.com/watch?v=U2ZN104hIcc) — pip basics in 10 min
- ▶ [Python Packaging Explained](https://www.youtube.com/watch?v=YM6cz5OHf14) — requirements.txt vs environment.yml

---

### CUDA & GPU Computing · 🟡 Intermediate

CUDA is NVIDIA's platform for running code on GPUs. Neural network training is fundamentally matrix multiplication at scale — GPUs do this 10–100× faster than CPUs because they have thousands of small parallel cores. PyTorch uses CUDA to accelerate all tensor operations.

**🔗 In this project:** YOLOv8n achieves 58.8 FPS because inference runs on a GPU. Without CUDA, training would take weeks instead of days.

- ▶ [But what is a GPU? (3Blue1Brown style)](https://www.youtube.com/watch?v=r9IQDQkSv_w) — visual intuition for GPU parallelism
- ▶ [CUDA Programming — Why GPUs for Deep Learning](https://www.youtube.com/watch?v=EMtPv1_bVMI) — explains why matrix ops map to GPU cores
- ▶ [PyTorch GPU Setup Guide](https://www.youtube.com/watch?v=UWlFM0R_x6I) — practical CUDA setup for deep learning

---

### PyTorch Fundamentals · 🟢 Beginner

PyTorch is the deep learning framework used throughout this project. It provides tensors (n-dimensional arrays like NumPy, but GPU-accelerated), automatic differentiation (autograd), and pre-built building blocks for neural networks (`nn.Module`).

**🔗 In this project:** Every model, loss function, and training loop is written in PyTorch. Understanding tensors and `nn.Module` is the single most important prerequisite.

- ▶ [PyTorch for Deep Learning — Full Course (freeCodeCamp)](https://www.youtube.com/watch?v=V_xro1bcAuA) — comprehensive 25-hour course — watch §1–3 first
- ▶ [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) — quick overview of tensors and autograd
- ▶ [nn.Module Explained](https://www.youtube.com/watch?v=GIkg3DkESA4) — how to build neural network layers in PyTorch

---

## 📁 Section 2 — Project Scaffold & Configuration

> Good project structure is not just tidiness — it makes the difference between code you can debug in 6 months and code that is a maze.

---

### ML Project Structure · 🟢 Beginner

A well-organised ML project separates concerns: data loading code lives in one place, model definitions in another, training logic in another. This mirrors software engineering best practices and makes it easy to swap out components.

**🔗 In this project:** The `sre_yolo/` scaffold has separate folders for `models/`, `data_utils/`, `training/`, and `evaluation/` — each section of the spec maps to exactly one folder.

- ▶ [Structuring Machine Learning Projects](https://www.youtube.com/watch?v=MUqNwgPjJvQ) — best practices from industry ML engineers
- ▶ [Cookiecutter Data Science — Project Templates](https://www.youtube.com/watch?v=2VuKIxzAyTE) — standard ML folder conventions explained

---
## 📂 Project File & Folder Reference — What Each File Does

> Think of the project as an assembly line. Data flows left to right through each component:

```
Raw Images  →  data_utils/  →  models/  →  training/  →  evaluation/  →  results/
                                   ↑
                             configs/  (controls everything)
```

---

### 📁 `configs/` — The Control Panel

No code here — only settings. Every other file reads from this folder. Changing a hyperparameter means editing one line here, not hunting through ten Python files.

| File | What it does | Who reads it | Output |
|---|---|---|---|
| `configs/train.yaml` | All 19 hyperparameters: lr, batch size, loss weights, image sizes, SR layer indices | trainer.py, loss.py, dataset.py — everything | None (read-only) |
| `configs/dataset.yaml` | Paths to data folders, class count (`nc: 1`), class names (`['lesion']`) | dataset.py, preprocess.py | None (read-only) |
| `configs/model.yaml` | Architecture flags — which backbone variant to use | sre_yolo.py | None (read-only) |

**Key values in `train.yaml` and why they matter:**

| Parameter | Value | Controls |
|---|---|---|
| `lr0` | 0.05 | Initial learning rate — how big each weight update step is |
| `epochs` | 100 | How many full passes through all training data |
| `batch_size` | 16 | How many images processed simultaneously on GPU |
| `c1` | 0.1 | SR loss weight — how much super-resolution influences training |
| `c2` | 7.5 | Bounding box loss weight — the primary detection objective |
| `c3` | 1.5 | DFL loss weight — box boundary distribution accuracy |
| `c4` | 0.5 | Classification loss weight — lesion vs background |
| `sr_layers` | [4, 8] | Which backbone layers feed the SR branch |
| `imgsz_lr` | 640 | Input image size to the detector |
| `imgsz_hr` | 1280 | Target image size for super-resolution reconstruction |

---

### 📁 `data/` — All Images and Split Lists

Never contains Python code — only image files, label files, and text files listing which images belong to each split.

**`data/raw/`**
Your original downloaded images, exactly as received. Never modified. `preprocess.py` reads from here.
- Output: Nothing new — just a safe backup of originals.

**`data/processed/lr/`**
All images resized to **640×640** — the exact size YOLOv8n expects. These are the actual images fed to the model every training step.
```
data/processed/lr/
    images/   ← 640×640 JPEG images
    labels/   ← YOLO .txt files (one per image, listing bounding boxes)
```

**`data/processed/hr/`**
Same images at **1280×1280** — the super-resolution reconstruction target. The SR branch tries to recreate these from the 640×640 input. The L1 loss compares SR branch output against these.
```
data/processed/hr/
    images/   ← 1280×1280 JPEG images
    # NO labels — HR is only used as an SR target, not for detection
```

**`data/splits/split1/` `split2/` `split3/`**
Each folder contains three plain text files. Every line is one absolute path to an LR image.
```
data/splits/split1/
    train.txt   ← ~3,100 lines
    val.txt     ← ~240 lines
    test.txt    ← ~135 lines
```
`dataset.py` reads these to know which images to load for each training/validation/test phase.

---

### 📁 `data_utils/` — Everything That Touches Data Before the Model

**`data_utils/preprocess.py`**
Three functions that run **once** before training to set up all data:

1. `filter_frames()` — scans `data/raw/`, removes blurry/dark/tiny images using Laplacian variance
2. `create_dual_resolution()` — produces both 640×640 LR and 1280×1280 HR versions of each image
3. `make_splits()` — groups images by patient, shuffles, writes train/val/test `.txt` files

Output:
```
results/filter_log.csv          ← log of every excluded image + reason
data/processed/lr/images/       ← LR images
data/processed/lr/labels/       ← YOLO label files
data/processed/hr/images/       ← HR images
data/splits/split{1,2,3}/*.txt  ← split file lists
```

**`data_utils/dataset.py`**
Defines `SREYOLODataset` — the class PyTorch calls thousands of times per epoch to load one sample. `__getitem__()` reads one image path from `train.txt`, loads the LR image, loads the matching HR image, loads bounding box labels, and returns them as GPU-ready tensors.

Output per `__getitem__()` call:
```python
{
  "lr_image": FloatTensor [3, 640, 640],    # fed to YOLOv8 backbone
  "hr_image": FloatTensor [3, 1280, 1280],  # compared against SR output
  "labels":   FloatTensor [N, 5],           # (class_id, cx, cy, w, h)
  "img_path": str                           # for debugging
}
```

**`data_utils/augment.py`**
Helper functions for random flips, scaling, and mosaic augmentation (combining 4 images into one). Called inside `dataset.py` when `augment=True`. Applied to **LR images and labels only** — never to HR images, which must stay clean as reconstruction targets. Output: modified tensors — not saved to disk.

---

### 📁 `models/` — The Neural Network Architecture

**`models/backbone.py`**
Wraps pretrained YOLOv8n and attaches PyTorch hooks at layers 4 and 8 to intercept feature maps mid-flow — like tapping a pipe at two points to sample what is flowing. The main detection flow continues unaffected.

Output per forward pass:
```python
(yolo_predictions,  {4: Tensor[B, 64, 80, 80],   # fine spatial detail → SR encoder
                     8: Tensor[B, 256, 20, 20]})  # semantic features  → SR encoder
```

**`models/sr_branch.py`**
The super-resolution decoder. Takes the two feature maps from the backbone and upscales them 4× into a full 1280×1280 image. Built from `ResBlock` (no BatchNorm, following EDSR) and `UpsamplePS` (pixel shuffle upsampling). **Only called during training — never at inference.**

Output:
```python
Tensor [B, 3, 1280, 1280]   # reconstructed HR image
                              # compared against data/processed/hr/ images
```

**`models/sre_yolo.py`**
The combined model — wires `BackboneWithHooks` and `SRBranch` together. The `inference` flag is the core innovation of the paper:

```python
# Training — both branches active
model(image, inference=False)
# → {"predictions": ..., "sr_output": Tensor[B, 3, 1280, 1280]}

# Inference — SR branch completely skipped
model(image, inference=True)
# → predictions only   (speed identical to vanilla YOLOv8n)
```

---

### 📁 `training/` — The Engine That Trains the Model

**`training/loss.py`**
Combines four loss terms into one number the optimiser minimises:

```
total_loss = 0.1·L_SR + 7.5·L_bbox + 1.5·L_dfl + 0.5·L_cls
```

Called once per batch. Its scalar output drives `loss.backward()` which computes gradients.

Output:
```python
{
  "total_loss": scalar tensor,  # the number that gets minimised
  "l_sr":       float,          # SR reconstruction quality
  "l_bbox":     float,          # bounding box accuracy
  "l_dfl":      float,          # box boundary distribution
  "l_cls":      float           # lesion classification
}
```

**`training/trainer.py`**
Orchestrates the full training loop:

```
for epoch in range(100):
    for batch in dataloader:          ← loads images from dataset.py
        predictions = model(batch)    ← calls sre_yolo.py
        loss = criterion(predictions) ← calls loss.py
        loss.backward()               ← computes gradients
        optimiser.step()              ← updates all weights
    validate()                        ← checks val AP after each epoch
    early_stop_check()                ← stops if no improvement for 50 epochs
    save_checkpoint()                 ← saves best model weights
```

Output:
```
weights/best_split1.pt    ← best model weights (main deliverable)
weights/last_split1.pt    ← most recent checkpoint for resuming
W&B dashboard             ← live loss curves and AP metrics
```

**`training/ablation.py`**
Runs a shortened 20-epoch training with one config override, then evaluates. Used only in Section 8 to systematically test every design choice in the paper.

Output:
```
results/ablations/sr_layers_48.json   ← {name, ap50, fps, gflops}
results/ablations/pretrain_coco.json
... (13 runs total)
results/ablation_summary.csv
```

---

### 📁 `evaluation/` — Measures How Good the Trained Model Is

**`evaluation/metrics.py`**
Four measurement functions — called after training is complete using saved checkpoint weights:

| Function | What it measures | Target value |
|---|---|---|
| `compute_ap50()` | AP@IoU=0.5 on full test set | 0.82 |
| `compute_ap50_by_size()` | AP for small / medium / large lesions separately | small: 0.80 |
| `measure_fps()` | Frames per second — times 500 forward passes | 58.8 FPS |
| `measure_gflops()` | Arithmetic operations per forward pass | 8.2 GFLOPs |

Output:
```python
{
  "ap50": 0.82,
  "ap50_small": 0.80, "ap50_medium": 0.82, "ap50_large": 0.85,
  "fps": 58.8,
  "gflops": 8.2
}
```

---

### 📄 Root-Level Files — Entry Points

**`train.py`**
CLI entry point for training. Reads flags, loads config, builds model and dataset, hands everything to `Trainer.fit()`.
```bash
python train.py --split 1 --weights coco.pt
```

**`evaluate.py`**
Loads a saved checkpoint and runs `metrics.py` on the test set.
```bash
python evaluate.py --weights weights/best_split1.pt --split 1
```
Output: printed metrics table + optional CSV file.

**`infer.py`**
Runs the model on an image, video file, or live webcam. Draws green bounding boxes on lesions with confidence scores and FPS counter overlay.
```bash
python infer.py --weights weights/best_split1.pt --source video.mp4
```
Output: annotated video/images saved to `results/inference/`

**`export.py`**
Converts trained PyTorch model to ONNX format for deployment on any hardware without needing PyTorch installed.
```bash
python export.py --weights weights/best_split1.pt
```
Output: `weights/sre_yolo_inference.onnx`

**`run_all_ablations.py`**
Runs all 13 ablation experiments automatically by calling `ablation.py` with different config overrides — one for each design choice tested in the paper.
Output: `results/ablation_summary.csv` — the data behind Tables 4–8 in the paper.

**`tests/`**
All pytest test files — one per section (`test_data.py`, `test_model.py`, `test_loss.py`, etc.). 51 tests total. These catch bugs before they waste hours of GPU training time. Run all with:
```bash
pytest tests/ -v
```

---

### How It All Flows Together

```
configs/train.yaml ──────────────────────────────────────────┐
                                                              ↓
data/raw/ → preprocess.py → data/processed/ → dataset.py → trainer.py
                                                    ↑              ↓
                                                loss.py       sre_yolo.py
                                                    ↑              ↑
                                             backbone.py + sr_branch.py
                                                              ↓
                                               weights/best_split1.pt
                                                              ↓
                                 evaluate.py → metrics.py → results/
                                 infer.py → results/inference/
                                 export.py → weights/sre_yolo_inference.onnx
```

code ran
```bash
mkdir -p /home/rl1231/sre_yolo/configs /home/rl1231/sre_yolo/data/raw /home/rl1231/sre_yolo/data/processed/lr /home/rl1231/sre_yolo/data/processed/hr /home/rl1231/sre_yolo/data/splits/split1/train /home/rl1231/sre_yolo/data/splits/split1/val /home/rl1231/sre_yolo/data/splits/split1/test /home/rl1231/sre_yolo/data/splits/split2/train /home/rl1231/sre_yolo/data/splits/split2/val /home/rl1231/sre_yolo/data/splits/split2/test /home/rl1231/sre_yolo/data/splits/split3/train /home/rl1231/sre_yolo/data/splits/split3/val /home/rl1231/sre_yolo/data/splits/split3/test /home/rl1231/sre_yolo/models /home/rl1231/sre_yolo/data_utils /home/rl1231/sre_yolo/training /home/rl1231/sre_yolo/evaluation /home/rl1231/sre_yolo/tests /home/rl1231/sre_yolo/weights /home/rl1231/sre_yolo/results
```

### YAML Configuration Files · 🟢 Beginner

YAML is a human-readable format for storing configuration. Instead of hardcoding hyperparameters in your Python files, you put them in a `.yaml` file. Changing a learning rate then means editing one line in one file, not hunting through code.

**🔗 In this project:** `configs/train.yaml` holds all 19 hyperparameters: learning rate, loss weights (c1=0.1, c2=7.5...), image sizes, SR layer indices, and more.

- ▶ [YAML Tutorial — Learn YAML in 10 Minutes](https://www.youtube.com/watch?v=BEki_rsWu4E) — syntax and use cases for YAML
- ▶ [Python yaml Library — Loading Configs](https://www.youtube.com/watch?v=YPrST4VKfUQ) — reading YAML files in Python scripts

---

### Git Version Control · 🟢 Beginner

Git tracks changes to your code over time. You can snapshot your work (commit), experiment on a copy (branch), and go back if something breaks. In ML, it also tracks which code version produced which experiment results.

**🔗 In this project:** The spec asks you to `git commit` after every section. This means you always have a working checkpoint to return to if a later section breaks something.

- ▶ [Git and GitHub for Beginners (freeCodeCamp)](https://www.youtube.com/watch?v=RGOj5yH7evk) — 1-hour complete beginner guide
- ▶ [Git for Machine Learning — Practical Guide](https://www.youtube.com/watch?v=VzAkMRXQoOM) — ML-specific git workflows

---

### argparse — CLI for Python Scripts · 🟢 Beginner

argparse lets you pass arguments to a Python script from the command line. Instead of editing code to change which dataset to use, you write: `python train.py --split 2 --weights coco.pt`.

**🔗 In this project:** `train.py` uses argparse so you can switch between splits, load pretrained weights, and toggle W&B logging all from the command line without touching code.

- ▶ [Python argparse Tutorial](https://www.youtube.com/watch?v=cdblJqEUDNo) — building command-line ML scripts

---


## 🗄️ Section 3 — Data Pipeline

> In ML, "garbage in, garbage out" is literal. The data pipeline section covers how raw endoscopy images are cleaned, resized, split, and served to the model. This is often 60% of a real project's work.

---

### Image Representation & OpenCV · 🟢 Beginner

A digital image is a 3D array of numbers: height × width × colour channels (RGB). OpenCV is the standard library for image loading, resizing, and manipulation in Python.

**🔗 In this project:** Every raw endoscopy frame is resized to 640×640 (model input) AND 1280×1280 (SR branch target). OpenCV handles both.

- ▶ [OpenCV Python Tutorial for Beginners (freeCodeCamp)](https://www.youtube.com/watch?v=oXlwWbU8l2o) — 4-hour complete OpenCV course
- ▶ [Image Processing Fundamentals](https://www.youtube.com/watch?v=gcNaQqZaTkI) — pixels, channels, and colour spaces explained

---

### YOLO Bounding Box Format · 🟢 Beginner

YOLO format stores boxes as: `class_id, cx, cy, w, h` — all normalised to [0,1] relative to image size. cx/cy is the box centre. This differs from Pascal VOC format (x_min, y_min, x_max, y_max).

**🔗 In this project:** All 3,892 frames use YOLO-format `.txt` label files. When you resize images from original size to 640×640, you must also scale the bounding box coordinates accordingly.

- ▶ [YOLO Object Detection Explained](https://www.youtube.com/watch?v=9s_FpMpdYW8) — what YOLO format means and why it works
- ▶ [Bounding Box Formats Explained](https://www.youtube.com/watch?v=VqMNODV7Pgk) — YOLO vs Pascal VOC vs COCO formats

---

### Image Quality Filtering (Laplacian Variance) · 🟡 Intermediate

The Laplacian operator computes second-order image derivatives — it responds strongly to edges. The variance of the Laplacian measures sharpness. A blurry image has low variance; a sharp image has high variance.

**🔗 In this project:** `filter_frames()` uses Laplacian variance < 100 to remove blurry endoscopy frames before training. Noisy labels from blurry frames would hurt model accuracy.

- ▶ [Blur Detection with OpenCV](https://www.youtube.com/watch?v=ukNqK5Qz0tA) — implementing Laplacian variance in Python
- ▶ [Image Quality Assessment Techniques](https://www.youtube.com/watch?v=k_fLKuBB6wM) — sharpness, brightness, and artefact detection

---

### Train / Validation / Test Split · 🟢 Beginner

You train on one portion of data, tune hyperparameters on validation, and report final performance on a held-out test set. The test set must NEVER influence training decisions.

**🔗 In this project:** `make_splits()` divides 3,892 frames into train/val/test. The split must be patient-level — all frames from one patient stay together — otherwise the model could memorise patient appearance and report fake high accuracy.

- ▶ [Train Test Split — Explained Visually](https://www.youtube.com/watch?v=_vdMKioCXqQ) — why you need 3 splits and how to do it
- ▶ [Data Leakage in Machine Learning](https://www.youtube.com/watch?v=jsdfsXAESQU) — the silent killer of ML project credibility

---

### PyTorch Dataset & DataLoader · 🟡 Intermediate

`torch.utils.data.Dataset` is an abstract class you subclass to teach PyTorch how to load one sample. `DataLoader` wraps a Dataset and handles batching, shuffling, and parallel loading.

**🔗 In this project:** `SREYOLODataset` returns a dict with `lr_image [3,640,640]`, `hr_image [3,1280,1280]`, and `labels` for every frame. The DataLoader then batches these for efficient GPU training.

- ▶ [PyTorch Dataset and DataLoader Explained](https://www.youtube.com/watch?v=PXOzkkB5eH0) — build a custom Dataset from scratch
- ▶ [Custom DataLoader Tutorial](https://www.youtube.com/watch?v=ZoZHd0Zm3RY) — handles images, labels, and augmentation

---

### Data Augmentation · 🟡 Intermediate

Augmentation artificially expands your dataset by applying random transformations to existing images. This prevents overfitting — the model can't memorise the training set if it looks slightly different every epoch.

**🔗 In this project:** `SREYOLODataset` applies random flips, scaling, and Mosaic (combining 4 images) to LR images during training. Augmentation is NEVER applied to HR images.

- ▶ [Data Augmentation for Object Detection](https://www.youtube.com/watch?v=GBCzM8VY7k8) — flip, scale, mosaic — and why they work
- ▶ [Albumentations Library Tutorial](https://www.youtube.com/watch?v=rAdLwKJBvPM) — fastest augmentation library for PyTorch
- ▶ [Mosaic Augmentation (YOLOv4 technique)](https://www.youtube.com/watch?v=v_N2zGVR7lw) — how 4-image mosaic improves small object detection

---

## 🧠 Section 4 — Model Architecture

> This is the heart of the paper. Three interlocking pieces: the YOLOv8 detector, an EDSR-inspired super-resolution branch, and the mechanism that fuses them during training but removes the SR branch at inference.

---

### Convolutional Neural Networks (CNNs) · 🟢 Beginner

CNNs apply small learnable filters (kernels) across an image, detecting features like edges, textures, and shapes. Deeper layers combine these into complex representations.

**🔗 In this project:** Both the YOLOv8n backbone and the SR branch are built from Conv2d layers. Understanding `kernel_size`, `stride`, and `padding` is essential for reading the architecture spec.

- ▶ [But what is a Convolutional Neural Network? (3Blue1Brown)](https://www.youtube.com/watch?v=KuXjwB4LzSA) — best visual intuition for CNNs — 20 min
- ▶ [CNN Explained — Stanford CS231n](https://www.youtube.com/watch?v=bNb2fEVKeEo) — rigorous explanation of conv, pooling, strides
- ▶ [nn.Conv2d Parameters Explained](https://www.youtube.com/watch?v=y2BaTt1fxJU) — kernel, stride, padding — practical PyTorch

---

### Object Detection & YOLO · 🟡 Intermediate

Object detection predicts both WHAT is in an image and WHERE (bounding box). YOLO does this in a single forward pass. YOLOv8 uses a CSP backbone, PAN neck, and decoupled head.

**🔗 In this project:** SRE-YOLO is built on YOLOv8n (nano). You hook into its backbone layers 4 and 8 to extract feature maps for the SR branch.

- ▶ [YOLO Object Detection Explained (Computerphile)](https://www.youtube.com/watch?v=ag3DLKsl2vk) — conceptual walkthrough
- ▶ [YOLOv8 Architecture Deep Dive](https://www.youtube.com/watch?v=wdN1_TL_u5Q) — backbone, neck, and head explained
- ▶ [YOLOv8 with Ultralytics — Full Tutorial](https://www.youtube.com/watch?v=m9fH9OWn8YM) — practical usage of the Ultralytics library

---

### Feature Maps & Backbone Hooks · 🟡 Intermediate

As an image passes through a CNN, each layer produces a feature map — a tensor encoding what the network has detected at that depth. PyTorch hooks intercept these intermediate tensors without modifying the model.

**🔗 In this project:** Hooks are registered on layers 4 (64 channels, fine spatial detail) and 8 (256 channels, semantic features). The ablation study shows layers 4+8 outperform layers 2+6 by +2% AP.

- ▶ [PyTorch Forward Hooks Tutorial](https://www.youtube.com/watch?v=UOvPeC8WOt8) — registering hooks to capture intermediate tensors
- ▶ [Feature Visualization in Neural Networks](https://www.youtube.com/watch?v=ghEmQSxT6tw) — what feature maps actually look like

---

### Residual Networks & ResBlocks · 🟡 Intermediate

A residual connection adds the input of a block directly to its output: `output = F(x) + x`. This solves the vanishing gradient problem and enables very deep networks to train effectively.

**🔗 In this project:** The SR decoder uses 16 residual blocks. BatchNorm is removed from each block (following EDSR) to preserve the full pixel-value range for super-resolution quality.

- ▶ [Residual Networks Explained (ResNet)](https://www.youtube.com/watch?v=GWt6Fu05voI) — skip connections and why they matter — 15 min
- ▶ [ResNet Paper Walkthrough](https://www.youtube.com/watch?v=sAl7W4_kFoA) — deep dive into He et al. 2015 paper

---

### Super-Resolution & EDSR · 🔵 Advanced

Super-resolution reconstructs a high-resolution image from a low-resolution input. EDSR (Lim et al. 2017) removed BatchNorm from residual blocks — preserving absolute pixel range information that SR critically depends on.

**🔗 In this project:** The SR branch forces the shared backbone features to encode fine spatial detail during training, which benefits the detector for small lesions. At inference, the branch is removed completely.

- ▶ [Super Resolution Deep Learning Explained](https://www.youtube.com/watch?v=KULkSwLk62I) — SR concepts, loss functions, and architectures
- ▶ [EDSR Paper Explained](https://www.youtube.com/watch?v=yp7c5sMYEUI) — why removing BatchNorm improves SR
- ▶ [Pixel Shuffle Upsampling Explained](https://www.youtube.com/watch?v=Vk-eWA-4pao) — sub-pixel convolution for fast upscaling

---

### Pixel Shuffle (Sub-pixel Convolution) · 🔵 Advanced

Pixel shuffle rearranges a tensor `[B, C×r², H, W]` into `[B, C, H×r, W×r]` — upscaling spatial dimensions by factor r without checkerboard artefacts.

**🔗 In this project:** Two `UpsamplePS` blocks (each with r=2) achieve a total 4× spatial upscale: from `[B, C, 320, 320]` → `[B, 3, 1280, 1280]`.

- ▶ [Pixel Shuffle / Sub-pixel Convolution Explained](https://www.youtube.com/watch?v=Vk-eWA-4pao) — visual walkthrough of how pixel shuffle upscales
- ▶ [nn.PixelShuffle in PyTorch](https://www.youtube.com/watch?v=5V6sSKMSAko) — implementing pixel shuffle from scratch

---

## 📉 Section 5 — Loss Function

> The loss function is the signal that drives learning. SRE-YOLO uses a carefully weighted combination of four loss terms.

---

### Loss Functions & Gradient Descent · 🟢 Beginner

A loss function measures how wrong the model's predictions are. Gradient descent uses backpropagation to find the direction that reduces the loss, then updates weights by a small step.

**🔗 In this project:** The combined SRE-YOLO loss has four terms. Gradients flow back through both the detection head AND the SR branch, so both benefit from every training step.

- ▶ [Loss Functions Explained](https://www.youtube.com/watch?v=Skc8nqJirJg) — MSE, L1, cross-entropy — when to use which
- ▶ [Gradient Descent, Step by Step (StatQuest)](https://www.youtube.com/watch?v=sDv4f4s2SB8) — clear visual explanation — highly recommended
- ▶ [Backpropagation Explained (3Blue1Brown)](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — the chain rule made visual — essential viewing

---

### L1 Loss (Mean Absolute Error) · 🟢 Beginner

L1 loss sums absolute differences: `L = mean(|y_pred - y_true|)`. More robust to outliers than L2. In image reconstruction, L1 often produces sharper results than L2.

**🔗 In this project:** `L_SR = L1(SR_output, HR_ground_truth)`. Minimising this forces the network to reconstruct fine details accurately.

- ▶ [L1 vs L2 Loss — When to Use Each](https://www.youtube.com/watch?v=65o6GDFUMvM) — MAE vs MSE tradeoffs for regression and SR

---

### IoU Loss & Bounding Box Regression · 🟡 Intermediate

IoU (Intersection over Union) measures overlap between predicted and ground-truth boxes. A perfect prediction has IoU=1. YOLOv8 uses CIoU which also penalises centre distance and aspect ratio mismatch.

**🔗 In this project:** `L_bbox` has coefficient c2=7.5 — the highest of the four, reflecting that accurate box localisation is the primary detection goal.

- ▶ [IoU Explained for Object Detection](https://www.youtube.com/watch?v=XXYG5ZWtjj0) — IoU, GIoU, DIoU, CIoU — visual walkthrough
- ▶ [Bounding Box Regression Loss Functions](https://www.youtube.com/watch?v=7LV0tBl0bYw) — how YOLO boxes are trained

---

### Distribution Focal Loss (DFL) · 🔵 Advanced

Instead of predicting a single coordinate value, the model predicts a probability distribution over possible values. DFL is the cross-entropy loss between this distribution and a one-hot target. It leads to more precise box edges.

**🔗 In this project:** `L_dfl` (c3=1.5) is specific to YOLOv8's decoupled detection head, which predicts box boundaries as distributions.

- ▶ [DFL — Distribution Focal Loss Explained](https://www.youtube.com/watch?v=l9OKOW7Kkng) — how YOLOv8 predicts box edges as distributions

---

### Weighted Multi-Task Loss · 🟡 Intermediate

When training on multiple objectives simultaneously, you combine their losses with scalar weights. The weights control how much each objective influences shared feature representations.

**🔗 In this project:** `total = 0.1·L_SR + 7.5·L_bbox + 1.5·L_dfl + 0.5·L_cls`. The small SR weight (0.1) prevents the SR branch from dominating training and harming detection accuracy.

- ▶ [Multi-Task Learning Loss Weighting](https://www.youtube.com/watch?v=qdRqjJiQHhg) — how to balance competing loss terms

---

## 🏋️ Section 6 — Training Pipeline

> The training pipeline orchestrates everything: optimiser, learning rate schedule, early stopping, checkpointing, and experiment logging.

---

### Optimisers: AdamW · 🟡 Intermediate

AdamW maintains a per-parameter learning rate based on gradient moments — much faster than SGD. The W suffix adds weight decay correctly, separate from the gradient update.

**🔗 In this project:** Trainer uses AdamW with `lr0=0.05` and `weight_decay=1e-4`.

- ▶ [Adam Optimizer Explained — StatQuest](https://www.youtube.com/watch?v=JXQT_vxqwIs) — intuitive explanation of adaptive learning rates
- ▶ [AdamW vs Adam — What's the Difference?](https://www.youtube.com/watch?v=0WRTelebsS4) — why weight decay matters

---

### Learning Rate Scheduling · 🟡 Intermediate

Cosine Annealing smoothly decreases the LR following a cosine curve from `lr_max` to `lr_min`, allowing fine-grained convergence.

**🔗 In this project:** `CosineAnnealingLR(T_max=100, eta_min=0.01)` decreases LR from 0.05 to 0.01 over 100 epochs.

- ▶ [Learning Rate Schedules Explained](https://www.youtube.com/watch?v=QzulmoOg2JE) — step, cosine, warmup — visual comparison
- ▶ [Cosine Annealing in PyTorch](https://www.youtube.com/watch?v=SKYMzNm7UoM) — implementing and visualising cosine scheduling

---

### Transfer Learning & Fine-tuning · 🟡 Intermediate

Starting from a model pretrained on a large dataset (COCO: 118K images) and fine-tuning on the target dataset dramatically reduces the data needed for good results.

**🔗 In this project:** Without COCO pretraining, the paper shows AP drops from 0.77 to 0.71 (Table 4) — a 6-point penalty for skipping this step.

- ▶ [Transfer Learning Explained Visually](https://www.youtube.com/watch?v=yofjFQddwHE) — why pretrained features transfer to new tasks
- ▶ [Fine-tuning YOLO on Custom Data](https://www.youtube.com/watch?v=0inNp1M8OBw) — practical COCO → custom fine-tune walkthrough

---

### Overfitting & Early Stopping · 🟢 Beginner

Overfitting is when a model memorises training data instead of learning generalisable patterns. Early stopping halts training when validation performance stops improving.

**🔗 In this project:** `patience=50` means training stops if validation AP does not improve for 50 consecutive epochs. With only 3,452 frames, overfitting is a real risk.

- ▶ [Overfitting and Underfitting Explained](https://www.youtube.com/watch?v=EuBBz3bI-aA) — the bias-variance tradeoff with visuals
- ▶ [Early Stopping in Neural Networks](https://www.youtube.com/watch?v=NnS0FJyVcDQ) — implementing and tuning patience

---

### Experiment Tracking with Weights & Biases · 🟡 Intermediate

W&B logs metrics, hyperparameters, and model outputs to an online dashboard. You can compare multiple runs side-by-side and reproduce any experiment.

**🔗 In this project:** `wandb.log()` is called every batch (loss) and every epoch (AP). The dashboard lets you watch the SR loss and detection AP co-evolve during training.

- ▶ [Weights & Biases (W&B) Full Tutorial](https://www.youtube.com/watch?v=G7GH0SeNBMA) — setup, logging, and comparing runs

---

## 📊 Section 7 — Evaluation & Metrics

---

### Mean Average Precision (AP@IoU=0.5) · 🟡 Intermediate

AP computes the area under the Precision-Recall curve at a given IoU threshold. AP@0.5 means a detection counts as correct if its IoU with the ground-truth box is ≥ 0.5.

**🔗 In this project:** The paper's main result is SRE-YOLO achieves AP@0.5 = 0.82 vs baseline 0.77 (+5%).

- ▶ [Mean Average Precision (mAP) Explained](https://www.youtube.com/watch?v=FppOzcDvaDI) — the most thorough AP explanation on YouTube
- ▶ [Precision, Recall, and F1 Score](https://www.youtube.com/watch?v=jJ7ff7Gcubg) — foundations needed before understanding mAP

---

### GFLOPs — Computational Complexity · 🟡 Intermediate

GFLOPs measures arithmetic operations per forward pass. Hardware-independent, unlike FPS.

**🔗 In this project:** SRE-YOLO has identical GFLOPs (8.2) to baseline at inference because the SR branch is completely removed. +5% AP at zero inference cost.

- ▶ [FLOPs and Model Complexity Explained](https://www.youtube.com/watch?v=RxOFNJQ2WLs) — GFLOPs, parameters, and inference speed

---

### Small Object Detection · 🔵 Advanced

Small objects (< 32×32 pixels) are notoriously hard to detect — they occupy few pixels and are disproportionately affected by feature map downsampling in deep networks.

**🔗 In this project:** SRE-YOLO improves AP for small lesions from 0.66 → 0.80 (+21%) on ENDO-LC ext. This is the most clinically significant result in the paper.

- ▶ [Small Object Detection — Challenges and Solutions](https://www.youtube.com/watch?v=r9IQDQkSv_w) — why small objects are hard and how SR helps

---

## 🔬 Section 8 — Ablation Studies

---

### Ablation Studies — What and Why · 🟢 Beginner

Ablation studies isolate each variable: change only X, keep everything else the same, measure the difference. This is the scientific method applied to neural network design.

**🔗 In this project:** 5 ablation sets test pre-training dataset, SR layer placement, SR architecture, and backbone size — justifying every design choice in SRE-YOLO.

- ▶ [How to Read ML Research Papers](https://www.youtube.com/watch?v=733m6qBH-jI) — understanding tables, ablations, and results sections
- ▶ [Ablation Studies in Deep Learning Research](https://www.youtube.com/watch?v=K0_GdBz3Y00) — why ablations are the core of experimental ML papers

---

## 🚀 Section 9 — Inference & Deployment

---

### Inference vs Training Mode · 🟢 Beginner

During inference, `model.eval()` and `torch.no_grad()` disable gradient tracking, saving 2–3× memory and compute. Dropout and BatchNorm also behave differently in eval mode.

**🔗 In this project:** `SREYOLO.forward(x, inference=True)` completely skips the SR branch — the core innovation. The SR branch is a training-time regulariser, not an inference-time cost.

- ▶ [PyTorch model.eval() vs model.train()](https://www.youtube.com/watch?v=GtPnWjnC90A) — dropout, batchnorm, and gradient modes explained

---

### ONNX — Model Export & Portability · 🟡 Intermediate

ONNX is a universal format for representing neural networks, decoupling your model from PyTorch. It can run on ONNX Runtime, TensorRT, CoreML, OpenVINO — any hardware.

**🔗 In this project:** `export.py` converts inference-mode SRE-YOLO (SR branch removed) to ONNX. Tests verify outputs match PyTorch to within 1e-4.

- ▶ [ONNX Export with PyTorch — Full Tutorial](https://www.youtube.com/watch?v=7nutT3Aacyw) — torch.onnx.export step by step
- ▶ [ONNX Runtime for Fast Inference](https://www.youtube.com/watch?v=UUL5rHBHUA4) — running ONNX models in production

---

### Non-Maximum Suppression (NMS) · 🟡 Intermediate

NMS removes duplicate overlapping bounding boxes: keep the highest-confidence box, suppress all boxes that overlap it by more than an IoU threshold.

**🔗 In this project:** `infer.py` applies NMS with `conf_thresh=0.25` and `iou_thresh=0.45`.

- ▶ [Non-Maximum Suppression Explained](https://www.youtube.com/watch?v=VAo84c1hQX8) — visual walkthrough of how NMS works

---

## 🧪 Section 10 — Integration Testing

---

### pytest — Python Testing Framework · 🟢 Beginner

pytest is the standard Python testing library. Functions starting with `test_` are automatically discovered and run. Fixtures create reusable test data.

**🔗 In this project:** Every section ends with pytest commands. 51 test cases across 10 sections verify the full pipeline end-to-end.

- ▶ [pytest Tutorial for Beginners (Corey Schafer)](https://www.youtube.com/watch?v=cHYq1MRoyI0) — setup, fixtures, parametrize — complete guide

---

### Synthetic Data for Testing · 🟡 Intermediate

Tests using synthetic data (random noise images with random labels) run in milliseconds, need no external files, and test the exact same code paths as real data.

**🔗 In this project:** All `tests/test_*.py` files create synthetic 640×640 images in a temp directory. pytest runs completely self-contained.

- ▶ [Generating Synthetic Test Data in Python](https://www.youtube.com/watch?v=VlFxoFZs_-Q) — numpy-based synthetic image and label generation

---

## 📚 Recommended Learning Path

| When | What to Watch / Read | Covers |
|---|---|---|
| Before §1–2 | 3Blue1Brown: Neural Networks series (4 videos) | Intuition for how neural nets learn |
| Before §3 | PyTorch Beginner Series (official docs) | Tensors, autograd, nn.Module |
| Before §3 | OpenCV Python Tutorial (freeCodeCamp, 4hr) | Image loading, resizing, drawing |
| Before §4 | Stanford CS231n Lecture 5 — CNNs | Convolution, pooling, strides |
| Before §4 | YOLO Object Detection Explained | YOLO architecture and format |
| Before §5 | StatQuest: Gradient Descent | Backprop and loss functions |
| Before §6 | W&B Tutorial — Experiment Tracking | W&B setup and logging |
| Before §7 | Mean Average Precision Explained | mAP, AP@0.5, PR curves |
| Before §9 | ONNX Export with PyTorch | Model export and portability |
| Before §10 | pytest Tutorial (Corey Schafer) | Writing and running tests |

> 💡 **Tip:** You will learn far more by building and breaking things than by watching videos. Start coding early, use the YouTube resources when you get stuck, and trust the tests to tell you when something is right.
