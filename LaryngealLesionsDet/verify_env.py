"""
verify_env.py
Imports every package required by the sre_yolo environment and prints its version.
Run with: python verify_env.py
Exit code 0 means all packages are present and correctly versioned.
"""

import sys

errors = []


def check(pkg_name: str, import_name: str, version_attr: str = "__version__") -> None:
    """Import a package and print its version; collect any failures."""
    try:
        mod = __import__(import_name)
        version = getattr(mod, version_attr, "unknown")
        print(f"  OK  {pkg_name:<30s} {version}")
    except Exception as exc:
        errors.append(f"FAIL  {pkg_name:<30s} {exc}")
        print(f"FAIL  {pkg_name:<30s} {exc}")


print(f"\nPython  {sys.version}\n")
print("=" * 60)

# ── Core numeric / data ──────────────────────────────────────────
check("numpy",           "numpy")
check("pandas",          "pandas")

# ── PyTorch stack ────────────────────────────────────────────────
check("torch",           "torch")
check("torchvision",     "torchvision")

# Confirm CUDA is visible to PyTorch
import torch
cuda_available = torch.cuda.is_available()
print(f"  {'OK' if cuda_available else 'WARN'}"
      f"  {'torch.cuda.is_available':<30s} {cuda_available}")

# ── Vision / augmentation ────────────────────────────────────────
check("opencv-python",   "cv2",           "__version__")
check("albumentations",  "albumentations")

# ── YOLO / detection ─────────────────────────────────────────────
check("ultralytics",     "ultralytics")

# ── COCO tools / metrics ─────────────────────────────────────────
check("pycocotools",     "pycocotools",   "__version__")

# ── Model complexity ─────────────────────────────────────────────
check("thop",            "thop")

# ── Experiment tracking ──────────────────────────────────────────
check("wandb",           "wandb")

# ── Visualisation ────────────────────────────────────────────────
check("matplotlib",      "matplotlib")
check("tqdm",            "tqdm")

# ── Serialisation / config ───────────────────────────────────────
check("pyyaml",          "yaml")

# ── Export / inference ───────────────────────────────────────────
check("onnx",            "onnx")
check("onnxruntime",     "onnxruntime",   "__version__")

# ── Testing ──────────────────────────────────────────────────────
check("pytest",          "pytest")

print("=" * 60)

if errors:
    print(f"\n{len(errors)} package(s) failed to import:\n")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print(f"\nAll packages imported successfully. Environment is ready.\n")
    sys.exit(0)
