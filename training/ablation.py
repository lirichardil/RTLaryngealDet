"""Ablation study runner for SRE-YOLO."""
from pathlib import Path
from typing import List, Dict

import yaml


ABLATION_GRID: List[Dict] = [
    # Baseline: no SR branch
    {"sr_enabled": False, "sr_decoder": None, "sr_resblocks": None, "tag": "baseline_no_sr"},
    # SR shallow decoder
    {"sr_enabled": True, "sr_decoder": "shallow", "sr_resblocks": 8,  "tag": "sr_shallow_r8"},
    # SR deep decoder — 8 resblocks
    {"sr_enabled": True, "sr_decoder": "deep",    "sr_resblocks": 8,  "tag": "sr_deep_r8"},
    # SR deep decoder — 16 resblocks (default)
    {"sr_enabled": True, "sr_decoder": "deep",    "sr_resblocks": 16, "tag": "sr_deep_r16"},
]


def run_ablation(base_cfg_path: Path, results_dir: Path = Path("results")) -> None:
    """Run all ablation variants sequentially.

    Args:
        base_cfg_path: Path to the base model.yaml to merge overrides into.
        results_dir: Directory where per-variant results are stored.
    """
    base_cfg = yaml.safe_load(base_cfg_path.read_text())

    for variant in ABLATION_GRID:
        tag = variant.pop("tag")
        cfg = {**base_cfg, **variant}
        out_dir = results_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist the merged config for reproducibility
        (out_dir / "model.yaml").write_text(yaml.dump(cfg))

        print(f"\n{'='*60}")
        print(f"Running ablation: {tag}")
        print(f"  Config: {cfg}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        # Import here to avoid circular imports at module level
        from models import SREYolo
        from training.trainer import Trainer

        model = SREYolo(cfg)
        # Trainer.fit() call omitted — integrate with real DataLoaders in run_all_ablations.py
        print(f"[{tag}] Model instantiated. Attach DataLoaders to start training.")
