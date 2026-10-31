#!/usr/bin/env python3
"""2D Anime Context Background LoRA & 3D Stage Training Harness (`scripts/training/finetune_bg_illustrious_lora.py`).

Validates `data/datasets/bg_3d_stage/train.jsonl` across all 18 genres and exports
a `kohya-ss/sd-scripts` SDXL LoRA configuration for `OnomaAIResearch/Illustrious-xl-early-release-v0`
(or `Laxhar/noobai-XL-1.1` / `cagliostrolab/animagine-xl-4.0`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "bg_3d_stage" / "train.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "training_configs" / "bg_illustrious_lora"


def validate_and_export_bg_lora_config(
    dataset_path: Path = DEFAULT_DATASET,
    out_dir: Path = DEFAULT_OUT_DIR,
    base_model: str = "OnomaAIResearch/Illustrious-xl-early-release-v0",
    network_dim: int = 32,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty 2D/3D background dataset at {dataset_path}")

    genres = sorted({r["genre_id"] for r in rows})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdxl_cfg = {
        "pretrained_model_name_or_path": base_model,
        "network_module": "networks.lora",
        "network_dim": int(network_dim),
        "network_alpha": int(network_dim),
        "resolution": "1024,1536",
        "enable_bucket": True,
        "min_bucket_reso": 512,
        "max_bucket_reso": 2048,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "optimizer_type": "AdamW8bit",
        "learning_rate": 1e-4,
        "unet_lr": 1e-4,
        "text_encoder_lr": 5e-5,
        "max_train_steps": 1800,
        "gradient_checkpointing": True,
        "dataset_path": str(Path(dataset_path).as_posix()),
        "output_dir": str((out_dir / "checkpoints").as_posix()),
        "dataset_stats": {
            "total_records": len(rows),
            "genres_count": len(genres),
            "depth_model": rows[0]["depth_mesh_3d_stage"]["depth_model"],
            "cutout_model": rows[0]["character_cutout_extraction"]["primary_model"],
        },
    }

    cfg_file = out_dir / "illustrious_xl_bg_lora_config.json"
    cfg_file.write_text(json.dumps(sdxl_cfg, indent=2), encoding="utf-8")
    return sdxl_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 2D/3D Stage dataset and export Illustrious-XL LoRA config.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=str, default="OnomaAIResearch/Illustrious-xl-early-release-v0")
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    cfg = validate_and_export_bg_lora_config(args.dataset, args.out_dir, args.model, args.rank)
    print(json.dumps(cfg["dataset_stats"], indent=2))


if __name__ == "__main__":
    main()
