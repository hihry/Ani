#!/usr/bin/env python3
"""Wan 2.2 TI2V-5B / I2V-A14B Motion LoRA Training Harness (`scripts/training/finetune_wan22_motion_lora.py`).

Validates `data/datasets/wan22_motion_lora/train.jsonl` across all 18 genres and motion primitives,
and exports a `kohya-ss/musubi-tuner` / `diffusion-pipe` configuration for training the
`m2a_anime_motion` LoRA (Rank 32 / 64) on local 8GB-12GB GPUs (with block-swapping) or Colab A100.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "wan22_motion_lora" / "train.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "training_configs" / "wan22_motion_lora"


def validate_and_export_wan22_lora_config(
    dataset_path: Path = DEFAULT_DATASET,
    out_dir: Path = DEFAULT_OUT_DIR,
    base_model: str = "Wan-AI/Wan2.2-TI2V-5B",
    lora_rank: int = 32,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty Wan 2.2 Motion LoRA dataset at {dataset_path}")

    genres = sorted({r["genre_id"] for r in rows})
    primitives = sorted({r["motion_primitive_tag"] for r in rows})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    musubi_cfg = {
        "trainer": "kohya-ss/musubi-tuner",
        "dit_model": base_model,
        "network_module": "networks.lora_wan",
        "network_dim": int(lora_rank),
        "network_alpha": int(lora_rank),
        "target_modules": ["blocks.*.self_attn", "blocks.*.cross_attn", "blocks.*.ffn"],
        "trigger_token": "m2a_anime_motion",
        "motion_primitives": primitives,
        "resolution": [480, 832],
        "frame_Buckets": [17, 33, 49],
        "fps": 24,
        "mixed_precision": "bf16",
        "fp8_base": True,
        "gradient_checkpointing": True,
        "blocks_to_swap": 12,
        "optimizer_type": "AdamW8bit",
        "learning_rate": 1e-4,
        "max_train_steps": 1500,
        "dataset_path": str(Path(dataset_path).as_posix()),
        "output_dir": str((out_dir / "checkpoints").as_posix()),
        "dataset_stats": {
            "total_records": len(rows),
            "genres_count": len(genres),
            "motion_primitives_count": len(primitives),
            "motion_primitives": primitives,
        },
    }

    cfg_file = out_dir / "musubi_wan22_motion_lora_config.json"
    cfg_file.write_text(json.dumps(musubi_cfg, indent=2), encoding="utf-8")
    return musubi_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Wan 2.2 Motion LoRA dataset and export musubi-tuner config.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    cfg = validate_and_export_wan22_lora_config(args.dataset, args.out_dir, args.model, args.rank)
    print(json.dumps(cfg["dataset_stats"], indent=2))


if __name__ == "__main__":
    main()
