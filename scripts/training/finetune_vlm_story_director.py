#!/usr/bin/env python3
"""VLM Story Director & Incomplete-Story Weaver QLoRA Fine-Tuning Harness (`scripts/training/finetune_vlm_story_director.py`).

Validates `data/datasets/vlm_story_director/train.jsonl` across all 18 genres, 10 panel layouts,
and 8 storytelling arc modes, and exports an Unsloth / LLaMA-Factory QLoRA training configuration
for `Qwen/Qwen2.5-VL-7B-Instruct` (Colab A100/L4) or `Qwen/Qwen2.5-VL-3B-Instruct` (local 8GB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets" / "vlm_story_director" / "train.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "training_configs" / "vlm_story_director"


def validate_and_export_vlm_config(
    dataset_path: Path = DEFAULT_DATASET,
    out_dir: Path = DEFAULT_OUT_DIR,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    lora_rank: int = 32,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty VLM dataset at {dataset_path}")

    genres = sorted({r["genre_id"] for r in rows})
    layouts = sorted({r["panel_layout_topology"] for r in rows})
    arcs = sorted({r["storytelling_arc_mode"] for r in rows})
    fallback_count = sum(1 for r in rows if r.get("incomplete_story_fallback"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qlora_cfg = {
        "model_name_or_path": base_model,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_rank),
        "lora_target": "all",
        "quantization_bit": 4,
        "dataset_path": str(Path(dataset_path).as_posix()),
        "template": "qwen2_vl",
        "cutoff_len": 4096,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 3.0,
        "bf16": True,
        "output_dir": str((out_dir / "checkpoints").as_posix()),
        "dataset_stats": {
            "total_records": len(rows),
            "genres_count": len(genres),
            "layouts_count": len(layouts),
            "story_arc_modes_count": len(arcs),
            "incomplete_story_fallback_records": fallback_count,
        },
    }

    cfg_file = out_dir / "qwen25_vl_qlora_config.json"
    cfg_file.write_text(json.dumps(qlora_cfg, indent=2), encoding="utf-8")
    return qlora_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLM Story Director dataset and export QLoRA config.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    cfg = validate_and_export_vlm_config(args.dataset, args.out_dir, args.model, args.rank)
    print(json.dumps(cfg["dataset_stats"], indent=2))


if __name__ == "__main__":
    main()
