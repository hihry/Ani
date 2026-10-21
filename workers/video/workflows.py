"""Tagged ComfyUI API workflow loader and parameter injector (`workers/video/workflows.py`)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

REQUIRED_TAGS = {
    "M2A_IMAGE",
    "M2A_POS",
    "M2A_NEG",
    "M2A_SEED",
    "M2A_LATENT",
    "M2A_SAMPLER",
    "M2A_OUTPUT",
}


def load_template(path: Path) -> Dict[str, Any]:
    path = Path(path)
    wf = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wf, dict):
        raise ValueError(f"{path}: ComfyUI API workflow must be a JSON object keyed by node ID")
    found = {
        n.get("_meta", {}).get("title")
        for n in wf.values()
        if isinstance(n, dict)
    }
    missing = REQUIRED_TAGS - found
    if missing:
        raise ValueError(f"{path}: workflow missing required _meta.title tags: {sorted(missing)}")
    return wf


def inject(
    template: Dict[str, Any],
    *,
    image_name: str,
    pos_prompt: str,
    neg_prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    steps: int,
    cfg: float,
    fps: int,
    filename_prefix: str = "m2a_clip",
) -> Dict[str, Any]:
    wf = copy.deepcopy(template)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        tag = node.get("_meta", {}).get("title")
        inp = node.setdefault("inputs", {})
        if tag == "M2A_IMAGE":
            inp["image"] = image_name
        elif tag == "M2A_POS":
            inp["text"] = pos_prompt
        elif tag == "M2A_NEG":
            inp["text"] = neg_prompt
        elif tag == "M2A_SEED":
            key = "noise_seed" if "noise_seed" in inp else "seed"
            inp[key] = int(seed)
        elif tag == "M2A_LATENT":
            inp["width"] = int(width)
            inp["height"] = int(height)
            length_key = "length" if "length" in inp else "num_frames"
            inp[length_key] = int(frames)
        elif tag == "M2A_SAMPLER":
            inp["steps"] = int(steps)
            if "cfg" in inp:
                inp["cfg"] = float(cfg)
            if "seed" in inp:
                inp["seed"] = int(seed)
            if "noise_seed" in inp:
                inp["noise_seed"] = int(seed)
        elif tag == "M2A_OUTPUT":
            if "fps" in inp:
                inp["fps"] = float(fps)
            if "frame_rate" in inp:
                inp["frame_rate"] = float(fps)
            if "filename_prefix" in inp:
                inp["filename_prefix"] = filename_prefix
    return wf
