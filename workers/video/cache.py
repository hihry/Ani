"""Content-addressed clip cache (`workers/video/cache.py`).

Caches conformed clips under `runs/_cache/clips/<sha256>.avi` + `<sha256>.json`
keyed by `(model_id, workflow_sha256, image_sha256, pos_prompt, neg_prompt, seed, width, height, frames, steps, cfg)`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_cache_key(
    *,
    model_id: str,
    workflow_template: Dict[str, Any],
    image_bytes: bytes,
    pos_prompt: str,
    neg_prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    steps: int,
    cfg: float,
) -> str:
    wf_canonical = json.dumps(workflow_template, sort_keys=True, separators=(",", ":")).encode("utf-8")
    wf_sha = hashlib.sha256(wf_canonical).hexdigest()
    img_sha = hashlib.sha256(image_bytes).hexdigest()
    payload = {
        "model_id": model_id,
        "workflow_sha256": wf_sha,
        "image_sha256": img_sha,
        "pos_prompt": pos_prompt,
        "neg_prompt": neg_prompt,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "frames": int(frames),
        "steps": int(steps),
        "cfg": round(float(cfg), 4),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ClipCache:
    def __init__(self, cache_dir: Path = Path("runs/_cache/clips")) -> None:
        self.cache_dir = Path(cache_dir)

    def get(self, key: str, dest_path: Path) -> Optional[Dict[str, Any]]:
        if not self.cache_dir.is_dir():
            return None
        meta_path = self.cache_dir / f"{key}.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ext = meta.get("ext", ".avi")
            cached_clip = self.cache_dir / f"{key}{ext}"
            if not cached_clip.is_file():
                return None
            dest_path = Path(dest_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_clip, dest_path)
            return meta
        except Exception:
            return None

    def put(self, key: str, clip_path: Path, meta: Dict[str, Any]) -> Tuple[Path, Path]:
        clip_path = Path(clip_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ext = clip_path.suffix or ".avi"
        cached_clip = self.cache_dir / f"{key}{ext}"
        meta_path = self.cache_dir / f"{key}.json"
        shutil.copy2(clip_path, cached_clip)
        meta_to_save = dict(meta)
        meta_to_save["ext"] = ext
        meta_to_save["cache_key"] = key
        meta_path.write_text(json.dumps(meta_to_save, indent=2), encoding="utf-8")
        return cached_clip, meta_path
