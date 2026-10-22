#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "workflows" / "comfyui" / "manifest.yaml"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def select_models(manifest: Dict[str, Any], tier: str | None, model_id: str | None) -> List[tuple[str, Dict[str, Any]]]:
    models = manifest.get("models", {})
    selected: List[tuple[str, Dict[str, Any]]] = []
    for mid, spec in models.items():
        if model_id and mid != model_id:
            continue
        if tier and spec.get("tier") != tier:
            continue
        selected.append((mid, spec))
    return selected


def resolve_models_root(comfy_dir: Path) -> Path:
    """Resolve model storage root, respecting ComfyUI extra_model_paths.yaml if present."""
    extra_paths_file = comfy_dir / "extra_model_paths.yaml"
    if extra_paths_file.is_file():
        try:
            extra_cfg = yaml.safe_load(extra_paths_file.read_text(encoding="utf-8")) or {}
            for section in extra_cfg.values():
                if isinstance(section, dict) and section.get("base_path"):
                    return Path(str(section["base_path"]))
        except Exception:
            pass
    return comfy_dir / "models"


def download_manifest_models(
    manifest_path: Path = DEFAULT_MANIFEST,
    comfy_dir: Path = Path.home() / "ComfyUI",
    tier: str | None = None,
    model_id: str | None = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    selected = select_models(manifest, tier=tier, model_id=model_id)
    if not selected:
        raise ValueError(f"No models matched tier={tier!r}, model_id={model_id!r}")

    models_root = resolve_models_root(comfy_dir)
    usage_probe = models_root if models_root.exists() else (models_root.parent if models_root.parent.exists() else REPO_ROOT)
    free_bytes = shutil.disk_usage(usage_probe).free
    plan: List[Dict[str, Any]] = []
    for mid, spec in selected:
        for file_entry in spec.get("files", []):
            target_subdir = file_entry.get("target_dir", "diffusion_models")
            filename = Path(file_entry["file"]).name
            dest = models_root / target_subdir / filename
            plan.append(
                {
                    "model_id": mid,
                    "repo": file_entry["repo"],
                    "file": file_entry["file"],
                    "revision": file_entry.get("revision", "main"),
                    "sha256": file_entry.get("sha256"),
                    "dest": str(dest),
                    "free_gb": round(free_bytes / (1024 ** 3), 2),
                }
            )

    if dry_run:
        return plan

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for live weight downloads; use --dry-run to verify manifest") from exc

    # Place HuggingFace cache on the same high-capacity volume as models_root so /home never fills up
    hf_cache_dir = models_root / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))

    for item in plan:
        dest_path = Path(item["dest"])
        expected_sha = item.get("sha256")
        if dest_path.is_file() and expected_sha and compute_sha256(dest_path) == expected_sha:
            print(f"[{item['model_id']}] Already verified: {dest_path}")
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{item['model_id']}] Downloading {item['repo']} :: {item['file']} -> {dest_path} ...")
        downloaded = Path(
            hf_hub_download(
                repo_id=item["repo"],
                filename=item["file"],
                revision=item["revision"],
                local_dir=str(dest_path.parent),
                cache_dir=str(hf_cache_dir),
            )
        )
        if downloaded != dest_path and downloaded.exists():
            shutil.move(str(downloaded), str(dest_path))
        if expected_sha:
            actual_sha = compute_sha256(dest_path)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA256 mismatch for {dest_path}: got {actual_sha}, expected {expected_sha}")
    return plan


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ComfyUI models for M2A Phase 3")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--comfy-dir", type=Path, default=Path.home() / "ComfyUI")
    parser.add_argument("--tier", choices=["t1s", "t1", "t2"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = download_manifest_models(
        manifest_path=args.manifest,
        comfy_dir=args.comfy_dir,
        tier=args.tier,
        model_id=args.model,
        dry_run=args.dry_run,
    )
    for entry in plan:
        print(f"[{entry['model_id']}] {entry['repo']} :: {entry['file']} -> {entry['dest']} (free: {entry['free_gb']} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
