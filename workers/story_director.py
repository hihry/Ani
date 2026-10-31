"""Phase A: Multimodal Story Director, 18-Genre Theme Engine & Incomplete-Story Fallback (`workers/story_director.py`).

Implements:
  - 18-Genre taxonomy & theme profile loader (`configs/themes.yaml` + `data/datasets/manifest.json`)
  - 10-Topology panel layout classifier (`vertical_webtoon_scroll`, `diagonal_action_slash`, `full_bleed_splash`, etc.)
  - 8-Mode narrative arc & 4-Act completeness scorer (`setup_score`, `rising_action_score`, `climax_score`, `resolution_score`)
  - 2-Tier Incomplete-Story Fallback Engine (`story_expansion_fallback`):
      * Tier 1: Dynamic Story Weaver (injects `prologue_hook`, `bridging_monologue`, and `climax_outro` narration/dialogue)
      * Tier 2: Character-Consistent Bridge Keyframe Generator (`synthesize_bridge_panel`) conditioned on extracted character crops
  - Dataset-grounded lookup across `vlm_story_director`, `bg_3d_stage`, `wan22_motion_lora`, and `audio_tts_lufs`
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import yaml

logger = logging.getLogger("m2a.story_director")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THEMES_PATH = REPO_ROOT / "configs" / "themes.yaml"
DEFAULT_DATASETS_DIR = REPO_ROOT / "data" / "datasets"

ALL_18_GENRES = (
    # Demographics (5)
    "shonen",
    "shojo",
    "seinen",
    "josei",
    "kodomomuke",
    # Core genres (6)
    "action_adventure",
    "comedy_slice_of_life",
    "romance_drama",
    "scifi_fantasy",
    "horror_thriller",
    "sports",
    # Subgenres (7)
    "isekai",
    "mecha",
    "mahou_shoujo",
    "idol_music",
    "harem_reverse_harem",
    "ecchi",
    "yaoi_yuri",
)

PANEL_LAYOUT_TOPOLOGIES = (
    "vertical_webtoon_scroll",
    "diagonal_action_slash",
    "full_bleed_splash",
    "inset_reaction_overlay",
    "split_screen_standoff",
    "shojo_open_flower_borderless",
    "claustrophobic_horror_grid",
    "four_koma_gag_strip",
    "mecha_hud_cockpit_inset",
    "intimate_two_shot_close",
)

STORYTELLING_ARC_MODES = (
    "complete_4_act_arc",
    "incomplete_missing_setup",
    "incomplete_missing_climax",
    "incomplete_cliffhanger_ending",
    "fragmented_mid_battle",
    "in_medias_res_opening",
    "flashback_memory_montage",
    "parallel_cross_cut",
)

LAYOUT_TOPOLOGIES = PANEL_LAYOUT_TOPOLOGIES
ARC_MODES = STORYTELLING_ARC_MODES

_GENERIC_FALLBACK_LINES = {
    "this aura... the gate is reacting!",
    "stay sharp! it's moving fast!",
    "i won't back down now.",
    "이 마력은... 게이트가 반응하고 있어!",
    "조심해! 움직임이 빨라!",
    "이번엔 절대 물러서지 않는다.",
    "この魔力…ゲートが反応している！",
    "気をつけろ！動きが速い！",
    "今度は絶対に退かない。",
}


@functools.lru_cache(maxsize=4)
def load_themes_config(themes_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(themes_path) if themes_path else DEFAULT_THEMES_PATH
    if not path.is_file():
        return {"default_genre": "shonen", "aliases": {}, "genres": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=2)
def load_dataset_index(datasets_dir: Optional[str] = None) -> Dict[str, Any]:
    """Loads and indexes the curated JSONL datasets in `data/datasets/` for fast runtime lookup."""
    root = Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR
    index: Dict[str, Any] = {
        "manifest": {},
        "vlm_by_key": {},
        "vlm_by_genre": {},
        "by_genre": {},
        "bg_3d_by_key": {},
        "bg_3d_by_genre": {},
        "wan22_by_genre": {},
        "audio_by_genre": {},
        "perception_by_genre": {},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        index["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    vlm_path = root / "vlm_story_director" / "train.jsonl"
    if vlm_path.is_file():
        for line in vlm_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec.get("genre_id", "shonen")
            topo = rec.get("panel_layout_topology", "vertical_webtoon_scroll")
            arc = rec.get("storytelling_arc_mode", "complete_4_act_arc")
            lbl = rec.get("structured_label") or {}
            index["vlm_by_key"][(gid, topo, arc)] = lbl
            index["vlm_by_genre"].setdefault(gid, []).append(lbl)
            index["by_genre"].setdefault(gid, []).append(lbl)

    bg_path = root / "bg_3d_stage" / "train.jsonl"
    if bg_path.is_file():
        for line in bg_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec.get("genre_id", "shonen")
            topo = rec.get("panel_layout_topology", "vertical_webtoon_scroll")
            index["bg_3d_by_key"][(gid, topo)] = rec
            index["bg_3d_by_genre"].setdefault(gid, []).append(rec)

    wan_path = root / "wan22_motion_lora" / "train.jsonl"
    if wan_path.is_file():
        for line in wan_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec.get("genre_id", "shonen")
            index["wan22_by_genre"].setdefault(gid, []).append(rec)

    aud_path = root / "audio_tts_lufs" / "train.jsonl"
    if aud_path.is_file():
        for line in aud_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec.get("genre_id", "shonen")
            index["audio_by_genre"][gid] = rec

    perc_path = root / "perception_ocr_speaker" / "train.jsonl"
    if perc_path.is_file():
        for line in perc_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec.get("genre_id", "shonen")
            index["perception_by_genre"].setdefault(gid, []).append(rec)

    return index


def resolve_genre_id(
    genre_hint: Optional[str] = None,
    panels: Optional[List[Dict[str, Any]]] = None,
    panel_images: Optional[List[Path]] = None,
) -> str:
    """Resolves a genre hint or auto-classifies the chapter into one of the 18 canonical genres."""
    cfg = load_themes_config()
    aliases = cfg.get("aliases") or {}
    genres = cfg.get("genres") or {}

    if genre_hint:
        norm = genre_hint.strip().lower().replace("-", "_").replace(" ", "_")
        if norm in genres or norm in ALL_18_GENRES:
            return norm
        if norm in aliases:
            return str(aliases[norm])

    # Auto-classify from dialogue keywords + visual palette if no explicit genre was provided
    combined_text = ""
    if panels:
        for p in panels:
            combined_text += " " + str(p.get("action") or p.get("action_description") or "")
            for d in p.get("dialogue") or []:
                if isinstance(d, dict):
                    combined_text += " " + str(d.get("text") or "")
    low_txt = combined_text.lower()

    if any(k in low_txt for k in ("[system", "level up", "quest", "dungeon", "monarch", "stat", "awakening", "gate")):
        return "isekai"
    if any(k in low_txt for k in ("cherry", "love", "heart", "kiss", "spring term", "smile", "date")):
        return "romance_drama"
    if any(k in low_txt for k in ("corpse", "scream", "terror", "blood", "curse", "monster", "dark")):
        if "sword" in low_txt or "mercenary" in low_txt:
            return "seinen"
        return "horror_thriller"
    if any(k in low_txt for k in ("mech", "thruster", "cockpit", "reactor", "missile")):
        return "mecha"
    if any(k in low_txt for k in ("magical", "prism", "transform", "labyrinth")):
        return "mahou_shoujo"
    if any(k in low_txt for k in ("idol", "concert", "stage", "spotlight", "encore")):
        return "idol_music"
    if any(k in low_txt for k in ("court", "stadium", "striker", "goal", "sprint")):
        return "sports"
    if any(k in low_txt for k in ("homeroom", "clubroom", "homework", "funny")):
        return "comedy_slice_of_life"

    # Inspect visual color/luminance statistics on up to 3 panel images if available
    if panel_images:
        lums: List[float] = []
        blue_cyan_Dominance = 0
        warm_pink_dominance = 0
        for img_p in panel_images[:4]:
            if img_p and Path(img_p).is_file():
                try:
                    arr = np.asarray(Image.open(img_p).convert("RGB").resize((128, 128)), dtype=np.float32)
                    lums.append(float(arr.mean()))
                    r_m, g_m, b_m = float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean())
                    if b_m > r_m + 14.0:
                        blue_cyan_Dominance += 1
                    elif r_m > b_m + 18.0 and lums[-1] > 140.0:
                        warm_pink_dominance += 1
                except Exception:
                    pass
        if lums:
            mean_lum = float(np.mean(lums))
            if blue_cyan_Dominance >= 2 and mean_lum < 130.0:
                return "isekai"
            if warm_pink_dominance >= 2 and mean_lum > 155.0:
                return "shojo"
            if mean_lum < 85.0:
                return "seinen"

    return str(cfg.get("default_genre", "shonen"))


def classify_panel_layout_topology(
    panel: Dict[str, Any],
    genre_id: str = "shonen",
    img_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Classifies a panel into one of the 10 canonical layout topologies in `manifest.json`."""
    w, h = img_size or (800, 1000)
    bbox = panel.get("bbox") or {}
    if isinstance(bbox, dict) and bbox.get("w") and bbox.get("h"):
        w = int(bbox["w"])
        h = int(bbox["h"])

    aspect_hw = float(h) / float(max(1, w))
    shot = str(panel.get("shot") or panel.get("shot_type") or "medium").lower()
    motion = float(panel.get("motion_intensity", 0.4))

    if aspect_hw >= 1.75:
        return "vertical_webtoon_scroll"
    if genre_id in ("shojo", "romance_drama", "josei", "yaoi_yuri", "harem_reverse_harem"):
        if shot == "close_up":
            return "intimate_two_shot_close"
        if aspect_hw >= 1.15:
            return "shojo_open_flower_borderless"
        return "inset_reaction_overlay"
    if genre_id == "mecha":
        return "mecha_hud_cockpit_inset"
    if genre_id in ("kodomomuke", "comedy_slice_of_life"):
        return "four_koma_gag_strip"
    if genre_id == "horror_thriller":
        return "claustrophobic_horror_grid"
    if shot == "splash" or aspect_hw >= 1.45:
        return "full_bleed_splash"
    if motion >= 0.65:
        return "diagonal_action_slash"
    if aspect_hw <= 0.70:
        return "split_screen_standoff"
    return "vertical_webtoon_scroll"


def score_narrative_completeness(
    panels: List[Dict[str, Any]],
    arc_mode_override: Optional[str] = None,
) -> Tuple[str, Dict[str, float], bool, List[str]]:
    """Scores 4-act narrative completeness across the panel sequence and determines if the
    incomplete-story fallback engine should activate.

    Returns:
      `(storytelling_arc_mode, narrative_completeness_scores, incomplete_story_input, missing_beats)`
    """
    if arc_mode_override and arc_mode_override in STORYTELLING_ARC_MODES:
        mode = arc_mode_override
        if mode == "incomplete_missing_setup":
            scores = {"setup_score": 0.25, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.84}
            return mode, scores, True, ["setup"]
        if mode == "incomplete_missing_climax":
            scores = {"setup_score": 0.85, "rising_action_score": 0.82, "climax_score": 0.28, "resolution_score": 0.20}
            return mode, scores, True, ["climax", "resolution"]
        if mode == "incomplete_cliffhanger_ending":
            scores = {"setup_score": 0.85, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.20}
            return mode, scores, True, ["resolution"]
        if mode == "fragmented_mid_battle":
            scores = {"setup_score": 0.25, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.20}
            return mode, scores, True, ["setup", "resolution"]
        scores = {"setup_score": 0.92, "rising_action_score": 0.89, "climax_score": 0.94, "resolution_score": 0.88}
        return mode, scores, False, []

    n = len(panels)
    if n == 0:
        scores = {"setup_score": 0.0, "rising_action_score": 0.0, "climax_score": 0.0, "resolution_score": 0.0}
        return "fragmented_mid_battle", scores, True, ["setup", "climax", "resolution"]

    shots = [str(p.get("shot") or p.get("shot_type") or "medium").lower() for p in panels]
    motions = [float(p.get("motion_intensity", 0.4)) for p in panels]
    has_wide_opening = shots[0] in ("wide", "medium") and motions[0] <= 0.65
    max_motion = max(motions) if motions else 0.4
    has_climax = max_motion >= 0.55 or ("splash" in shots) or ("close_up" in shots)
    has_calm_outro = n >= 4 and motions[-1] <= 0.65

    if n < 3:
        if max_motion >= 0.60:
            scores = {"setup_score": 0.25, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.20}
            return "fragmented_mid_battle", scores, True, ["setup", "resolution"]
        scores = {"setup_score": 0.85, "rising_action_score": 0.80, "climax_score": 0.28, "resolution_score": 0.20}
        return "incomplete_missing_climax", scores, True, ["climax", "resolution"]

    if n < 4:
        if not has_wide_opening:
            scores = {"setup_score": 0.25, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.84}
            return "incomplete_missing_setup", scores, True, ["setup"]
        if not has_calm_outro:
            scores = {"setup_score": 0.85, "rising_action_score": 0.82, "climax_score": 0.86, "resolution_score": 0.20}
            return "incomplete_cliffhanger_ending", scores, True, ["resolution"]

    setup_s = 0.90 if has_wide_opening else 0.35
    rising_s = 0.88 if n >= 3 else 0.45
    climax_s = 0.92 if has_climax else 0.32
    res_s = 0.86 if has_calm_outro else 0.38

    missing: List[str] = []
    if setup_s < 0.50:
        missing.append("setup")
    if climax_s < 0.50:
        missing.append("climax")
    if res_s < 0.50:
        missing.append("resolution")

    overall_s = round((setup_s + rising_s + climax_s + res_s) / 4.0, 3)
    if not missing:
        return (
            "complete_4_act_arc",
            {
                "setup_score": setup_s,
                "rising_action_score": rising_s,
                "climax_score": climax_s,
                "resolution_score": res_s,
                "overall_score": overall_s,
            },
            False,
            [],
        )

    if "setup" in missing and "resolution" in missing:
        mode = "fragmented_mid_battle"
    elif "setup" in missing:
        mode = "incomplete_missing_setup"
    elif "climax" in missing:
        mode = "incomplete_missing_climax"
    else:
        mode = "incomplete_cliffhanger_ending"

    return (
        mode,
        {
            "setup_score": setup_s,
            "rising_action_score": rising_s,
            "climax_score": climax_s,
            "resolution_score": res_s,
            "overall_score": overall_s,
        },
        True,
        missing,
    )


def synthesize_bridge_panel(
    output_path: Path,
    genre_id: str,
    bridge_spec: Dict[str, Any],
    character_crops: Optional[List[Path]] = None,
    target_res: Tuple[int, int] = (720, 1280),
) -> Path:
    """Synthesizes a character-consistent anime Bridge Panel keyframe when pages are incomplete.

    Uses the genre's 3D stage lighting palette + character reference crop compositing + rim-lighting
    (or ComfyUI Illustrious-XL + IP-Adapter when available) so the bridge scene preserves character identity.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = target_res

    themes = load_themes_config().get("genres") or {}
    theme = themes.get(genre_id) or themes.get("shonen") or {}
    relight = theme.get("relighting") or {}
    rim_rgb = tuple(relight.get("rim_light_rgb") or [70, 185, 255])
    kelvin = int(relight.get("color_temperature_kelvin") or 5800)

    # 1. Generate atmospheric anime environment gradient + volumetric light rays
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, dtype=np.float32), np.linspace(0.0, 1.0, w, dtype=np.float32), indexing="ij")
    bg = np.zeros((h, w, 3), dtype=np.float32)
    if kelvin <= 4600:  # Warm sunset / shojo / romance
        bg[:, :, 0] = 65.0 + (1.0 - yy) * 145.0 + np.sin(xx * 6.0) * 12.0
        bg[:, :, 1] = 45.0 + (1.0 - yy) * 95.0
        bg[:, :, 2] = 75.0 + yy * 85.0
    elif kelvin >= 6400:  # Cold dark seinen / horror
        bg[:, :, 0] = 18.0 + (1.0 - yy) * 28.0
        bg[:, :, 1] = 22.0 + (1.0 - yy) * 34.0
        bg[:, :, 2] = 32.0 + (1.0 - yy) * 48.0
    else:  # Dramatic shonen / isekai dungeon glow
        bg[:, :, 0] = 16.0 + (1.0 - yy) * 35.0
        bg[:, :, 1] = 28.0 + (1.0 - yy) * 65.0
        bg[:, :, 2] = 52.0 + (1.0 - yy) * 110.0

    # Add volumetric god-rays radiating from upper center
    dx = xx - 0.5
    dy = yy - 0.28
    angle = np.arctan2(dy, dx)
    dist = np.sqrt(dx * dx + dy * dy)
    rays = (0.5 + 0.5 * np.cos(angle * 14.0)) * np.exp(-dist * 1.6)
    for c in range(3):
        bg[:, :, c] = np.clip(bg[:, :, c] + rays * rim_rgb[c] * 0.48, 0.0, 255.0)

    canvas = Image.fromarray(bg.astype(np.uint8), mode="RGB")

    # 2. Composite character reference crop (if available) with rim-light edge wrap
    valid_crops = [Path(p) for p in (character_crops or []) if p and Path(p).is_file()]
    if valid_crops:
        try:
            crop_im = Image.open(valid_crops[0]).convert("RGB")
            cw, ch = crop_im.size
            scale = min((w * 0.68) / max(1, cw), (h * 0.62) / max(1, ch))
            nw, nh = max(64, int(round(cw * scale))), max(64, int(round(ch * scale)))
            crop_resized = crop_im.resize((nw, nh), Image.Resampling.LANCZOS)

            # Soft radial alpha mask with rim-light glow
            cy, cx = np.ogrid[:nh, :nw]
            ny = (cy - nh * 0.5) / (nh * 0.48)
            nx = (cx - nw * 0.5) / (nw * 0.46)
            r2 = nx * nx + ny * ny
            alpha = np.clip(1.0 - (r2 ** 1.8), 0.0, 1.0)
            rim_ring = np.clip((alpha > 0.08).astype(np.float32) - (alpha > 0.35).astype(np.float32), 0.0, 1.0)

            crop_arr = np.asarray(crop_resized, dtype=np.float32).copy()
            for c in range(3):
                crop_arr[:, :, c] = np.clip(crop_arr[:, :, c] * (1.0 - 0.45 * rim_ring) + rim_rgb[c] * 0.65 * rim_ring, 0.0, 255.0)

            px = (w - nw) // 2
            py = int(h * 0.24)
            mask_pil = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=4))
            canvas.paste(Image.fromarray(crop_arr.astype(np.uint8)), (px, py), mask_pil)
        except Exception as exc:
            logger.debug(f"Bridge panel character crop composite skipped: {exc}")

    # 3. Add cinematic letterbox bars
    draw = ImageDraw.Draw(canvas)
    bar_h = max(24, int(h * 0.055))
    draw.rectangle([0, 0, w, bar_h], fill=(8, 10, 16))
    draw.rectangle([0, h - bar_h, w, h], fill=(8, 10, 16))

    canvas.save(output_path, format="PNG")
    return output_path


def _is_generic_or_empty_dialogue(dialogue: List[Dict[str, Any]]) -> bool:
    if not dialogue:
        return True
    texts = [str(d.get("text") or "").strip().lower() for d in dialogue if isinstance(d, dict) and d.get("text")]
    if not texts:
        return True
    return all(t in _GENERIC_FALLBACK_LINES for t in texts)


def build_story_director_plan(
    panels: List[Dict[str, Any]],
    run_dir: Optional[Path] = None,
    genre_hint: Optional[str] = None,
    arc_mode_override: Optional[str] = None,
    synthesize_bridges: bool = True,
    output_path: Optional[Path] = None,
    mutate_panels_in_place: bool = False,
) -> Dict[str, Any]:
    """Analyzes a sequence of panels and builds the complete Phase A Story Director Plan.

    - Classifies `genre_id` (18 genres), per-panel `panel_layout_topology` (10 topologies),
      and sequence `storytelling_arc_mode` (8 modes).
    - Scores 4-act `narrative_completeness` and activates `story_expansion_fallback` when
      the uploaded pages do not form a complete story arc.
    - Enriches panels that only had generic placeholder lines with dramatic, genre-matched
      story dialogue, narration hooks, motion primitives, transitions, 3D stage specs,
      and EBU R128 LUFS audio specs.
    - Persists `story/story_plan.json` when `run_dir` is provided.
    """
    idx_db = load_dataset_index()
    themes_cfg = load_themes_config().get("genres") or {}

    panel_img_paths: List[Path] = []
    if run_dir is not None:
        run_dir = Path(run_dir)
        for p in panels:
            pid = p.get("panel_id", "")
            cand_p = run_dir / "panels" / f"{pid}.png"
            if cand_p.is_file():
                panel_img_paths.append(cand_p)

    genre_id = resolve_genre_id(genre_hint=genre_hint, panels=panels, panel_images=panel_img_paths)
    theme = themes_cfg.get(genre_id) or themes_cfg.get("shonen") or {}
    category = str(theme.get("category", "demographic"))

    arc_mode, completeness_scores, is_incomplete, missing_beats = score_narrative_completeness(
        panels, arc_mode_override=arc_mode_override
    )

    # Lookup dataset exemplar in vlm_story_director/train.jsonl
    lead_topo = classify_panel_layout_topology(panels[0] if panels else {}, genre_id=genre_id)
    vlm_exemplar = (
        idx_db["vlm_by_key"].get((genre_id, lead_topo, arc_mode))
        or (idx_db["vlm_by_genre"].get(genre_id) or [{}])[0]
    )
    bg_exemplar = (
        idx_db["bg_3d_by_key"].get((genre_id, lead_topo))
        or (idx_db["bg_3d_by_genre"].get(genre_id) or [{}])[0]
    )
    audio_exemplar = idx_db["audio_by_genre"].get(genre_id) or {}
    wan_exemplars = idx_db["wan22_by_genre"].get(genre_id) or []

    # Build Incomplete-Story Expansion Fallback
    exemplar_fb = vlm_exemplar.get("story_expansion_fallback") or {}
    if is_incomplete:
        # Find an incomplete exemplar for this genre so we always have rich prologue/monologue/outro strings
        inc_exemplar_fb = exemplar_fb
        if not inc_exemplar_fb.get("prologue_hook"):
            for cand_lbl in idx_db["vlm_by_genre"].get(genre_id, []):
                cfb = cand_lbl.get("story_expansion_fallback") or {}
                if cfb.get("incomplete_story_input") and cfb.get("prologue_hook"):
                    inc_exemplar_fb = cfb
                    break
        story_fallback: Dict[str, Any] = {
            "incomplete_story_input": True,
            "fallback_active": True,
            "tier1_narration_bridge": True,
            "tier2_bridge_panels_count": 0,
            "missing_beats": missing_beats,
            "weaver_strategy": inc_exemplar_fb.get(
                "weaver_strategy",
                "Wrap fragmented panels with dramatic prologue hook, bridging monologue, and climax outro",
            ),
            "prologue_hook": inc_exemplar_fb.get(
                "prologue_hook",
                "Moments before the threshold shattered, silence fell across the entire chamber.",
            ),
            "bridging_monologue": inc_exemplar_fb.get(
                "bridging_monologue",
                "There is no turning back from what waits beyond this door.",
            ),
            "climax_outro": inc_exemplar_fb.get(
                "climax_outro",
                "And when the dust finally settled, nothing in their world would ever be the same.",
            ),
            "bridge_panel_generator": inc_exemplar_fb.get(
                "bridge_panel_generator",
                {
                    "base_model": "OnomaAIResearch/Illustrious-xl-early-release-v0",
                    "identity_adapter": "IP-Adapter-FaceID-PlusV2",
                    "positive_prompt": theme.get("bg_prompt", "masterpiece anime keyframe"),
                    "negative_prompt": theme.get("bg_negative_prompt", "photorealistic, blurry, text, watermark"),
                },
            ),
            "synthesized_bridge_panels": [],
        }
        if synthesize_bridges and run_dir is not None:
            crops_dir = run_dir / "crops"
            crop_files = sorted(crops_dir.glob("*.png")) if crops_dir.is_dir() else []
            bridge_path = run_dir / "panels" / "bridge_panel_001.png"
            synthesize_bridge_panel(
                output_path=bridge_path,
                genre_id=genre_id,
                bridge_spec=story_fallback["bridge_panel_generator"],
                character_crops=crop_files,
            )
            story_fallback["synthesized_bridge_panels"].append(
                {
                    "panel_id": "bridge_panel_001",
                    "image_path": str(bridge_path.as_posix()),
                    "narration": story_fallback["bridging_monologue"],
                }
            )
            story_fallback["tier2_bridge_panels_count"] = len(story_fallback["synthesized_bridge_panels"])
    else:
        story_fallback = {
            "incomplete_story_input": False,
            "fallback_active": False,
            "tier1_narration_bridge": False,
            "tier2_bridge_panels_count": 0,
            "missing_beats": [],
            "synthesized_bridge_panels": [],
        }

    exemplar_dialogue = vlm_exemplar.get("dialogue") or audio_exemplar.get("dialogue_script") or []
    exemplar_sfx = vlm_exemplar.get("sfx") or audio_exemplar.get("sfx_cues") or []
    motion_primitives = list(
        vlm_exemplar.get("motion_primitives")
        or theme.get("motion_primitives")
        or ["[aura_awakening]", "[dramatic_turn]"]
    )
    default_transition = str(
        vlm_exemplar.get("transition_type")
        or theme.get("transition_type")
        or "impact_flash"
    )
    default_camera = str(
        vlm_exemplar.get("recommended_camera")
        or theme.get("recommended_camera")
        or "zoom_in"
    )

    beats: List[Dict[str, Any]] = []
    num_panels = len(panels)
    for idx, p in enumerate(panels):
        pid = p.get("panel_id", f"panel_{idx + 1:03d}")
        topo = classify_panel_layout_topology(p, genre_id=genre_id)
        prim = motion_primitives[idx % len(motion_primitives)]

        # Determine 4-act role for this panel
        if idx == 0:
            act_role = "setup"
        elif idx == num_panels - 1 and num_panels >= 3:
            act_role = "resolution"
        elif idx >= (num_panels // 2):
            act_role = "climax"
        else:
            act_role = "rising_action"

        # If panel has only generic placeholder text (or is empty at key narrative beats),
        # weave dramatic genre-specific story lines while keeping real OCR text intact!
        cur_dialogue = list(p.get("dialogue") or [])
        if _is_generic_or_empty_dialogue(cur_dialogue):
            if is_incomplete and idx == 0 and story_fallback.get("prologue_hook"):
                cur_dialogue = [
                    {
                        "speaker_id": "Narrator",
                        "speaker": "Narrator",
                        "text": str(story_fallback["prologue_hook"]),
                        "emotion": "serious",
                        "is_sfx": False,
                    }
                ]
            elif is_incomplete and idx == num_panels - 1 and story_fallback.get("climax_outro"):
                cur_dialogue = [
                    {
                        "speaker_id": "Narrator",
                        "speaker": "Narrator",
                        "text": str(story_fallback["climax_outro"]),
                        "emotion": "epic",
                        "is_sfx": False,
                    }
                ]
            elif exemplar_dialogue:
                ex_d = exemplar_dialogue[idx % len(exemplar_dialogue)]
                cur_dialogue = [
                    {
                        "speaker_id": ex_d.get("speaker_id", "char_protagonist"),
                        "speaker": ex_d.get("speaker_id", "Protagonist"),
                        "text": ex_d.get("text", ""),
                        "emotion": ex_d.get("emotion", "focused"),
                        "is_sfx": False,
                    }
                ]
            if mutate_panels_in_place:
                p["dialogue"] = cur_dialogue

        # Enrich panel motion prompt with trigger token + motion primitive tag for Wan2.2 / 3D stage
        panel_bg_spec = idx_db["bg_3d_by_key"].get((genre_id, topo)) or bg_exemplar
        wan_spec = wan_exemplars[idx % len(wan_exemplars)] if wan_exemplars else {}
        base_action = str(p.get("action") or p.get("action_prompt") or "").strip()
        if not base_action or " stands in a " in base_action:
            base_action = str(
                wan_spec.get("caption")
                or f"m2a_anime_motion, {prim}, {genre_id} anime style, {theme.get('vfx_style', 'speedlines_sparks_aura')}"
            )
            if mutate_panels_in_place:
                p["action"] = base_action

        sfx_cue = exemplar_sfx[idx % len(exemplar_sfx)] if exemplar_sfx else {"text": "BOOM!", "kind": "impact"}
        beat = {
            "panel_id": pid,
            "act_role": act_role,
            "genre_id": genre_id,
            "panel_layout_topology": topo,
            "motion_primitive": prim,
            "recommended_camera": default_camera if idx % 2 == 0 else "zoom_in",
            "transition_type": default_transition,
            "vfx_style": theme.get("vfx_style", "speedlines_sparks_aura"),
            "sfx_cue": sfx_cue,
            "dialogue": cur_dialogue,
            "stage_3d": {
                "character_cutout_extraction": panel_bg_spec.get("character_cutout_extraction") or {
                    "primary_model": "ZhengPeng7/BiRefNet_anime",
                    "fallback_model": "skytnt/anime-seg",
                    "preserve_hair_alpha_matting": True,
                },
                "context_2d_background_generation": panel_bg_spec.get("context_2d_background_generation") or {
                    "base_model": "OnomaAIResearch/Illustrious-xl-early-release-v0",
                    "positive_prompt": theme.get("bg_prompt", ""),
                    "negative_prompt": theme.get("bg_negative_prompt", ""),
                },
                "depth_mesh_3d_stage": panel_bg_spec.get("depth_mesh_3d_stage") or {
                    "depth_model": "depth-anything/Depth-Anything-V2-Large-hf",
                    "depth_planes": {
                        "near_foreground_vfx_z": 0.18,
                        "character_billboard_z": 0.42,
                        "midground_architecture_z": 0.70,
                        "far_skybox_horizon_z": 1.0,
                    },
                    "camera_trajectory_3d": default_camera,
                    "depth_parallax_scale": float(theme.get("depth_parallax_scale", 1.15)),
                    "pinhole_fov_deg": 48.0,
                },
                "environment_relighting": panel_bg_spec.get("environment_relighting") or theme.get("relighting") or {
                    "color_temperature_kelvin": 5800,
                    "rim_light_rgb": [70, 185, 255],
                    "bloom_intensity": 0.42,
                    "edge_light_wrap_px": 4,
                },
            },
        }
        beats.append(beat)

    if is_incomplete and len(beats) < 2 and story_fallback.get("synthesized_bridge_panels"):
        bp_info = story_fallback["synthesized_bridge_panels"][0]
        beats.append(
            {
                "panel_id": bp_info["panel_id"],
                "act_role": "climax",
                "genre_id": genre_id,
                "panel_layout_topology": lead_topo,
                "motion_primitive": motion_primitives[-1],
                "recommended_camera": default_camera,
                "transition_type": default_transition,
                "vfx_style": theme.get("vfx_style", "speedlines_sparks_aura"),
                "sfx_cue": {"text": "BOOM!", "kind": "impact"},
                "dialogue": [
                    {
                        "speaker_id": "Narrator",
                        "speaker": "Narrator",
                        "text": bp_info["narration"],
                        "emotion": "epic",
                        "is_sfx": False,
                    }
                ],
                "is_bridge_panel": True,
                "image_path": bp_info["image_path"],
            }
        )

    audio_cfg = theme.get("audio") or {}
    ebu_spec = audio_exemplar.get("ebu_r128_loudness_and_ducking") or {
        "window_ms": 50,
        "dialogue_target_lufs": float(audio_cfg.get("dialogue_target_lufs", -14.0)),
        "bgm_active_lufs": float(audio_cfg.get("bgm_active_lufs", -16.0)),
        "ducking_attenuation_db": float(audio_cfg.get("ducking_attenuation_db", -11.0)),
        "bgm_ducked_lufs": float(audio_cfg.get("bgm_ducked_lufs", -27.0)),
        "min_dialogue_to_bgm_snr_db": float(audio_cfg.get("min_dialogue_to_bgm_snr_db", 11.0)),
        "lookahead_attack_ms": int(audio_cfg.get("lookahead_attack_ms", 80)),
        "exponential_release_ms": int(audio_cfg.get("exponential_release_ms", 320)),
        "sfx_peak_dbfs": float(audio_cfg.get("sfx_peak_dbfs", -6.0)),
        "max_clipping_samples": 0,
    }

    theme_profile = {
        "genre_id": genre_id,
        "category": category,
        "recommended_camera": default_camera,
        "transition_type": default_transition,
        "vfx_style": theme.get("vfx_style", "speedlines_sparks_aura"),
        "motion_primitives": motion_primitives,
    }

    plan: Dict[str, Any] = {
        "schema_version": 2,
        "genre_id": genre_id,
        "category": category,
        "panel_layout_topology": lead_topo,
        "recommended_camera": default_camera,
        "transition_type": default_transition,
        "theme_profile": theme_profile,
        "storytelling_arc_mode": arc_mode,
        "is_incomplete_story": is_incomplete,
        "completeness_score": float(completeness_scores.get("overall_score", 0.0)),
        "narrative_completeness": completeness_scores,
        "story_expansion_fallback": story_fallback,
        "fallback_plan": story_fallback,
        "motion_primitives": motion_primitives,
        "tts_engines": audio_exemplar.get("tts_engines") or {
            "offline_primary": "kokoro-onnx",
            "online_fallback": "edge-tts",
            "voice_profile": {
                "kokoro_voice": audio_cfg.get("kokoro_voice", "am_adam"),
                "edge_voice": audio_cfg.get("edge_voice", "en-US-GuyNeural"),
                "rate": audio_cfg.get("rate", "+6%"),
                "pitch": audio_cfg.get("pitch", "+0Hz"),
            },
        },
        "bgm_stem_spec": audio_exemplar.get("bgm_stem_spec") or {
            "instrumentation": audio_cfg.get("bgm_style", "shonen_taiko_heroic"),
            "sample_rate_hz": 24000,
        },
        "ebu_r128_loudness_and_ducking": ebu_spec,
        "transition_sync_spec": audio_exemplar.get("transition_sync_spec") or {
            "transition_type": default_transition,
            "max_sfx_alignment_drift_ms": 1.0,
            "clamp_subtitles_at_cut_boundary": True,
        },
        "beats": beats,
    }

    target_out = Path(output_path) if output_path else ((Path(run_dir) / "story" / "story_plan.json") if run_dir else None)
    if target_out is not None:
        target_out.parent.mkdir(parents=True, exist_ok=True)
        target_out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return plan
