"""Phase 2 Narrative Understanding, OCR, Speaker Attribution & Clean-Plate Evaluation Script.

Evaluates Phase 2 against data/eval/ground_truth.json on:
  1. Speech Bubble Detection Precision, Recall, F1, and Mean IoU (Target: F1 >= 0.90)
  2. OCR Character Error Rate (CER) (Target: CER <= 0.10)
  3. Speaker Attribution Accuracy (Target: >= 0.90)
  4. Shot Type Classification Accuracy (Target: >= 0.85)
  5. Clean-Plate Text Removal Rate (Target: 100% bubble removal, 0 residual text bubbles)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m2a.eval.metrics import character_error_rate
from workers.assembler import inpaint_panel_bubbles
from workers.eval_segmentation import calculate_iou
from workers.understanding import (
    analyze_panel,
    count_residual_text_bubbles,
    detect_speech_bubbles_and_text,
)


def evaluate_understanding(gt_path: Path, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """Runs Phase 2 evaluation across all ground-truth panels."""
    gt_data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    eval_dir = Path(gt_path).parent
    gt_panels = gt_data.get("panels", [])

    bubble_tp = 0
    bubble_fp = 0
    bubble_fn = 0
    bubble_ious: List[float] = []
    cer_scores: List[float] = []
    speaker_correct = 0
    speaker_total = 0
    shot_correct = 0
    clean_plate_zero_bubbles = 0
    per_panel_reports: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="m2a_eval_p2_") as tmp_str:
        tmp_dir = Path(tmp_str)
        crops_dir = tmp_dir / "crops"
        clean_dir = tmp_dir / "clean_plates"
        crops_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

        for p in gt_panels:
            pid = p["panel_id"]
            page_path = eval_dir / p["page"]
            pb = p["bbox"]
            page_img = Image.open(page_path).convert("RGB")
            panel_crop = page_img.crop((pb["x"], pb["y"], pb["x"] + pb["w"], pb["y"] + pb["h"]))
            crop_file = tmp_dir / f"{pid}.png"
            panel_crop.save(crop_file, format="PNG")

            # 1. Pure CV Speech Bubble Detection evaluation
            detected_bubbles = detect_speech_bubbles_and_text(panel_crop)
            gt_dialogues = p.get("dialogue", [])
            rel_gt_boxes = [
                {
                    "x": int(d["bbox"]["x"]) - int(pb["x"]),
                    "y": int(d["bbox"]["y"]) - int(pb["y"]),
                    "w": int(d["bbox"]["w"]),
                    "h": int(d["bbox"]["h"]),
                }
                for d in gt_dialogues
                if d.get("bbox")
            ]

            matched_gt = set()
            matched_det = set()
            for d_idx, det in enumerate(detected_bubbles):
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g_box in enumerate(rel_gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou = calculate_iou(det["bbox"], g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
                if best_iou >= iou_threshold and best_g_idx >= 0:
                    matched_gt.add(best_g_idx)
                    matched_det.add(d_idx)
                    bubble_ious.append(best_iou)

            tp = len(matched_det)
            fp = len(detected_bubbles) - tp
            fn = len(rel_gt_boxes) - tp
            bubble_tp += tp
            bubble_fp += fp
            bubble_fn += fn

            # 2. Full Phase 2 semantic analysis (OCR + Speaker + Shot + Character Crops)
            sem = analyze_panel(
                panel_image_path=crop_file,
                panel_id=pid,
                language="en",
                reading_order=p.get("reading_order"),
                crops_dir=crops_dir,
            )

            if sem.get("shot") == p.get("shot_type"):
                shot_correct += 1

            pred_lines: List[Dict[str, Any]] = []
            for d in sem.get("dialogue", []):
                pred_lines.append({"text": d.get("text", ""), "speaker_id": d.get("speaker_id", "")})
            for s in sem.get("sfx", []):
                pred_lines.append({"text": s.get("text", ""), "speaker_id": "char_sfx"})

            for g_idx, gd in enumerate(gt_dialogues):
                gt_text = gd.get("text", "")
                gt_spk = f"char_{gd.get('speaker', 'narrator').lower().replace(' ', '_')}"
                speaker_total += 1
                if g_idx < len(pred_lines):
                    pred_text = pred_lines[g_idx]["text"]
                    pred_spk = pred_lines[g_idx]["speaker_id"]
                    cer_scores.append(character_error_rate(pred_text, gt_text))
                    if pred_spk == gt_spk:
                        speaker_correct += 1
                else:
                    cer_scores.append(1.0)

            # 3. Clean-Plate Inpainting & Residual Bubble Verification
            clean_file = clean_dir / f"{pid}_clean.png"
            bboxes_to_clean = [d["bbox"] for d in sem.get("dialogue", []) if d.get("bbox")]
            bboxes_to_clean.extend([s["bbox"] for s in sem.get("sfx", []) if s.get("bbox")])
            inpaint_panel_bubbles(crop_file, clean_file, bubble_bboxes=bboxes_to_clean)
            clean_img = Image.open(clean_file).convert("RGB")
            residual_bubbles = count_residual_text_bubbles(clean_img)
            if residual_bubbles == 0:
                clean_plate_zero_bubbles += 1

            per_panel_reports.append(
                {
                    "panel_id": pid,
                    "bubble_tp": tp,
                    "bubble_fp": fp,
                    "bubble_fn": fn,
                    "mean_bubble_iou": round(sum(bubble_ious[-tp:]) / tp, 4) if tp > 0 else 0.0,
                    "shot_match": sem.get("shot") == p.get("shot_type"),
                    "residual_bubbles_after_inpaint": residual_bubbles,
                }
            )

    precision = bubble_tp / (bubble_tp + bubble_fp) if (bubble_tp + bubble_fp) > 0 else 0.0
    recall = bubble_tp / (bubble_tp + bubble_fn) if (bubble_tp + bubble_fn) > 0 else 0.0
    bubble_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_bubble_iou = sum(bubble_ious) / len(bubble_ious) if bubble_ious else 0.0
    mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
    speaker_acc = speaker_correct / speaker_total if speaker_total > 0 else 0.0
    shot_acc = shot_correct / len(gt_panels) if gt_panels else 0.0
    clean_plate_rate = clean_plate_zero_bubbles / len(gt_panels) if gt_panels else 0.0

    passed_all = (
        bubble_f1 >= 0.90
        and mean_cer <= 0.10
        and speaker_acc >= 0.90
        and shot_acc >= 0.85
        and clean_plate_rate == 1.0
    )

    return {
        "total_panels": len(gt_panels),
        "bubble_precision": round(precision, 4),
        "bubble_recall": round(recall, 4),
        "bubble_f1": round(bubble_f1, 4),
        "mean_bubble_iou": round(mean_bubble_iou, 4),
        "ocr_cer": round(mean_cer, 4),
        "speaker_attribution_accuracy": round(speaker_acc, 4),
        "shot_classification_accuracy": round(shot_acc, 4),
        "clean_plate_removal_rate": round(clean_plate_rate, 4),
        "passed_target": bool(passed_all),
        "panels": per_panel_reports,
    }


if __name__ == "__main__":
    gt_file = Path("data/eval/ground_truth.json")
    res = evaluate_understanding(gt_file)
    print("=" * 68)
    print("PHASE 2 UNDERSTANDING, OCR & CLEAN-PLATE EVALUATION REPORT")
    print("=" * 68)
    print(f"Total Panels Evaluated:        {res['total_panels']}")
    print(f"Speech Bubble Precision:       {res['bubble_precision']:.4f}")
    print(f"Speech Bubble Recall:          {res['bubble_recall']:.4f}")
    print(f"Speech Bubble F1 (Target>=.9): {res['bubble_f1']:.4f}")
    print(f"Mean Bubble IoU:               {res['mean_bubble_iou']:.4f}")
    print(f"OCR CER (Target<=0.10):        {res['ocr_cer']:.4f}")
    print(f"Speaker Attribution Accuracy:  {res['speaker_attribution_accuracy']:.4f}")
    print(f"Shot Classification Accuracy:  {res['shot_classification_accuracy']:.4f}")
    print(f"Clean-Plate Removal Rate:      {res['clean_plate_removal_rate']:.4f}")
    print(f"Overall Status:                {'PASSED' if res['passed_target'] else 'FAILED'}")
    print("=" * 68)
