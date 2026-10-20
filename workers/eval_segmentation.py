"""Panel detection evaluation script for Phase 1.

Evaluates panel segmentation against data/eval/ground_truth.json
using Intersection over Union (IoU >= 0.5), Precision, Recall, and F1 score.
Target in proposal: F1 >= 0.90.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from workers.slicer import detect_panels_gutter
except ImportError:
    from slicer import detect_panels_gutter


def calculate_iou(box_a: Dict[str, int], box_b: Dict[str, int]) -> float:
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    ax1, ay1, aw, ah = box_a["x"], box_a["y"], box_a["w"], box_a["h"]
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1, bw, bh = box_b["x"], box_b["y"], box_b["w"], box_b["h"]
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def evaluate_detection(gt_path: Path, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """Runs evaluation across all pages in ground truth file."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pages = gt_data["source_pages"]
    gt_panels = gt_data["panels"]

    gt_by_page: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pages}
    for p in gt_panels:
        page_name = p.get("page")
        if page_name in gt_by_page:
            gt_by_page[page_name].append(p)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    matched_count = 0
    per_page_results = []

    for page_name in pages:
        page_path = gt_path.parent / page_name
        page_gt = gt_by_page[page_name]

        detected = detect_panels_gutter(page_path)

        matched_gt = set()
        matched_det = set()
        page_ious = []

        for det_idx, det in enumerate(detected):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(page_gt):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_det.add(det_idx)
                page_ious.append(best_iou)
                total_iou += best_iou
                matched_count += 1

        tp = len(matched_det)
        fp = len(detected) - tp
        fn = len(page_gt) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        page_p = tp / len(detected) if len(detected) > 0 else 0.0
        page_r = tp / len(page_gt) if len(page_gt) > 0 else 0.0
        page_f1 = (2 * page_p * page_r) / (page_p + page_r) if (page_p + page_r) > 0 else 0.0
        mean_iou = sum(page_ious) / len(page_ious) if page_ious else 0.0

        per_page_results.append(
            {
                "page": page_name,
                "gt_count": len(page_gt),
                "detected_count": len(detected),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": page_p,
                "recall": page_r,
                "f1": page_f1,
                "mean_iou": mean_iou,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    overall_mean_iou = total_iou / matched_count if matched_count > 0 else 0.0

    return {
        "total_gt_panels": len(gt_panels),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": overall_mean_iou,
        "iou_threshold": iou_threshold,
        "passed_target": f1 >= 0.90,
        "pages": per_page_results,
    }


if __name__ == "__main__":
    gt_file = Path("data/eval/ground_truth.json")
    results = evaluate_detection(gt_file)
    print("=" * 60)
    print("PANEL DETECTION EVALUATION REPORT (Phase 1 Target: F1 >= 0.90)")
    print("=" * 60)
    print(f"Total Ground Truth Panels: {results['total_gt_panels']}")
    print(f"True Positives:            {results['total_tp']}")
    print(f"False Positives:           {results['total_fp']}")
    print(f"False Negatives:           {results['total_fn']}")
    print(f"Precision:                 {results['precision']:.4f}")
    print(f"Recall:                    {results['recall']:.4f}")
    print(f"F1 Score:                  {results['f1']:.4f}")
    print(f"Mean IoU:                  {results['mean_iou']:.4f}")
    print(f"Status:                    {'PASSED (F1 >= 0.90)' if results['passed_target'] else 'FAILED'}")
    print("=" * 60)
    for p in results["pages"]:
        print(
            f"  {p['page']}: GT={p['gt_count']}, Det={p['detected_count']}, "
            f"P={p['precision']:.2f}, R={p['recall']:.2f}, F1={p['f1']:.2f}, IoU={p['mean_iou']:.4f}"
        )
