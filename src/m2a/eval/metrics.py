"""Quality metrics for evaluation."""
import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _calculate_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0.0:
        return 0.0
    return interArea / denom


def panel_detection_f1(
    predicted_boxes: List[Tuple[float, float, float, float]],
    gt_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> float:
    """Calculates F1 score for panel detection."""
    if not predicted_boxes and not gt_boxes:
        return 1.0
    if not predicted_boxes or not gt_boxes:
        return 0.0

    matched_gt = set()
    true_positives = 0

    for pred in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = _calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    precision = true_positives / len(predicted_boxes)
    recall = true_positives / len(gt_boxes)
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def character_error_rate(predicted_text: str, gt_text: str) -> float:
    """Calculates Character Error Rate (CER) using dynamic programming Levenshtein distance."""
    if not gt_text:
        return 0.0 if not predicted_text else 1.0

    m, n = len(predicted_text), len(gt_text)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if predicted_text[i - 1] == gt_text[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n] / float(n)


def speaker_attribution_accuracy(predicted: dict, gt: dict) -> float:
    """Calculates accuracy of assigning speech balloons to characters."""
    if not gt:
        return 1.0
    correct = sum(1 for k, v in gt.items() if predicted.get(k) == v)
    return correct / float(len(gt))


def identity_similarity(source_image: np.ndarray, frame_image: np.ndarray) -> float:
    """Calculates normalized color-histogram cosine similarity between source and frame."""
    if source_image.size == 0 or frame_image.size == 0:
        return 0.0

    def _hist_vec(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            h, _ = np.histogram(arr, bins=64, range=(0, 256))
        else:
            chans = [np.histogram(arr[:, :, c], bins=32, range=(0, 256))[0] for c in range(min(3, arr.shape[2]))]
            h = np.concatenate(chans)
        v = h.astype(np.float32)
        return v / max(1e-6, float(np.linalg.norm(v)))

    v1 = _hist_vec(source_image)
    v2 = _hist_vec(frame_image)
    return float(np.clip(np.dot(v1, v2), 0.0, 1.0))


def temporal_stability(frames: List[np.ndarray]) -> float:
    """Calculates temporal stability (1.0 - flicker variance) across video frames."""
    if len(frames) < 2:
        return 1.0
    grays = [f.mean(axis=2).astype(np.float32) / 255.0 if f.ndim == 3 else f.astype(np.float32) / 255.0 for f in frames]
    deltas = [float(np.mean(np.abs(grays[i + 1] - grays[i]))) for i in range(len(grays) - 1)]
    flicker_std = float(np.std(deltas))
    return float(np.clip(1.0 - flicker_std * 5.0, 0.0, 1.0))
