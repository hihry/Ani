"""Quality metrics for evaluation."""
import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def _calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def panel_detection_f1(predicted_boxes: List[Tuple[float, float, float, float]], 
                       gt_boxes: List[Tuple[float, float, float, float]], 
                       iou_threshold: float = 0.5) -> float:
    """Calculates F1 score for panel detection."""
    logger.info(f"Computing panel detection F1 with {len(predicted_boxes)} preds, {len(gt_boxes)} GTs")
    
    if not predicted_boxes and not gt_boxes:
        return 1.0
    if not predicted_boxes or not gt_boxes:
        return 0.0
        
    matched_gt = set()
    true_positives = 0
    
    for pred in predicted_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = _calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            
    precision = true_positives / len(predicted_boxes)
    recall = true_positives / len(gt_boxes)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def character_error_rate(predicted_text: str, gt_text: str) -> float:
    """Calculates Character Error Rate (CER) between predicted and ground truth text."""
    # TODO: Implement actual edit distance / Levenshtein
    logger.info("Computing character error rate")
    return 0.0

def speaker_attribution_accuracy(predicted: dict, gt: dict) -> float:
    """Calculates accuracy of assigning speech balloons to characters."""
    logger.info("Computing speaker attribution accuracy")
    return 0.0

def identity_similarity(source_image: np.ndarray, frame_image: np.ndarray) -> float:
    """Calculates character identity similarity (e.g. using CLIP)."""
    # TODO: Implement actual CLIP embedding similarity
    logger.info("Computing identity similarity (CLIP stub)")
    return 1.0

def temporal_stability(frames: List[np.ndarray]) -> float:
    """Calculates temporal stability (flicker/jitter) across video frames."""
    # TODO: Implement actual temporal consistency metric (e.g., optical flow variance)
    logger.info("Computing temporal stability")
    return 1.0
