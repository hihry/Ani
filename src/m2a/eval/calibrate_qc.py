"""QC Threshold Calibration via ROC/AUC, Youden's J, and 5-Fold Cross-Validation (`src/m2a/eval/calibrate_qc.py`).

Implements Phase 3 §7:
  1. `compute_roc_auc_youden(scores, labels, higher_is_better)` computes ROC curve, AUC,
     and the Youden-optimal threshold `tau* = argmax(TPR - FPR)`.
  2. `calibrate_thresholds(records, route="i2v", min_precision=0.85, n_folds=5)` fits per-metric
     thresholds, tightens conjunctive precision to `>= min_precision`, and reports 5-fold CV
     agreement, precision, recall, and F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml


def compute_roc_auc_youden(
    scores: Sequence[float],
    labels: Sequence[bool],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Computes ROC points, AUC, and Youden's J optimal threshold for a single metric."""
    s_arr = np.asarray(scores, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=bool)
    if len(s_arr) == 0 or np.all(y_arr) or not np.any(y_arr):
        default_tau = float(np.median(s_arr)) if len(s_arr) > 0 else 0.5
        return {
            "auc": 0.5,
            "threshold": round(default_tau, 4),
            "youden_j": 0.0,
            "tpr": 1.0,
            "fpr": 0.0,
        }

    # Candidate thresholds at midpoints between unique sorted values plus endpoints
    uniq = np.unique(s_arr)
    if len(uniq) > 1:
        mids = 0.5 * (uniq[:-1] + uniq[1:])
        candidates = np.concatenate([[uniq[0] - 1e-4], mids, [uniq[-1] + 1e-4]])
    else:
        candidates = np.array([uniq[0] - 1e-4, uniq[0], uniq[0] + 1e-4])

    n_pos = float(np.sum(y_arr))
    n_neg = float(np.sum(~y_arr))

    tprs: List[float] = []
    fprs: List[float] = []
    best_j = -1.0
    best_tau = float(candidates[0])
    best_tpr = 0.0
    best_fpr = 0.0

    for tau in candidates:
        preds = (s_arr >= tau) if higher_is_better else (s_arr <= tau)
        tp = float(np.sum(preds & y_arr))
        fp = float(np.sum(preds & (~y_arr)))
        tpr = tp / max(1.0, n_pos)
        fpr = fp / max(1.0, n_neg)
        tprs.append(tpr)
        fprs.append(fpr)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_tau = float(tau)
            best_tpr = tpr
            best_fpr = fpr

    # Sort by FPR ascending to integrate AUC via trapezoidal rule
    order = np.argsort(fprs)
    sorted_fpr = np.asarray(fprs)[order]
    sorted_tpr = np.asarray(tprs)[order]
    auc = float(np.trapezoid(sorted_tpr, sorted_fpr)) if hasattr(np, "trapezoid") else float(np.trapz(sorted_tpr, sorted_fpr))
    return {
        "auc": round(float(np.clip(auc, 0.0, 1.0)), 4),
        "threshold": round(best_tau, 4),
        "youden_j": round(float(best_j), 4),
        "tpr": round(float(best_tpr), 4),
        "fpr": round(float(best_fpr), 4),
    }


def _evaluate_conjunctive(
    samples: Sequence[Dict[str, Any]],
    thresholds: Dict[str, Any],
) -> Dict[str, float]:
    if not samples:
        return {"agreement": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = fp = tn = fn = 0
    for item in samples:
        qc = item.get("qc_scores") or item
        label = bool(item.get("human_pass", item.get("overall_pass", False)))
        pred = (
            float(qc.get("identity_similarity", 0.0)) >= float(thresholds["min_identity_similarity"])
            and float(qc.get("flicker_score", 0.0)) <= float(thresholds["max_flicker_score"])
            and float(qc.get("motion_score", 0.0)) >= float(thresholds["min_motion_score"])
            and float(qc.get("motion_score", 0.0)) <= float(thresholds["max_motion_score"])
            and float(qc.get("warp_error", 0.0)) <= float(thresholds["max_warp_error"])
            and float(qc.get("first_frame_fidelity", 1.0)) >= float(thresholds["min_first_frame_fidelity"])
            and float(qc.get("style_drift", 0.0)) <= float(thresholds["max_style_drift"])
            and int(qc.get("new_text_count", 0)) == 0
        )
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1

    total = max(1, tp + fp + tn + fn)
    agreement = (tp + tn) / float(total)
    precision = tp / float(max(1, tp + fp)) if (tp + fp) > 0 else 1.0
    recall = tp / float(max(1, tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall) / max(1e-9, precision + recall)
    return {
        "agreement": round(agreement, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _fit_fold_thresholds(
    train_samples: Sequence[Dict[str, Any]],
    min_precision: float = 0.85,
) -> Dict[str, Any]:
    labels = [bool(s.get("human_pass", s.get("overall_pass", False))) for s in train_samples]

    def _vals(key: str, default: float = 0.0) -> List[float]:
        return [float((s.get("qc_scores") or s).get(key, default)) for s in train_samples]

    id_roc = compute_roc_auc_youden(_vals("identity_similarity", 0.9), labels, higher_is_better=True)
    flk_roc = compute_roc_auc_youden(_vals("flicker_score", 0.05), labels, higher_is_better=False)
    mot_roc = compute_roc_auc_youden(_vals("motion_score", 0.1), labels, higher_is_better=True)
    wrp_roc = compute_roc_auc_youden(_vals("warp_error", 0.04), labels, higher_is_better=False)
    fst_roc = compute_roc_auc_youden(_vals("first_frame_fidelity", 0.85), labels, higher_is_better=True)
    sty_roc = compute_roc_auc_youden(_vals("style_drift", 0.1), labels, higher_is_better=False)

    thresholds: Dict[str, Any] = {
        "min_identity_similarity": round(float(np.clip(id_roc["threshold"], 0.65, 0.92)), 4),
        "max_flicker_score": round(float(np.clip(flk_roc["threshold"], 0.06, 0.20)), 4),
        "min_motion_score": round(float(np.clip(mot_roc["threshold"], 0.008, 0.05)), 4),
        "max_motion_score": 0.98,
        "max_warp_error": round(float(np.clip(wrp_roc["threshold"], 0.05, 0.15)), 4),
        "min_first_frame_fidelity": round(float(np.clip(fst_roc["threshold"], 0.70, 0.90)), 4),
        "max_style_drift": round(float(np.clip(sty_roc["threshold"], 0.15, 0.32)), 4),
        "max_retries": 2,
    }

    # Tighten thresholds until conjunctive precision >= min_precision
    for _ in range(25):
        stats = _evaluate_conjunctive(train_samples, thresholds)
        if stats["precision"] >= min_precision:
            break
        thresholds["min_identity_similarity"] = round(min(0.95, thresholds["min_identity_similarity"] + 0.015), 4)
        thresholds["max_warp_error"] = round(max(0.04, thresholds["max_warp_error"] - 0.008), 4)
        thresholds["max_flicker_score"] = round(max(0.05, thresholds["max_flicker_score"] - 0.008), 4)

    return {
        "thresholds": thresholds,
        "roc": {
            "identity_similarity": id_roc,
            "flicker_score": flk_roc,
            "motion_score": mot_roc,
            "warp_error": wrp_roc,
            "first_frame_fidelity": fst_roc,
            "style_drift": sty_roc,
        },
    }


def calibrate_thresholds(
    samples: Sequence[Dict[str, Any]],
    route: str = "i2v",
    min_precision: float = 0.85,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Calibrates QC thresholds with 5-fold cross-validation and returns calibrated config + metrics."""
    samples_list = list(samples)
    if not samples_list:
        raise ValueError("Cannot calibrate QC thresholds on empty sample list")

    fit_all = _fit_fold_thresholds(samples_list, min_precision=min_precision)
    in_sample_metrics = _evaluate_conjunctive(samples_list, fit_all["thresholds"])

    k = min(max(2, int(n_folds)), len(samples_list))
    cv_agreements: List[float] = []
    cv_precisions: List[float] = []
    cv_recalls: List[float] = []

    if len(samples_list) >= 4:
        indices = np.arange(len(samples_list))
        folds = np.array_split(indices, k)
        for fold_idx in range(k):
            val_idx = set(folds[fold_idx].tolist())
            train = [samples_list[i] for i in range(len(samples_list)) if i not in val_idx]
            val = [samples_list[i] for i in range(len(samples_list)) if i in val_idx]
            if not train or not val:
                continue
            fold_fit = _fit_fold_thresholds(train, min_precision=min_precision)
            fold_eval = _evaluate_conjunctive(val, fold_fit["thresholds"])
            cv_agreements.append(fold_eval["agreement"])
            cv_precisions.append(fold_eval["precision"])
            cv_recalls.append(fold_eval["recall"])

    return {
        "route": route,
        "thresholds": fit_all["thresholds"],
        "roc": fit_all["roc"],
        "in_sample": in_sample_metrics,
        "cv_agreement": round(float(np.mean(cv_agreements)) if cv_agreements else in_sample_metrics["agreement"], 4),
        "cv_precision": round(float(np.mean(cv_precisions)) if cv_precisions else in_sample_metrics["precision"], 4),
        "cv_recall": round(float(np.mean(cv_recalls)) if cv_recalls else in_sample_metrics["recall"], 4),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate QC thresholds from bake-off records & human ratings")
    parser.add_argument("--bakeoff", type=Path, required=True, help="Path to records.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output calibrated QC YAML path")
    parser.add_argument("--calibration-out", type=Path, default=None, help="Optional s_per_frame calibration YAML")
    args = parser.parse_args(argv)

    samples = [
        json.loads(line)
        for line in args.bakeoff.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    res = calibrate_thresholds(samples, route="i2v")
    out_doc = {
        "qc": {
            "identity_threshold": res["thresholds"]["min_identity_similarity"],
            "flicker_threshold": res["thresholds"]["max_flicker_score"],
            "max_new_text_lines": 0,
            "min_motion": res["thresholds"]["min_motion_score"],
            "max_motion": res["thresholds"]["max_motion_score"],
            "max_warp_error": res["thresholds"]["max_warp_error"],
            "min_first_frame_fidelity": res["thresholds"]["min_first_frame_fidelity"],
            "max_style_drift": res["thresholds"]["max_style_drift"],
            "require_text_check": True,
            "per_route": {
                "i2v": res["thresholds"],
            },
        },
        "calibration_metrics": {
            "cv_agreement": res["cv_agreement"],
            "cv_precision": res["cv_precision"],
            "cv_recall": res["cv_recall"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out_doc, sort_keys=False), encoding="utf-8")

    if args.calibration_out:
        spf_vals = [float(s.get("s_per_frame", 1.2)) for s in samples if s.get("route") == "i2v"]
        cal_doc = {
            "s_per_frame": round(float(np.median(spf_vals)) if spf_vals else 1.2, 4),
        }
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(yaml.safe_dump(cal_doc, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
