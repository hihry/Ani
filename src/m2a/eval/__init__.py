"""Phase 3 bake-off runner, human rubric aggregation, and QC threshold calibration package."""
from m2a.eval.bakeoff import BakeoffRecord, run_bakeoff
from m2a.eval.calibrate_qc import calibrate_thresholds, compute_roc_auc_youden
from m2a.eval.rubric import (
    RubricRating,
    bradley_terry_ranks,
    compute_quality_score,
    krippendorff_alpha_ordinal,
)

__all__ = [
    "BakeoffRecord",
    "run_bakeoff",
    "calibrate_thresholds",
    "compute_roc_auc_youden",
    "RubricRating",
    "bradley_terry_ranks",
    "compute_quality_score",
    "krippendorff_alpha_ordinal",
]
