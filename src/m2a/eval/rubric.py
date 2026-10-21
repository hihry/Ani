"""Human rubric scoring, Krippendorff's ordinal alpha, and Bradley-Terry model ranking (`src/m2a/eval/rubric.py`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class RubricRating:
    anon_clip_id: str
    rater_id: str
    motion_plausibility: int  # 1..5
    character_fidelity: int   # 1..5
    temporal_stability: int   # 1..5
    prompt_adherence: int     # 1..5
    style_preservation: int   # 1..5
    overall_pass: bool

    @property
    def quality_score(self) -> float:
        return compute_quality_score(
            motion=self.motion_plausibility,
            fidelity=self.character_fidelity,
            stability=self.temporal_stability,
            prompt=self.prompt_adherence,
            style=self.style_preservation,
        )


def compute_quality_score(
    *,
    motion: float,
    fidelity: float,
    stability: float,
    prompt: float,
    style: float,
) -> float:
    """Computes composite quality score Q in [0, 1] from 1..5 rubric ratings (§6.3)."""
    weighted = (
        0.25 * float(motion)
        + 0.25 * float(fidelity)
        + 0.20 * float(stability)
        + 0.15 * float(prompt)
        + 0.15 * float(style)
    )
    q = (weighted - 1.0) / 4.0
    return round(float(np.clip(q, 0.0, 1.0)), 4)


def krippendorff_alpha_ordinal(
    ratings_by_unit: Sequence[Sequence[Optional[float]]],
) -> float:
    """Computes Krippendorff's alpha for ordinal/interval ratings across units (items x raters).

    `ratings_by_unit` is a list of items, where each item is a sequence of ratings from raters
    (None or NaN indicates missing rating).
    Returns 1.0 when raters agree perfectly, ~0.0 when ratings are statistically independent.
    """
    valid_units: List[List[float]] = []
    all_values: List[float] = []
    for unit in ratings_by_unit:
        vals = [float(v) for v in unit if v is not None and not np.isnan(float(v))]
        if len(vals) >= 2:
            valid_units.append(vals)
            all_values.extend(vals)

    if not valid_units or len(all_values) < 2:
        return 0.0

    # Check if all ratings across all units are identical -> perfect agreement (1.0)
    if float(np.std(all_values)) < 1e-9:
        for u in valid_units:
            if float(np.std(u)) > 1e-9:
                break
        else:
            return 1.0

    # Observed disagreement Do
    do_num = 0.0
    n_total = len(all_values)
    for vals in valid_units:
        m_u = len(vals)
        pair_sum = 0.0
        for i in range(m_u):
            for j in range(i + 1, m_u):
                pair_sum += (vals[i] - vals[j]) ** 2
        do_num += (2.0 * pair_sum) / float(m_u - 1)
    d_o = do_num / float(n_total)

    # Expected disagreement De across all pairs of values
    arr = np.asarray(all_values, dtype=np.float64)
    diffs = arr[:, None] - arr[None, :]
    d_e = float(np.sum(diffs ** 2)) / float(n_total * (n_total - 1))
    if d_e <= 1e-12:
        return 1.0 if d_o <= 1e-12 else 0.0

    alpha = 1.0 - (d_o / d_e)
    return round(float(np.clip(alpha, -1.0, 1.0)), 4)


def bradley_terry_ranks(
    panel_model_scores: Dict[str, Dict[str, float]],
    iterations: int = 60,
) -> Dict[str, float]:
    """Computes Bradley-Terry model strength parameters p_i (summing to 1.0) from per-panel quality scores."""
    models_set: set[str] = set()
    for m_scores in panel_model_scores.values():
        models_set.update(m_scores.keys())
    models = sorted(models_set)
    if not models:
        return {}
    if len(models) == 1:
        return {models[0]: 1.0}

    idx_of = {m: i for i, m in enumerate(models)}
    k = len(models)
    # Build pairwise win matrix W[i, j] (with 0.5 for ties and Laplace smoothing 0.05)
    w_mat = np.full((k, k), 0.05, dtype=np.float64)
    np.fill_diagonal(w_mat, 0.0)

    for m_scores in panel_model_scores.values():
        present = list(m_scores.items())
        for a_idx in range(len(present)):
            m_a, s_a = present[a_idx]
            i = idx_of[m_a]
            for b_idx in range(a_idx + 1, len(present)):
                m_b, s_b = present[b_idx]
                j = idx_of[m_b]
                if s_a > s_b + 1e-6:
                    w_mat[i, j] += 1.0
                elif s_b > s_a + 1e-6:
                    w_mat[j, i] += 1.0
                else:
                    w_mat[i, j] += 0.5
                    w_mat[j, i] += 0.5

    p = np.ones(k, dtype=np.float64) / float(k)
    for _ in range(max(1, iterations)):
        p_new = np.zeros_like(p)
        for i in range(k):
            w_i = float(np.sum(w_mat[i, :]))
            denom = 0.0
            for j in range(k):
                if i == j:
                    continue
                n_ij = w_mat[i, j] + w_mat[j, i]
                if n_ij > 0 and (p[i] + p[j]) > 1e-12:
                    denom += n_ij / (p[i] + p[j])
            p_new[i] = w_i / max(1e-9, denom)
        total = float(np.sum(p_new))
        p = p_new / max(1e-9, total)

    return {m: round(float(p[idx_of[m]]), 4) for m in models}
