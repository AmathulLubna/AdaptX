"""Evaluation metrics: accuracy, calibration, fairness, stability, recovery.

Every metric here is a pure function of arrays -- no model, no config, no state.
That keeps them trivially testable and lets the auditor verify them in isolation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain top-1 accuracy."""
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall.

    Reported alongside plain accuracy because under prior shift a model can look
    healthy on accuracy while a minority class has collapsed entirely. The
    imbalance scenario is scored on this, not on raw accuracy.
    """
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0
    recalls = [float(np.mean(y_pred[y_true == c] == c)) for c in classes
               if np.any(y_true == c)]
    return float(np.mean(recalls)) if recalls else 0.0


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float:
    """Standard ECE: mean |confidence - accuracy| over equal-width bins.

    Confidence is the max predicted probability. Empty bins contribute nothing,
    so the estimator stays well defined on small evaluation sets.
    """
    if y_true.size == 0:
        return 0.0
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        weight = float(np.mean(in_bin))
        if weight > 0:
            error += weight * abs(float(np.mean(correct[in_bin]) - np.mean(confidence[in_bin])))
    return float(error)


def fairness_gap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Largest accuracy difference between any two sub-populations.

    The track requires "fair calibration across heterogeneous sub-populations".
    Max-gap (rather than variance) is used because it is the quantity a
    constraint can be stated on, and it is the worst case a deployer cares about.
    """
    tags = np.unique(groups)
    scores = [accuracy(y_true[groups == g], y_pred[groups == g]) for g in tags
              if np.sum(groups == g) > 0]
    if len(scores) < 2:
        return 0.0
    return float(max(scores) - min(scores))


def prediction_stability(scores: Sequence[float]) -> float:
    """Map dispersion across perturbation levels into a [0, 1] stability score.

    `1 / (1 + 4*std)` is monotone decreasing in the spread, bounded, and needs no
    reference scale. A model that scores identically at every drift level gets
    1.0; one whose accuracy swings by 0.25 std gets 0.5.
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size <= 1:
        return 1.0
    return float(1.0 / (1.0 + 4.0 * arr.std()))


def ood_recovery(failed: float, repaired: float, reference: float) -> float:
    """Fraction of the lost performance that the repair restored.

    recovery = (repaired - failed) / (reference - failed)

    Clipped to [0, 1]. Values above 1 mean the repair beat the full-retrain
    ceiling -- a real and reportable outcome (warm-start adaptation keeps the
    original model's knowledge, which a from-scratch retrain discards), but
    letting it exceed 1 would inflate the benchmark mean and misrepresent the
    system. `beat_reference` in the report records those cases explicitly
    instead. If the reference is not meaningfully above the failed model there is
    nothing to recover and the metric is undefined -- we return 1.0, since no
    loss means no shortfall.
    """
    denominator = reference - failed
    if denominator <= 1e-6:
        return 1.0
    return float(np.clip((repaired - failed) / denominator, 0.0, 1.0))


def sampling_sigma(accuracy_estimate: float, n: int) -> float:
    """Binomial standard error of an accuracy estimate.

    This is what makes detection adaptive rather than threshold-tuned: a 3-point
    accuracy drop is noise on 200 samples and a real failure on 5000, and this
    function is the only place that distinction is encoded.
    """
    if n <= 0:
        return 1.0
    p = float(np.clip(accuracy_estimate, 1e-6, 1 - 1e-6))
    return float(np.sqrt(p * (1 - p) / n))


def mean_over_levels(per_level: Dict[float, float], levels: "Iterable[float] | None" = None) -> float:
    """Average a per-level metric, optionally restricted to given levels."""
    keys = list(per_level) if levels is None else [lv for lv in levels if lv in per_level]
    if not keys:
        return 0.0
    return float(np.mean([per_level[k] for k in keys]))
