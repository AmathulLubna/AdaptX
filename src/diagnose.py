"""Stage 2 -- failure diagnosis: an evidence vector over candidate causes.

WHAT MAKES THIS A DIAGNOSIS AND NOT A CLASSIFIER
------------------------------------------------
No model is trained to predict the cause, and the ground-truth label is never
read (enforced by `tests/test_no_leakage.py`). Each cause is scored by a
statistic that is a *direct measurement of its mechanism*:

  cause                signature statistic
  -------------------  -------------------------------------------------------
  environment_shift    mean standardised divergence across ALL features
  feature_instability  CONCENTRATION of that divergence in a few features
  input_corruption     non-finite rate + heavy-tail outlier rate + variance
                       inflation
  class_imbalance      total-variation distance between old and new label priors
  overfitting          train/validation gap WITH no input-space movement

Raw statistics are squashed by `1 - exp(-s / scale)`, a monotone bijection from
[0, inf) to [0, 1). Monotonicity is the key property: a larger perturbation can
only raise the confidence, never lower it, so an unseen drift magnitude of 2.0
behaves like a stronger version of 1.0 rather than falling off a tuned cliff.

DISAMBIGUATION
--------------
Environment shift and feature instability share the same raw divergence, so they
must be separated by its *shape*. We use the normalised concentration ratio
(share of total divergence carried by the top-2 features against the share a
uniform spread would give). Feature instability is then gated by high
concentration, and environment shift is damped by it. Without this gate the two
causes are statistically indistinguishable and the diagnosis is a coin flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import Config
from .detect import DetectionReport
from .scenarios import CAUSES, Dataset


@dataclass
class Diagnosis:
    """Confidence in [0, 1] per cause, plus the raw evidence behind it."""

    confidences: Dict[str, float]
    evidence: Dict[str, float] = field(default_factory=dict)

    @property
    def primary(self) -> str:
        """Highest-confidence cause. Ties break by the canonical CAUSES order."""
        return max(CAUSES, key=lambda c: (self.confidences.get(c, 0.0), -CAUSES.index(c)))

    def as_dict(self) -> dict:
        """Serialise confidences, evidence and the leading hypothesis."""
        return {
            "confidences": {k: round(v, 4) for k, v in self.confidences.items()},
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
            "primary": self.primary,
        }


def _squash(statistic: float, scale: float) -> float:
    """Monotone [0, inf) -> [0, 1) map with `scale` as the half-saturation point."""
    return float(1.0 - np.exp(-max(statistic, 0.0) / max(scale, 1e-9)))


def _finite_stats(X: np.ndarray) -> "Tuple[np.ndarray, np.ndarray, float]":
    """Per-feature mean and std over finite entries, plus the non-finite rate.

    Computed on finite entries only so that a corrupted pool does not poison the
    shift statistic -- corruption is measured separately, by its own signature.
    """
    Z = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(Z)
    rate = float(bad.mean()) if Z.size else 0.0
    masked = np.where(bad, np.nan, Z)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(masked, axis=0)
        std = np.nanstd(masked, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0)
    return mean, std, rate


def feature_divergence(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Per-feature standardised mean shift |mu_new - mu_old| / sigma_old.

    Standardising by the ORIGINAL scale makes the statistic unit-free and
    comparable across features, which is what lets us talk about the shift being
    "concentrated" in some of them.
    """
    ref_mean, ref_std, _ = _finite_stats(reference)
    cur_mean, _, _ = _finite_stats(current)
    denom = np.maximum(ref_std, 1e-6)
    return np.abs(cur_mean - ref_mean) / denom


def concentration_ratio(divergence: np.ndarray, top_k: int = 2) -> float:
    """How much of the total divergence sits in the top-k features, normalised.

    Returns 0 when the shift is perfectly uniform and approaches 1 when a single
    feature carries everything. Normalising by `top_k / d` (the uniform share)
    removes the dependence on feature count, so the same threshold works whether
    the benchmark hands us 12 features or 200.
    """
    total = float(divergence.sum())
    d = divergence.shape[0]
    if total <= 1e-9 or d <= top_k:
        return 0.0
    top = float(np.sort(divergence)[-top_k:].sum())
    uniform_share = top_k / d
    observed_share = top / total
    return float(np.clip((observed_share - uniform_share) / (1.0 - uniform_share), 0.0, 1.0))


def outlier_rate(reference: np.ndarray, current: np.ndarray, z: float = 4.0) -> float:
    """Fraction of finite entries beyond `z` original standard deviations.

    A pure translation moves the whole cloud but barely changes the count of
    extreme points; spike noise creates them. So this separates corruption from
    shift even when both move the mean.
    """
    ref_mean, ref_std, _ = _finite_stats(reference)
    Z = np.asarray(current, dtype=np.float64)
    finite = np.isfinite(Z)
    if not finite.any():
        return 1.0
    scores = np.zeros_like(Z)
    np.divide(Z - ref_mean, np.maximum(ref_std, 1e-6), out=scores, where=finite)
    return float(np.mean(np.abs(scores[finite]) > z))


def heavy_tail_rate(current: np.ndarray, z: float = 5.0) -> float:
    """Fraction of entries that are extreme RELATIVE TO THEIR OWN DISTRIBUTION.

    This is the statistic that separates genuine corruption from a feature that
    has merely been rescaled, and it replaced an absolute-z-score version that
    confounded the two badly.

    The distinction: a feature multiplied by 2.3 is still Gaussian, so measured
    against its OWN median and MAD it has no unusual tail -- but measured against
    the ORIGINAL scale it looks full of outliers. Spike noise, by contrast, is
    heavy-tailed with respect to its own distribution, because the spikes are not
    part of the process that generated the bulk.

    Median and MAD (scaled by 1.4826 to be consistent for the normal) are used
    rather than mean and standard deviation precisely because the spikes we are
    hunting would otherwise inflate the very scale we measure them against.
    For clean Gaussian data the expected rate at z=5 is ~6e-7, i.e. zero.
    """
    Z = np.asarray(current, dtype=np.float64)
    finite = np.isfinite(Z)
    if not finite.any():
        return 1.0
    masked = np.where(finite, Z, np.nan)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(masked, axis=0)
        mad = np.nanmedian(np.abs(masked - median), axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    scale = np.maximum(np.where(np.isfinite(mad), mad, 1.0) * 1.4826, 1e-6)
    scores = np.zeros_like(Z)
    np.divide(Z - median, scale, out=scores, where=finite)
    return float(np.mean(np.abs(scores[finite]) > z))


def variance_inflation(reference: np.ndarray, current: np.ndarray) -> float:
    """Mean relative growth in per-feature spread; 0 if the spread shrank."""
    _, ref_std, _ = _finite_stats(reference)
    _, cur_std, _ = _finite_stats(current)
    ratio = cur_std / np.maximum(ref_std, 1e-6)
    return float(np.mean(np.maximum(ratio - 1.0, 0.0)))


def prior_shift(reference_labels: np.ndarray, current_labels: np.ndarray,
                n_classes: int) -> float:
    """Total-variation distance between old and new class priors, in [0, 1]."""
    def prior(labels: np.ndarray) -> np.ndarray:
        """Empirical class distribution, uniform when no labels are present."""
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
        total = counts.sum()
        return counts / total if total else np.full(n_classes, 1.0 / n_classes)

    return float(0.5 * np.abs(prior(reference_labels) - prior(current_labels)).sum())


def diagnose(train: Dataset, pool: Dataset, detection: DetectionReport,
             cfg: Config) -> Diagnosis:
    """Compute the evidence vector. Ground truth is never consulted."""
    divergence = feature_divergence(train.X, pool.X)
    mean_divergence = float(divergence.mean())
    concentration = concentration_ratio(divergence)
    _, _, missing = _finite_stats(pool.X)
    spikes = outlier_rate(train.X, pool.X)
    tails = heavy_tail_rate(pool.X)
    inflation = variance_inflation(train.X, pool.X)
    priors = prior_shift(train.y, pool.y, cfg.data.n_classes)

    # Weighting reflects each term's SPECIFICITY to corruption, not its size.
    # Missing values are unambiguous (nothing else creates them) and heavy tails
    # are nearly so, hence the large weights. Variance inflation is the weakest
    # evidence -- a rescaled feature inflates variance without any corruption --
    # so it contributes only a small confirmatory nudge.
    corruption_stat = 2.0 * missing + 1.5 * tails + 0.08 * inflation

    # Environment shift: broad movement. Damped when the movement is concentrated
    # (that is feature instability) or when it is explained by corruption noise.
    env_raw = mean_divergence * (1.0 - 0.75 * concentration)
    env = _squash(env_raw, cfg.diagnosis.shift_scale)
    env *= float(np.clip(1.0 - 1.5 * corruption_stat, 0.15, 1.0))

    # Feature instability: gated on concentration AND on the WORST single
    # feature. The product form means both conditions must hold -- a concentrated
    # but tiny shift and a large but uniform shift are both correctly rejected.
    # Max divergence is the right magnitude term here (not the mean): the claim
    # being tested is "at least one feature moved a lot", and averaging over the
    # untouched features dilutes exactly the signal we want.
    max_divergence = float(divergence.max()) if divergence.size else 0.0
    feature = _squash(concentration, cfg.diagnosis.concentration_scale) * _squash(
        max_divergence, cfg.diagnosis.shift_scale)

    corruption = _squash(corruption_stat, cfg.diagnosis.corruption_scale)

    # Class imbalance: a label-prior move that is NOT a side effect of the inputs
    # moving. Under covariate shift with a fixed labelling function, P(y) shifts
    # automatically as P(x) shifts -- so an undamped prior statistic fires on
    # every shift scenario and makes imbalance indistinguishable from the rest.
    # Damping by the covariate divergence isolates genuine prior shift, where
    # P(x|y) is untouched and only the mixing proportions changed.
    imbalance = _squash(priors, cfg.diagnosis.imbalance_scale)
    imbalance *= float(np.clip(1.0 - 1.6 * mean_divergence, 0.05, 1.0))

    # Overfitting: a train/validation gap that CANNOT be blamed on the inputs
    # moving. The damping factor is the discriminating half of the test -- a
    # large gap under a large shift is shift-induced, not memorisation.
    # The damping is aggressive (2.5x) because the gap statistic is the least
    # specific of the five: a shifted environment also widens train-vs-unseen
    # performance, so without strong damping overfitting wins on shift scenarios.
    # Corruption is damped out for the same reason -- noisy inputs inflate the
    # gap without any memorisation having occurred.
    overfit_raw = max(detection.generalization_gap, 0.0)
    overfit = _squash(overfit_raw, cfg.diagnosis.overfit_scale)
    overfit *= float(np.clip(1.0 - 2.5 * mean_divergence, 0.02, 1.0))
    overfit *= float(np.clip(1.0 - 2.0 * corruption_stat, 0.02, 1.0))

    confidences = {
        "environment_shift": round(float(env), 4),
        "feature_instability": round(float(feature), 4),
        "input_corruption": round(float(corruption), 4),
        "class_imbalance": round(float(imbalance), 4),
        "overfitting": round(float(overfit), 4),
    }
    evidence = {
        "mean_feature_divergence": mean_divergence,
        "divergence_concentration": concentration,
        "missing_rate": missing,
        "outlier_rate": spikes,
        "heavy_tail_rate": tails,
        "variance_inflation": inflation,
        "label_prior_shift": priors,
        "generalization_gap": detection.generalization_gap,
        "max_feature_divergence": float(divergence.max()) if divergence.size else 0.0,
    }
    return Diagnosis(confidences, evidence)
