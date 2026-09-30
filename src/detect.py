"""Stage 1 -- failure detection.

PRINCIPLE
---------
"Different" is not "broken". A model whose accuracy moves by less than the
sampling noise of the estimate has not failed; declaring a failure there would
trigger a needless repair, burn compute, and risk making a healthy model worse.
The control scenario in the benchmark exists to punish exactly that.

DECISION RULE
-------------
A failure is declared when the OOD accuracy drop exceeds BOTH
  * `noise_sigmas` times the pooled binomial standard error, AND
  * a small absolute floor (`min_absolute_drop`),
or when a secondary symptom is severe (calibration blow-up, fairness collapse,
or a train/validation gap indicating memorisation with no shift at all).

The two-part accuracy test is deliberately conservative: the sigma term alone
would fire on huge evaluation sets for operationally irrelevant drops, and the
absolute floor alone would not adapt to sample size. Requiring both gives a test
that is neither sample-size-naive nor trigger-happy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import Config
from .metrics import (accuracy, balanced_accuracy, expected_calibration_error,
                      fairness_gap, mean_over_levels, prediction_stability,
                      sampling_sigma)
from .models import FittedModel
from .scenarios import Scenario


@dataclass
class DetectionReport:
    """Symptom vector plus the binary decision and the reason for it."""

    failure_detected: bool
    reason: str
    train_accuracy: float
    val_accuracy: float
    ood_accuracy: float
    ood_balanced_accuracy: float
    accuracy_drop: float
    balanced_accuracy_drop: float
    noise_threshold: float
    generalization_gap: float
    calibration_error: float
    fairness_gap: float
    stability: float
    per_level_accuracy: Dict[float, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialise the symptom vector for the JSON repair report."""
        return {
            "failure_detected": self.failure_detected,
            "reason": self.reason,
            "train_accuracy": round(self.train_accuracy, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "ood_accuracy": round(self.ood_accuracy, 4),
            "ood_balanced_accuracy": round(self.ood_balanced_accuracy, 4),
            "accuracy_drop": round(self.accuracy_drop, 4),
            "balanced_accuracy_drop": round(self.balanced_accuracy_drop, 4),
            "noise_threshold": round(self.noise_threshold, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "calibration_error": round(self.calibration_error, 4),
            "fairness_gap": round(self.fairness_gap, 4),
            "stability": round(self.stability, 4),
            "per_level_accuracy": {str(k): round(v, 4) for k, v in self.per_level_accuracy.items()},
        }


def evaluate_on_levels(model: FittedModel, scenario: Scenario,
                       levels: "tuple[float, ...]") -> Dict[float, float]:
    """Accuracy of `model` at each requested perturbation level."""
    out: Dict[float, float] = {}
    for level in levels:
        data = scenario.ood.get(level)
        if data is None:
            continue
        out[level] = accuracy(data.y, model.predict(data.X))
    return out


def detect_failure(model: FittedModel, scenario: Scenario, cfg: Config) -> DetectionReport:
    """Run Stage 1 and return the full symptom vector.

    Only VALIDATION drift levels are inspected here. The hidden test levels exist
    to check that the repair generalises; using them for detection would be
    leakage from the evaluation set into the decision process.
    """
    train_acc = accuracy(scenario.train.y, model.predict(scenario.train.X))
    val_acc = accuracy(scenario.val.y, model.predict(scenario.val.X))

    per_level = evaluate_on_levels(model, scenario, cfg.data.val_levels)
    ood_acc = mean_over_levels(per_level)

    # Balanced accuracy on BOTH sides, because a prior shift can leave overall
    # accuracy almost untouched while a minority class collapses: the majority
    # class grows, and predicting it more often compensates in the raw average.
    # Detecting that failure requires per-class recall, not accuracy.
    val_balanced = balanced_accuracy(scenario.val.y, model.predict(scenario.val.X))
    ood_balanced_levels = [
        balanced_accuracy(scenario.ood[lv].y, model.predict(scenario.ood[lv].X))
        for lv in cfg.data.val_levels if lv in scenario.ood
    ]
    ood_balanced_mean = float(np.mean(ood_balanced_levels)) if ood_balanced_levels else val_balanced

    nominal = scenario.ood.get(1.0)
    if nominal is None:
        nominal = scenario.val
    predictions = model.predict(nominal.X)
    balanced = balanced_accuracy(nominal.y, predictions)
    calibration = expected_calibration_error(nominal.y, model.predict_proba(nominal.X),
                                             cfg.detection.calibration_bins)
    fairness = fairness_gap(nominal.y, predictions, nominal.groups)
    stability = prediction_stability(list(per_level.values()))

    # Pooled standard error of the difference between two accuracy estimates.
    n_eval = int(sum(len(scenario.ood[lv]) for lv in per_level)) or len(scenario.val)
    sigma = float(np.hypot(sampling_sigma(val_acc, len(scenario.val)),
                           sampling_sigma(ood_acc, n_eval)))
    threshold = max(cfg.detection.noise_sigmas * sigma, cfg.detection.min_absolute_drop)

    drop = val_acc - ood_acc
    gap = train_acc - val_acc

    # The memorisation test gets its own adaptive floor. A small training set
    # produces a wide train/validation gap purely from sampling, so a fixed
    # constant here would flag a perfectly healthy model trained on little data
    # -- exactly the false positive the control scenario is designed to catch.
    gap_sigma = float(np.hypot(sampling_sigma(train_acc, len(scenario.train)),
                               sampling_sigma(val_acc, len(scenario.val))))
    gap_threshold = max(cfg.detection.overfit_gap_threshold,
                        cfg.detection.noise_sigmas * gap_sigma)

    balanced_drop = val_balanced - ood_balanced_mean

    failed, reason = _decide(drop, threshold, balanced_drop, gap, gap_threshold,
                             calibration, fairness, cfg)
    return DetectionReport(failed, reason, train_acc, val_acc, ood_acc, balanced,
                           drop, balanced_drop, threshold, gap, calibration,
                           fairness, stability, per_level)


def _decide(drop: float, threshold: float, balanced_drop: float, gap: float,
            gap_threshold: float, calibration: float, fairness: float,
            cfg: Config) -> "tuple[bool, str]":
    """Priority-ordered decision. First matching rule wins and names itself.

    Ordering matters for explainability: the reason string is what the jury and
    the repair-selection stage read, so the most specific symptom must win over
    the most generic one.
    """
    if drop > threshold:
        return True, "ood_accuracy_drop_exceeds_sampling_noise"
    # Checked against the same noise floor, but on per-class recall. This is the
    # only rule that can catch a pure prior shift, where overall accuracy is
    # preserved by the majority class while a minority class stops working.
    if balanced_drop > threshold:
        return True, "minority_class_recall_collapse"
    if gap > gap_threshold:
        return True, "train_validation_gap_indicates_memorisation"
    if fairness > cfg.repair.max_fairness_gap:
        return True, "subpopulation_accuracy_gap_exceeds_cap"
    if calibration > 0.25:
        return True, "severe_miscalibration"
    return False, "no_meaningful_degradation"
