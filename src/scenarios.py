"""Synthetic multi-scenario benchmark with known-but-hidden failure causes.

Each scenario ships a model that was trained adequately on its original
distribution, then an environment that has moved. The `hidden_cause` field is the
evaluator's ground truth and is NEVER read by the diagnosis or repair code --
`tests/test_no_leakage.py` enforces that by static inspection.

DATA-GENERATING PROCESS
-----------------------
Features X ~ N(0, I) in R^d. Labels come from a fixed random linear teacher
applied to the *informative* subset of features and pushed through a softmax, so
the Bayes-optimal boundary is well defined and unchanged by covariate shift. This
matters: it means a covariate-shift scenario is genuinely repairable (the
relationship P(y|x) is intact, only P(x) moved), which is exactly the regime
PatchML claims to address.

A binary sub-population attribute `g` is derived from a non-informative feature,
so any fairness gap the model develops is an artefact of the repair rather than
of the label process -- that makes the fairness constraint meaningful.

PERTURBATION LEVELS
-------------------
Severity is a continuous multiplier, never a discrete switch. Validation uses
levels (0.8, 1.0, 1.2); the held-out test uses (0.6, 1.0, 1.5, 2.0). Because no
threshold anywhere in the system is fitted to a specific level, unseen
magnitudes degrade performance smoothly instead of breaking the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import Config

# Canonical cause labels. The diagnosis module emits confidences over exactly
# this vocabulary, so adding a cause is a one-line change in both places.
CAUSES: Tuple[str, ...] = (
    "environment_shift",
    "feature_instability",
    "input_corruption",
    "class_imbalance",
    "overfitting",
)


@dataclass
class Dataset:
    """A labelled split, with a sub-population attribute for fairness checks."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class Scenario:
    """One benchmark case handed to AdaptX.

    Attributes
    ----------
    train, val:
        Original in-distribution data the deployed model was fitted on.
    pool:
        Newly collected data from the *changed* environment. PatchML selects a
        subset of this.
    ood:
        Evaluation sets keyed by perturbation level.
    hidden_cause:
        Ground truth, for scoring only. Never passed to the doctor.
    """

    name: str
    hidden_cause: str
    train: Dataset
    val: Dataset
    pool: Dataset
    ood: Dict[float, Dataset]
    notes: str = ""
    overparameterized: bool = False
    metadata: dict = field(default_factory=dict)


def _teacher(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Fixed random linear teacher over the informative features only."""
    weights = np.zeros((cfg.data.n_features, cfg.data.n_classes))
    informative = slice(0, cfg.data.n_informative)
    weights[informative] = rng.normal(0.0, 1.1, size=(cfg.data.n_informative, cfg.data.n_classes))
    return weights


def _labels(X: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Softmax sampling gives a non-trivial Bayes error (~10-15%).

    Deterministic argmax labelling would make every scenario perfectly separable
    and the recovery metric degenerate, so a stochastic teacher is used instead.
    """
    logits = X @ weights
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    cumulative = probs.cumsum(axis=1)
    draws = rng.random((X.shape[0], 1))
    return (draws > cumulative).sum(axis=1).astype(np.int64)


def _groups(X: np.ndarray, cfg: Config) -> np.ndarray:
    """Sub-population tag from a NON-informative feature (fairness probe)."""
    probe = min(cfg.data.n_informative, cfg.data.n_features - 1)
    return (X[:, probe] > 0).astype(np.int64)


def _sample(n: int, cfg: Config, weights: np.ndarray, rng: np.random.Generator) -> Dataset:
    X = rng.normal(0.0, 1.0, size=(n, cfg.data.n_features))
    return Dataset(X, _labels(X, weights, rng), _groups(X, cfg))


# --------------------------------------------------------------------------
# Perturbation operators. Each maps (Dataset, level, rng) -> Dataset.
# They are pure functions so a Round-2 surprise perturbation is a new function
# plus one registry entry -- no change to the doctor.
#
# CRITICAL MODELLING DECISION: labels come from the LATENT (pre-perturbation)
# inputs, while the model observes the perturbed ones.
#
# Regenerating labels from the perturbed inputs -- which is the obvious thing to
# write -- makes the whole benchmark vacuous. P(y|x) stays intact, the shifted
# region is still correctly labelled by the same boundary, and a well-fit model
# generalises straight through it; we measured a "damaged" model holding 0.87
# accuracy at drift level 2.0, i.e. no failure to diagnose and nothing to repair.
#
# Labelling from the latent frame encodes the failure that actually happens in
# deployment: the OBSERVATION MAP changed (sensors recalibrated, a feature is now
# recorded in different units, measurements got noisy) while the underlying truth
# did not. The deployed model then reads displaced inputs through an unchanged
# decision boundary and is systematically wrong -- which is precisely the failure
# a small, well-chosen patch of newly-labelled data can correct.
# --------------------------------------------------------------------------


def _perturb_environment(data: Dataset, level: float,
                         direction: np.ndarray) -> Dataset:
    """Global covariate shift: the whole input cloud translates and dilates.

    `direction` carries unit-magnitude COMPONENTS (not unit norm), so the
    per-feature standardised divergence is ~1.1 sigma at level 1.0. Two earlier
    settings were measured and rejected: a unit-norm direction spreads the budget
    over d features (~0.17 each, no measurable degradation), and 0.6 sigma cost
    only 1.7 accuracy points -- below the detector's own noise floor, so the
    scenario was not a failure at all. At 1.1 sigma the shifted cloud sits
    largely outside the training support and the deployed boundary is genuinely
    displaced, which is the regime PatchML is built for.
    """
    X = (data.X + level * 1.1 * direction) * (1.0 + 0.2 * level)
    return Dataset(X, data.y.copy(), data.groups.copy())


def _perturb_feature(data: Dataset, level: float,
                     unstable: np.ndarray) -> Dataset:
    """A few individually important features move; the rest are untouched.

    This is the signature that separates feature instability from a global
    environment shift: the divergence is *concentrated*, not spread out.
    """
    X = data.X.copy()
    X[:, unstable] = X[:, unstable] * (1.0 + 1.3 * level) + 1.9 * level
    return Dataset(X, data.y.copy(), data.groups.copy())


def _perturb_corruption(data: Dataset, level: float,
                        rng: np.random.Generator) -> Dataset:
    """Measurement damage: heavy noise, spikes, and missing values.

    Labels are generated from the CLEAN signal, then the inputs are damaged --
    the information is destroyed at the sensor, not at the labeller. That is what
    makes robust preprocessing (rather than more data) the correct treatment.
    """
    X = data.X.copy()
    y, groups = data.y.copy(), data.groups.copy()
    X = X + rng.normal(0.0, 0.85 * level, size=X.shape)
    spike = rng.random(X.shape) < 0.02 * level
    X[spike] += rng.normal(0.0, 7.0 * level, size=int(spike.sum()))
    missing = rng.random(X.shape) < 0.05 * level
    X[missing] = np.nan
    return Dataset(X, y, groups)


def _perturb_imbalance(data: Dataset, level: float, cfg: Config,
                       rng: np.random.Generator) -> Dataset:
    """Class priors skew: the majority class swells, minorities shrink.

    Implemented by resampling with class-dependent acceptance probability, so
    P(x|y) is untouched and only P(y) moves -- the textbook prior-shift setting.
    """
    keep_prob = np.linspace(1.0, max(0.05, 1.0 - 0.75 * level), cfg.data.n_classes)
    accept = rng.random(data.y.shape[0]) < keep_prob[data.y]
    idx = np.flatnonzero(accept)
    if idx.size < 20:  # never return a degenerate split
        idx = np.arange(data.y.shape[0])
    # Resample with replacement back to the original size so downstream code
    # sees a constant n and cannot infer the cause from the row count alone.
    idx = rng.choice(idx, size=data.y.shape[0], replace=True)
    return Dataset(data.X[idx], data.y[idx], data.groups[idx])


def _identity(data: Dataset) -> Dataset:
    return Dataset(data.X.copy(), data.y.copy(), data.groups.copy())


def build_scenarios(cfg: Config) -> List[Scenario]:
    """Construct the full benchmark: five failure modes plus a healthy control.

    The control is essential for scoring: a system that shouts "failure" at every
    input is useless, so the benchmark must be able to punish false positives.
    """
    builders: Tuple[Callable[[Config], Scenario], ...] = (
        _scenario_environment,
        _scenario_feature,
        _scenario_corruption,
        _scenario_imbalance,
        _scenario_overfit,
        _scenario_control,
    )
    return [build(cfg) for build in builders]


def _splits(cfg: Config, rng: np.random.Generator, weights: np.ndarray,
            n_train: "int | None" = None) -> "Tuple[Dataset, Dataset]":
    train = _sample(n_train or cfg.data.n_train, cfg, weights, rng)
    val = _sample(cfg.data.n_val, cfg, weights, rng)
    return train, val


def _levels(cfg: Config) -> Tuple[float, ...]:
    """Union of validation and hidden-test levels, deduplicated and ordered."""
    return tuple(sorted(set(cfg.data.val_levels) | set(cfg.data.test_levels)))


def _scenario_environment(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed + 101)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights)
    # Random SIGNS, unit magnitude per component: the shift is broad (every
    # feature moves by a comparable amount), which is the defining signature of
    # an environment shift as opposed to a few features destabilising.
    direction = np.where(rng.random(cfg.data.n_features) > 0.5, 1.0, -1.0)

    pool_clean = _sample(cfg.data.n_pool, cfg, weights, rng)
    pool = _perturb_environment(pool_clean, 1.0, direction)
    ood = {
        lvl: _perturb_environment(_sample(cfg.data.n_ood, cfg, weights, rng),
                                  lvl, direction)
        for lvl in _levels(cfg)
    }
    return Scenario("env_shift", "environment_shift", train, val, pool, ood,
                    notes="Global covariate translation + mild dilation.")


def _scenario_feature(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed + 202)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights)
    unstable = rng.choice(cfg.data.n_informative, size=2, replace=False)

    pool = _perturb_feature(_sample(cfg.data.n_pool, cfg, weights, rng),
                            1.0, unstable)
    ood = {
        lvl: _perturb_feature(_sample(cfg.data.n_ood, cfg, weights, rng),
                              lvl, unstable)
        for lvl in _levels(cfg)
    }
    return Scenario("feature_shift", "feature_instability", train, val, pool, ood,
                    notes="Two informative features rescaled and offset.",
                    metadata={"n_unstable": int(unstable.size)})


def _scenario_corruption(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed + 303)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights)

    pool = _perturb_corruption(_sample(cfg.data.n_pool, cfg, weights, rng), 1.0, rng)
    ood = {
        lvl: _perturb_corruption(_sample(cfg.data.n_ood, cfg, weights, rng), lvl, rng)
        for lvl in _levels(cfg)
    }
    return Scenario("input_corruption", "input_corruption", train, val, pool, ood,
                    notes="Gaussian noise, heavy-tailed spikes, missing entries.")


def _scenario_imbalance(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed + 404)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights)

    pool = _perturb_imbalance(_sample(cfg.data.n_pool, cfg, weights, rng), 1.0, cfg, rng)
    ood = {
        lvl: _perturb_imbalance(_sample(cfg.data.n_ood, cfg, weights, rng), lvl, cfg, rng)
        for lvl in _levels(cfg)
    }
    return Scenario("class_shift", "class_imbalance", train, val, pool, ood,
                    notes="Class priors skew; P(x|y) unchanged.")


def _scenario_overfit(cfg: Config) -> Scenario:
    """Generalisation failure with NO distribution change.

    The environment is stationary; the model simply memorised a tiny training
    set. A system that reflexively answers "shift" will get this one wrong, which
    is precisely why it is in the benchmark.
    """
    rng = np.random.default_rng(cfg.seed + 505)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights, n_train=70)

    pool = _sample(cfg.data.n_pool, cfg, weights, rng)
    ood = {lvl: _sample(cfg.data.n_ood, cfg, weights, rng) for lvl in _levels(cfg)}
    return Scenario("overfit", "overfitting", train, val, pool, ood,
                    notes="Tiny training set, over-capacity model, stationary data.",
                    overparameterized=True)


def _scenario_control(cfg: Config) -> Scenario:
    """Healthy model. Correct behaviour is to detect NOTHING and change NOTHING."""
    rng = np.random.default_rng(cfg.seed + 606)
    weights = _teacher(rng, cfg)
    train, val = _splits(cfg, rng, weights)

    pool = _sample(cfg.data.n_pool, cfg, weights, rng)
    ood = {lvl: _identity(_sample(cfg.data.n_ood, cfg, weights, rng)) for lvl in _levels(cfg)}
    return Scenario("control", "none", train, val, pool, ood,
                    notes="No perturbation. False-positive probe.")
