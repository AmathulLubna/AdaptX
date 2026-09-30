"""Base learner wrapper: a small MLP with deterministic, bounded training.

WHY AN MLP
----------
The track asks for "deep evolutionary networks". An MLP is the smallest model
that (a) has a genuine non-convex loss surface, so regularisation and capacity
repairs actually bite, and (b) supports warm-start fine-tuning, which is what
makes PatchML cheap -- we adapt existing weights instead of retraining.

NO GRADIENT EXPLOSION (hard constraint 2 of the track spec)
-----------------------------------------------------------
Three structural guards, none of them a runtime check that could fail silently:
  1. Inputs are standardised by a scaler fitted on the ORIGINAL training data,
     so activations stay O(1) even when the new environment has shifted.
  2. `alpha` (L2) is bounded strictly positive, making the objective strongly
     convex in the weight norm and bounding the gradient of the penalty term.
  3. `max_iter` and `learning_rate_init` are capped in `ModelConfig`, and Adam
     normalises gradient magnitude by its second-moment estimate, so a step can
     never exceed ~`learning_rate_init` regardless of loss curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import Config


@dataclass
class FittedModel:
    """A trained classifier plus the preprocessing it was fitted with.

    Keeping the scaler bound to the model is not cosmetic: a repair that changes
    preprocessing (robust clipping, imputation) must travel with the weights, or
    the deployed pipeline silently diverges from the evaluated one.
    """

    scaler: StandardScaler
    net: MLPClassifier
    classes: np.ndarray
    n_params: int
    train_cost: float = 0.0
    feature_weights: "np.ndarray | None" = None
    clip_sigma: float = 0.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply imputation -> optional robust clipping -> scaling -> weighting."""
        Z = impute(X, self.scaler.mean_)
        Z = self.scaler.transform(Z)
        if self.clip_sigma > 0:
            Z = np.clip(Z, -self.clip_sigma, self.clip_sigma)
        if self.feature_weights is not None:
            Z = Z * self.feature_weights
        return Z

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class predictions, with the model's own preprocessing applied."""
        return self.net.predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, used for calibration and PatchML margins."""
        return self.net.predict_proba(self.transform(X))


def impute(X: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the original training mean.

    Using the ORIGINAL mean rather than the new batch mean is deliberate: the new
    batch may itself be corrupted, and imputing from corrupted statistics would
    propagate the damage. Non-finite (not just NaN) is checked because spike
    noise can overflow to +/-inf.
    """
    Z = np.asarray(X, dtype=np.float64).copy()
    bad = ~np.isfinite(Z)
    if bad.any():
        Z[bad] = np.take(fallback, np.nonzero(bad)[1])
    return Z


def count_params(net: MLPClassifier) -> int:
    """Total trainable parameters -- the deployment-cost objective."""
    return int(sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_))


def _make_net(cfg: Config, hidden: Tuple[int, ...], alpha: float,
              max_iter: int, seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        max_iter=max_iter,
        learning_rate_init=cfg.model.learning_rate_init,
        solver="adam",
        random_state=int(seed % (2**31 - 1)),  # explicit => reproducible
        shuffle=True,
        early_stopping=False,
        n_iter_no_change=10,
        tol=1e-4,
    )


def train_model(X: np.ndarray, y: np.ndarray, cfg: Config, seed: int,
                hidden: "Tuple[int, ...] | None" = None,
                alpha: "float | None" = None,
                max_iter: "int | None" = None,
                sample_weight_classes: "np.ndarray | None" = None) -> FittedModel:
    """Fit a scaler and an MLP from scratch, deterministically.

    `sample_weight_classes` gives per-class weights. scikit-learn's MLP has no
    `sample_weight` argument, so class weighting is applied by *replication*
    (resampling proportional to weight) -- mathematically equivalent to weighting
    the empirical risk, and it keeps the solver untouched.
    """
    rng = np.random.default_rng(seed)
    hidden = hidden or cfg.model.hidden_sizes
    alpha = cfg.model.alpha if alpha is None else alpha
    max_iter = max_iter or cfg.model.max_iter

    Xs = np.asarray(X, dtype=np.float64)
    finite_mean = np.nanmean(np.where(np.isfinite(Xs), Xs, np.nan), axis=0)
    finite_mean = np.where(np.isfinite(finite_mean), finite_mean, 0.0)
    scaler = StandardScaler().fit(impute(Xs, finite_mean))

    Xw, yw = Xs, np.asarray(y)
    if sample_weight_classes is not None:
        Xw, yw = _replicate_by_class_weight(Xw, yw, sample_weight_classes, rng)

    net = _make_net(cfg, tuple(hidden), alpha, max_iter, seed)
    cost = _fit_with_cost(net, scaler.transform(impute(Xw, scaler.mean_)), yw)
    return FittedModel(scaler, net, net.classes_, count_params(net), cost)


def _replicate_by_class_weight(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                               rng: np.random.Generator) -> "Tuple[np.ndarray, np.ndarray]":
    """Resample rows with probability proportional to their class weight."""
    classes = np.unique(y)
    w = np.ones(y.shape[0], dtype=np.float64)
    for i, cls in enumerate(classes):
        if i < weights.shape[0]:
            w[y == cls] = max(float(weights[i]), 1e-6)
    probs = w / w.sum()
    idx = rng.choice(y.shape[0], size=y.shape[0], replace=True, p=probs)
    return X[idx], y[idx]


def _fit_with_cost(net: MLPClassifier, Xz: np.ndarray, y: np.ndarray) -> float:
    """Fit and return a hardware-independent cost proxy.

    Wall-clock time is NOT used: it varies with machine load and would make the
    fitness non-deterministic, breaking the reproducibility guarantee. Instead
    cost = n_samples * n_iterations * n_parameters / 1e9, which is proportional
    to the FLOPs actually performed and is a pure function of the configuration.
    """
    import warnings

    with warnings.catch_warnings():
        # Convergence warnings are expected: iteration caps are a deliberate
        # budget constraint, not a defect.
        warnings.simplefilter("ignore")
        net.fit(Xz, y)
    return float(Xz.shape[0] * net.n_iter_ * count_params(net)) / 1e9


def finetune(base: FittedModel, X: np.ndarray, y: np.ndarray, cfg: Config,
             seed: int, iters: "int | None" = None,
             feature_weights: "np.ndarray | None" = None,
             clip_sigma: float = 0.0) -> FittedModel:
    """Warm-start adaptation of an existing model on a repair set.

    Copies the fitted network and continues Adam from the deployed weights, so
    the repair is an *intervention on the existing model*, not a replacement.
    This is the operational difference between AdaptX and AutoML, and it is
    also why the parameter-count objective usually reports zero change.

    All classes must appear in `y` for `partial_fit`; the caller guarantees this
    by always mixing in a replay sample from the original training data.
    """
    import copy
    import warnings

    net = copy.deepcopy(base.net)
    repaired = FittedModel(base.scaler, net, base.classes, base.n_params,
                           feature_weights=feature_weights, clip_sigma=clip_sigma)
    Xz = repaired.transform(X)
    iters = iters or cfg.repair.finetune_iters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(iters):
            net.partial_fit(Xz, y, classes=base.classes)

    repaired.n_params = count_params(net)
    repaired.train_cost = float(Xz.shape[0] * iters * repaired.n_params) / 1e9
    return repaired
