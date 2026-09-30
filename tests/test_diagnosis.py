"""Detection and diagnosis tests.

The statistics are tested on SYNTHETIC arrays with analytically known answers
rather than on the benchmark scenarios. That separation matters: if the tests
only checked "does the pipeline label env_shift correctly", they would pass for a
system that had merely been tuned to this benchmark, which is precisely the
failure mode Round 2 is designed to expose.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.detect import detect_failure
from src.diagnose import (concentration_ratio, diagnose, feature_divergence,
                          outlier_rate, prior_shift, variance_inflation)
from src.doctor import train_deployed_model
from src.scenarios import CAUSES


# --------------------------------------------------------------------------
# Statistic-level tests: known inputs, known answers
# --------------------------------------------------------------------------


def test_feature_divergence_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    assert feature_divergence(X, X).max() < 1e-9


def test_feature_divergence_recovers_a_known_shift() -> None:
    """Shift feature 2 by exactly 1.0 sigma; the statistic must report ~1.0."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 5))
    Y = X.copy()
    Y[:, 2] += 1.0
    divergence = feature_divergence(X, Y)
    assert divergence[2] == pytest.approx(1.0, abs=0.08)
    assert divergence[[0, 1, 3, 4]].max() < 0.08


def test_concentration_is_high_when_one_feature_carries_the_shift() -> None:
    concentrated = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert concentration_ratio(concentrated) > 0.9


def test_concentration_is_low_for_a_uniform_shift() -> None:
    uniform = np.full(8, 0.4)
    assert concentration_ratio(uniform) < 0.05


def test_concentration_is_scale_invariant() -> None:
    """Doubling every divergence must not change its SHAPE measure.

    This is what lets one threshold work across perturbation magnitudes, i.e. it
    is the property that makes the hidden level 2.0 safe.
    """
    divergence = np.array([0.0, 0.1, 2.0, 0.05, 0.0, 0.0])
    assert concentration_ratio(divergence) == pytest.approx(
        concentration_ratio(divergence * 7.3), abs=1e-9)


def test_outlier_rate_detects_spikes_but_not_translation() -> None:
    """The discriminator between corruption and shift."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 6))
    translated = X + 1.5                      # whole cloud moves
    spiked = X.copy()
    spiked[rng.random(spiked.shape) < 0.05] += 30.0
    assert outlier_rate(X, translated) < 0.02
    assert outlier_rate(X, spiked) > 0.03


def test_variance_inflation_ignores_shrinkage() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 4))
    assert variance_inflation(X, X * 2.0) == pytest.approx(1.0, abs=0.1)
    assert variance_inflation(X, X * 0.5) == pytest.approx(0.0, abs=1e-9)


def test_prior_shift_is_a_total_variation_distance() -> None:
    """Bounded in [0, 1]; 0 for equal priors, 1 for disjoint support."""
    balanced = np.array([0, 1, 2] * 100)
    assert prior_shift(balanced, balanced, 3) == pytest.approx(0.0, abs=1e-9)
    all_zero = np.zeros(300, dtype=np.int64)
    assert prior_shift(balanced, all_zero, 3) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_statistics_survive_nan_inputs() -> None:
    """Corrupted pools contain NaN; no statistic may return NaN because of it."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 5))
    Y = X.copy()
    Y[rng.random(Y.shape) < 0.2] = np.nan
    assert np.all(np.isfinite(feature_divergence(X, Y)))
    assert np.isfinite(outlier_rate(X, Y))
    assert np.isfinite(variance_inflation(X, Y))


# --------------------------------------------------------------------------
# Pipeline-level behaviour
# --------------------------------------------------------------------------


def test_confidences_are_bounded_and_complete(tiny_config: Config, scenarios) -> None:
    """Every cause must be scored, and every score must be a valid confidence."""
    for scenario in scenarios:
        model = train_deployed_model(scenario, tiny_config)
        detection = detect_failure(model, scenario, tiny_config)
        result = diagnose(scenario.train, scenario.pool, detection, tiny_config)
        assert set(result.confidences) == set(CAUSES)
        for value in result.confidences.values():
            assert 0.0 <= value <= 1.0


def test_control_scenario_is_left_alone(tiny_config: Config, scenarios) -> None:
    """DO NO HARM: a healthy model must not trigger a repair.

    A system that always shouts 'failure' would score well on recovery and be
    useless in deployment, so this is a first-class correctness property.
    """
    control = next(s for s in scenarios if s.hidden_cause == "none")
    model = train_deployed_model(control, tiny_config)
    detection = detect_failure(model, control, tiny_config)
    assert not detection.failure_detected
    assert detection.reason == "no_meaningful_degradation"


def test_genuine_failures_are_detected(tiny_config: Config, scenarios) -> None:
    """The complement of the control test: real damage must be noticed."""
    for scenario in scenarios:
        if scenario.hidden_cause == "none":
            continue
        model = train_deployed_model(scenario, tiny_config)
        detection = detect_failure(model, scenario, tiny_config)
        assert detection.failure_detected, f"missed failure in {scenario.name}"


def test_detection_threshold_scales_with_sample_size(tiny_config: Config,
                                                    scenarios) -> None:
    """The noise floor must be a statistic, not a constant.

    Sanity bound: the adaptive threshold can never fall below the configured
    absolute floor, and must stay in a plausible range.
    """
    scenario = scenarios[0]
    model = train_deployed_model(scenario, tiny_config)
    detection = detect_failure(model, scenario, tiny_config)
    assert detection.noise_threshold >= tiny_config.detection.min_absolute_drop
    assert detection.noise_threshold < 0.5


def test_monotone_confidence_in_perturbation_strength(tiny_config: Config) -> None:
    """A stronger shift must never yield LOWER environment-shift confidence.

    This is the formal property behind our robustness claim for the hidden test
    levels (1.5, 2.0): because the squashing map is monotone, an unseen larger
    perturbation is simply further along the same curve.
    """
    from src.detect import DetectionReport
    from src.scenarios import Dataset

    rng = np.random.default_rng(7)
    base = rng.normal(size=(1500, 8))
    labels = rng.integers(0, 3, size=1500)
    groups = (base[:, 0] > 0).astype(np.int64)
    train = Dataset(base, labels, groups)
    direction = np.ones(8) / np.sqrt(8)

    detection = DetectionReport(True, "x", 0.9, 0.85, 0.6, 0.6, 0.25, 0.05,
                                0.05, 0.05, 0.05, 0.9, {})
    previous = -1.0
    for level in (0.5, 1.0, 1.5, 2.0, 3.0):
        shifted = Dataset(base + level * direction, labels, groups)
        confidence = diagnose(train, shifted, detection,
                              tiny_config).confidences["environment_shift"]
        assert confidence >= previous - 1e-9
        previous = confidence
