"""End-to-end pipeline, report schema, and robustness tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import Config
from src.doctor import aggregate, diagnose_and_repair, run_benchmark
from src.metrics import (accuracy, balanced_accuracy, expected_calibration_error,
                         fairness_gap, ood_recovery, prediction_stability,
                         sampling_sigma)
from src.report import build_report, case_report, write_report
from src.scenarios import build_scenarios

REQUIRED_CASE_FIELDS = (
    "scenario", "failure_detected", "diagnosis", "selected_repair",
    "performance_before", "performance_after", "ood_recovery",
    "intervention_size", "parameter_ratio", "training_cost_gflops",
    "stability", "fairness_gap", "convergence_log",
)


# --------------------------------------------------------------------------
# Metric unit tests (pure functions, exact answers)
# --------------------------------------------------------------------------


def test_accuracy_and_balanced_accuracy_on_known_arrays() -> None:
    y = np.array([0, 0, 0, 0, 1])
    predictions = np.array([0, 0, 0, 0, 0])
    assert accuracy(y, predictions) == pytest.approx(0.8)
    # Class 1 has zero recall, so balanced accuracy must expose the collapse.
    assert balanced_accuracy(y, predictions) == pytest.approx(0.5)


def test_perfect_calibration_scores_zero_error() -> None:
    y = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_detects_overconfidence() -> None:
    """Always 99% confident, always wrong => error near 1."""
    y = np.array([1, 1, 1, 1])
    proba = np.tile(np.array([0.99, 0.01]), (4, 1))
    assert expected_calibration_error(y, proba) > 0.9


def test_fairness_gap_is_zero_when_groups_match() -> None:
    y = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])
    assert fairness_gap(y, predictions, groups) == pytest.approx(0.0)


def test_fairness_gap_measures_the_worst_pair() -> None:
    y = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 0])      # group 1 entirely wrong
    groups = np.array([0, 0, 1, 1])
    assert fairness_gap(y, predictions, groups) == pytest.approx(1.0)


def test_recovery_formula_endpoints() -> None:
    assert ood_recovery(0.6, 0.6, 0.9) == pytest.approx(0.0)   # no improvement
    assert ood_recovery(0.6, 0.9, 0.9) == pytest.approx(1.0)   # full recovery
    assert ood_recovery(0.6, 0.75, 0.9) == pytest.approx(0.5)  # half
    assert ood_recovery(0.9, 0.95, 0.9) == pytest.approx(1.0)  # nothing to recover
    assert ood_recovery(0.6, 0.99, 0.9) == pytest.approx(1.0)  # capped, not inflated


def test_stability_rewards_flat_performance() -> None:
    assert prediction_stability([0.8, 0.8, 0.8]) == pytest.approx(1.0)
    assert prediction_stability([0.9, 0.5, 0.7]) < 0.9
    assert prediction_stability([0.8]) == pytest.approx(1.0)


def test_sampling_sigma_shrinks_with_sample_size() -> None:
    """The statistical basis of adaptive detection."""
    assert sampling_sigma(0.8, 100) > sampling_sigma(0.8, 10000)
    assert sampling_sigma(0.8, 10000) == pytest.approx(0.004, abs=1e-3)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results(tiny_config: Config, scenarios):
    return run_benchmark(scenarios, tiny_config)


def test_benchmark_covers_every_scenario(results, scenarios) -> None:
    assert len(results) == len(scenarios)


def test_healthy_model_is_returned_unmodified(results) -> None:
    """DO NO HARM, end to end: no repair object, no change in scores."""
    control = next(r for r in results if r.hidden_cause == "none")
    assert control.candidate is None
    assert control.selected_family == "none"
    assert control.baseline_test == control.repaired_test


def test_repairs_never_reduce_accuracy(results) -> None:
    """A repair must not leave the patient worse than it found them.

    Checked on the HIDDEN test levels, so this is a genuine generalisation claim
    rather than a restatement of the search objective. A small tolerance absorbs
    sampling noise on the finite evaluation sets.
    """
    from src.metrics import mean_over_levels
    for case in results:
        before = mean_over_levels(case.baseline_test)
        after = mean_over_levels(case.repaired_test)
        assert after >= before - 0.02, f"{case.scenario} regressed: {before} -> {after}"


def test_parameter_budget_is_honoured(results, tiny_config: Config) -> None:
    """The deployment-cost hard constraint, checked on delivered repairs."""
    for case in results:
        if case.candidate is not None:
            assert case.candidate.param_ratio <= tiny_config.repair.param_budget_ratio + 1e-6


def test_repairs_are_evaluated_on_unseen_drift_levels(results, tiny_config) -> None:
    """The search optimises 0.8/1.0/1.2; scoring uses 0.6/1.0/1.5/2.0.

    If these sets ever coincided, our robustness claim would be circular.
    """
    unseen = set(tiny_config.data.test_levels) - set(tiny_config.data.val_levels)
    assert unseen, "hidden test levels must differ from validation levels"
    for case in results:
        assert unseen.issubset(set(case.repaired_test))


def test_report_schema_is_complete(results, tiny_config: Config) -> None:
    for case in results:
        payload = case_report(case)
        for field in REQUIRED_CASE_FIELDS:
            assert field in payload, f"missing {field}"


def test_report_is_json_serialisable(results, tiny_config: Config, tmp_path) -> None:
    """The submission is a JSON file, so non-serialisable types must fail here."""
    payload = build_report(results, tiny_config, 1.23)
    target = write_report(payload, tmp_path / "report.json")
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["summary"]["n_cases"] == len(results)
    assert "fitness" in reloaded["summary"]


def test_report_contains_a_convergence_log_for_every_repair(results) -> None:
    """Required output of the track spec."""
    for case in results:
        if case.candidate is None:
            continue
        assert case.convergence, f"{case.scenario} produced no convergence log"
        last = case.convergence[-1]
        assert {"generation", "front_size", "best_per_objective",
                "n_evaluations"} <= set(last)


def test_aggregate_fields_are_in_range(results) -> None:
    summary = aggregate(results)
    assert 0.0 <= summary["diagnosis_accuracy"] <= 1.0
    assert 0.0 <= summary["mean_ood_recovery"] <= 1.0
    assert 0.0 <= summary["fitness"] <= 1.0
    assert summary["n_cases"] == len(results)


def test_no_hardcoded_fitness(results, tiny_config: Config) -> None:
    """The score must be computed from the run, not asserted into existence.

    Changing the seed must change the numbers; if it did not, the pipeline would
    be returning a constant and the auditor would be right to reject it.
    """
    other = Config(seed=tiny_config.seed + 1, search=tiny_config.search,
                   data=tiny_config.data, model=tiny_config.model,
                   repair=tiny_config.repair)
    other_results = run_benchmark(build_scenarios(other), other)
    assert aggregate(other_results) != aggregate(results)


def test_pipeline_survives_a_pool_full_of_nans(tiny_config: Config, scenarios) -> None:
    """Adaptability probe: extreme corruption must degrade, not crash.

    This stands in for a Round-2 perturbation we have not seen. The system must
    still produce a report rather than raising.
    """
    import copy
    scenario = copy.deepcopy(next(s for s in scenarios
                                  if s.hidden_cause == "environment_shift"))
    rng = np.random.default_rng(0)
    scenario.pool.X[rng.random(scenario.pool.X.shape) < 0.4] = np.nan
    case = diagnose_and_repair(scenario, tiny_config)
    assert case.detection is not None
    assert 0.0 <= case.recovery <= 1.0


def test_pipeline_handles_a_shape_it_was_not_configured_for(tiny_config: Config) -> None:
    """No data shape may be hard-coded -- a stated Round-2 defence.

    We rebuild the benchmark with a different feature count, class count and
    sample size and require the whole pipeline to run unchanged.
    """
    from src.config import DataConfig
    reshaped = tiny_config.evolve(
        data=DataConfig(n_features=17, n_informative=9, n_classes=4,
                        n_train=200, n_val=150, n_pool=220, n_ood=150))
    scenario = build_scenarios(reshaped)[0]
    case = diagnose_and_repair(scenario, reshaped)
    assert case.detection is not None
    assert len(case.diagnosis.confidences) == 5
