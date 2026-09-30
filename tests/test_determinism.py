"""Determinism tests -- the hard constraint of the track spec.

These are the highest-value tests in the suite. The earlier prototype was
deterministic for a single evaluation and for generation 1, then diverged in
later generations, so a test that only checks one evaluation would have PASSED on
the broken build. Every test here therefore runs a MULTI-GENERATION search, which
is the regime where the bug actually lived.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.determinism import genome_seed, stable_hash
from src.doctor import (diagnose_and_repair, run_search, train_deployed_model)
from src.metrics import mean_over_levels
from src.detect import evaluate_on_levels
from src.nsga2 import REAL, Problem, minimize
from src.repairs import RepairContext, build_patchml


def test_stable_hash_is_process_independent() -> None:
    """BLAKE2b, not Python's salted hash(), so seeds survive a new process."""
    assert stable_hash(b"PatchML") == stable_hash(b"PatchML")
    assert stable_hash(b"PatchML") != stable_hash(b"Regularisation")
    assert 0 <= stable_hash(b"x") < 2**63


def test_genome_seed_is_content_addressed() -> None:
    """Equal genomes => equal seeds; this is what makes fitness a function."""
    a = np.array([1.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 1.0])
    c = np.array([1.0, 1.0, 1.0])
    assert genome_seed(5, a) == genome_seed(5, b)
    assert genome_seed(5, a) != genome_seed(5, c)
    assert genome_seed(5, a) != genome_seed(6, a)


def _toy_problem(n: int = 8) -> Problem:
    def evaluate(x: np.ndarray):
        return np.array([float(x.sum()), float(np.abs(x - 0.5).sum())]), np.array([0.0])

    return Problem(n, np.zeros(n), np.ones(n), np.full(n, REAL), evaluate, 2, 1)


def test_nsga2_multi_generation_is_bit_identical() -> None:
    """Twenty generations, twice, byte for byte.

    Generation count is deliberately high: the historical failure appeared only
    after the population had been through several selection rounds.
    """
    problem = _toy_problem()
    first = minimize(problem, pop_size=10, generations=20, seed=99)
    second = minimize(problem, pop_size=10, generations=20, seed=99)
    assert np.array_equal(first.genomes, second.genomes)
    assert np.array_equal(first.objectives, second.objectives)
    assert first.history == second.history


def test_nsga2_different_seeds_diverge() -> None:
    """A guard against the opposite bug: determinism by accidental constancy.

    If the search ignored its RNG entirely it would also be 'deterministic', so
    we must show the seed genuinely drives the trajectory.
    """
    problem = _toy_problem()
    a = minimize(problem, pop_size=10, generations=5, seed=1)
    b = minimize(problem, pop_size=10, generations=5, seed=2)
    assert not np.array_equal(a.genomes, b.genomes)


def test_nsga2_does_not_touch_the_global_rng() -> None:
    """Root cause 1: a shared global stream coupled evaluation to operators.

    We advance NumPy's legacy global stream between two runs. If any part of the
    search or the evaluator drew from it, the results would differ.
    """
    problem = _toy_problem()
    np.random.seed(0)
    first = minimize(problem, pop_size=10, generations=8, seed=42)
    np.random.seed(0)
    for _ in range(1000):
        np.random.random()  # deliberately desynchronise the global stream
    second = minimize(problem, pop_size=10, generations=8, seed=42)
    assert np.array_equal(first.objectives, second.objectives)


def test_patchml_search_is_reproducible(tiny_config: Config, env_scenario) -> None:
    """End-to-end: the real PatchML fitness, over multiple generations, twice.

    This is the regression test for the reported non-determinism. It exercises
    model fitting inside the fitness function, which is where the floating-point
    and RNG coupling issues lived.
    """
    base = train_deployed_model(env_scenario, tiny_config)
    context = RepairContext(
        env_scenario, base, tiny_config,
        failed_score=mean_over_levels(
            evaluate_on_levels(base, env_scenario, tiny_config.data.val_levels)),
        seed=tiny_config.seed,
    )
    first, first_result = run_search("PatchML", context, tiny_config)
    second, second_result = run_search("PatchML", context, tiny_config)

    assert first is not None and second is not None
    assert np.array_equal(first_result.objectives, second_result.objectives)
    assert first_result.history == second_result.history
    assert first.intervention_size == pytest.approx(second.intervention_size)
    assert first.score == pytest.approx(second.score, abs=0.0)


def test_repeated_evaluation_of_one_genome_is_identical(tiny_config: Config,
                                                       env_scenario) -> None:
    """The fitness map must be a function: same genome in, same vector out."""
    base = train_deployed_model(env_scenario, tiny_config)
    context = RepairContext(
        env_scenario, base, tiny_config,
        failed_score=mean_over_levels(
            evaluate_on_levels(base, env_scenario, tiny_config.data.val_levels)),
        seed=tiny_config.seed,
    )
    family = build_patchml(context)
    genome = np.zeros(family.problem.n_var)
    genome[:20] = 1.0

    # A fresh evaluator each time defeats the cache, so this tests the underlying
    # decode+fit path rather than a memoised lookup.
    from src.repairs import make_evaluator
    runs = [make_evaluator(family.decode, context)(genome.copy()) for _ in range(3)]
    for objectives, violations in runs[1:]:
        assert np.array_equal(objectives, runs[0][0])
        assert np.array_equal(violations, runs[0][1])


def test_full_case_is_reproducible(tiny_config: Config, scenarios) -> None:
    """A whole Detect->Diagnose->Repair cycle, run twice, must agree exactly."""
    scenario = scenarios[0]
    a = diagnose_and_repair(scenario, tiny_config)
    b = diagnose_and_repair(scenario, tiny_config)
    assert a.diagnosis.confidences == b.diagnosis.confidences
    assert a.selected_family == b.selected_family
    assert a.detection.as_dict() == b.detection.as_dict()
    assert a.recovery == pytest.approx(b.recovery, abs=0.0)
    assert a.repaired_test == b.repaired_test
