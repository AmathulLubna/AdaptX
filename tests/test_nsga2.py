"""Correctness tests for the hand-written NSGA-II engine.

The engine replaces a published library, so its core routines must be proved
against cases whose answers are known analytically -- otherwise we have traded a
dependency for an unverified reimplementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nsga2 import (BINARY, REAL, Problem, crowding_distance, dominates,
                       fast_non_dominated_sort, minimize)


def test_dominance_basic() -> None:
    """Pareto dominance on feasible points: better in one, no worse in any."""
    assert dominates(np.array([1.0, 2.0]), 0.0, np.array([2.0, 3.0]), 0.0)
    assert not dominates(np.array([1.0, 3.0]), 0.0, np.array([2.0, 2.0]), 0.0)
    assert not dominates(np.array([1.0, 2.0]), 0.0, np.array([1.0, 2.0]), 0.0)


def test_constraint_domination_rules() -> None:
    """Feasible beats infeasible; among infeasible, less violation wins."""
    good, bad = np.array([9.0, 9.0]), np.array([0.0, 0.0])
    assert dominates(good, 0.0, bad, 0.5)          # feasible wins despite worse objectives
    assert not dominates(bad, 0.5, good, 0.0)
    assert dominates(bad, 0.1, bad, 0.9)           # smaller violation wins
    assert not dominates(bad, 0.9, bad, 0.1)


def test_non_dominated_sort_assigns_known_fronts() -> None:
    """A hand-checked example with two clean fronts."""
    objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0],   # front 0
                           [4.0, 5.0], [5.0, 6.0]])              # front 1, then 2
    ranks = fast_non_dominated_sort(objectives, np.zeros(5))
    assert list(ranks[:3]) == [0, 0, 0]
    assert ranks[3] == 1
    assert ranks[4] == 2


def test_every_individual_gets_a_rank() -> None:
    """No -1 sentinels may survive: an unranked individual breaks survival."""
    rng = np.random.default_rng(0)
    objectives = rng.random((40, 3))
    ranks = fast_non_dominated_sort(objectives, np.zeros(40))
    assert ranks.min() >= 0


def test_crowding_distance_boundaries_are_infinite() -> None:
    """Extremes must be preserved, or the front collapses inward over time."""
    objectives = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(objectives)
    assert np.isinf(distance[0]) and np.isinf(distance[2])
    assert np.isfinite(distance[1])


def test_crowding_distance_handles_degenerate_objective() -> None:
    """A constant objective must not divide by zero."""
    objectives = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 0.25]])
    distance = crowding_distance(objectives)
    assert np.all(~np.isnan(distance))


def _zdt1(n: int = 6) -> Problem:
    """ZDT1: convex Pareto front f2 = 1 - sqrt(f1) at g = 1. A standard probe."""
    def evaluate(x: np.ndarray):
        f1 = float(x[0])
        g = 1.0 + 9.0 * float(np.mean(x[1:]))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2]), np.array([0.0])

    return Problem(n, np.zeros(n), np.ones(n), np.full(n, REAL), evaluate, 2, 1)


def test_converges_towards_the_known_zdt1_front() -> None:
    """Quality, not just reproducibility: the search must actually improve.

    Convergence is asserted as a monotone improvement between an early and a late
    generation, which is the property elitism guarantees. We do not assert a
    specific final value -- that would be a hard-coded result.
    """
    result = minimize(_zdt1(), pop_size=20, generations=25, seed=3)
    early = result.history[0]["best_per_objective"][1]
    late = result.history[-1]["best_per_objective"][1]
    assert late <= early
    assert late < 1.6  # true optimum is 1.0; loose bound, not a memorised number


def test_elitism_never_loses_the_best_objective_value() -> None:
    """(mu + lambda) truncation implies per-objective best is non-increasing."""
    result = minimize(_zdt1(), pop_size=20, generations=15, seed=11)
    bests = [h["best_per_objective"][1] for h in result.history]
    assert all(b <= a + 1e-12 for a, b in zip(bests, bests[1:]))


def test_binary_genes_stay_binary() -> None:
    """Bit-flip and uniform crossover must not leak fractional values."""
    n = 12

    def evaluate(x: np.ndarray):
        return np.array([float(x.sum()), float(n - x.sum())]), np.array([0.0])

    problem = Problem(n, np.zeros(n), np.ones(n), np.full(n, BINARY), evaluate, 2, 1)
    result = minimize(problem, pop_size=10, generations=10, seed=5)
    assert np.all(np.isin(result.genomes, (0.0, 1.0)))


def test_real_genes_respect_bounds() -> None:
    """SBX and polynomial mutation must clip to the feasible box."""
    n = 5
    lower, upper = np.full(n, -2.0), np.full(n, 3.0)

    def evaluate(x: np.ndarray):
        return np.array([float(np.sum(x**2)), float(np.sum((x - 1) ** 2))]), np.array([0.0])

    problem = Problem(n, lower, upper, np.full(n, REAL), evaluate, 2, 1)
    result = minimize(problem, pop_size=10, generations=10, seed=6)
    assert np.all(result.genomes >= lower - 1e-9)
    assert np.all(result.genomes <= upper + 1e-9)


def test_constraints_are_respected_when_satisfiable() -> None:
    """The search must end up in the feasible region when one exists."""
    n = 4

    def evaluate(x: np.ndarray):
        objectives = np.array([float(x.sum()), float(-x.sum())])
        violation = np.array([max(0.0, 2.0 - float(x.sum()))])  # require sum >= 2
        return objectives, violation

    problem = Problem(n, np.zeros(n), np.ones(n), np.full(n, REAL), evaluate, 2, 1)
    result = minimize(problem, pop_size=12, generations=20, seed=8)
    genomes, _ = result.pareto_front()
    assert genomes.shape[0] > 0
    assert np.all(genomes.sum(axis=1) >= 2.0 - 1e-6)


def test_warm_starts_enter_the_initial_population() -> None:
    """Seeded genomes must actually be used, or the innovation claim is empty."""
    n = 6
    warm = np.full(n, 0.25)

    def evaluate(x: np.ndarray):
        return np.array([float(x.sum()), float(np.std(x))]), np.array([0.0])

    problem = Problem(n, np.zeros(n), np.ones(n), np.full(n, REAL), evaluate, 2, 1)
    result = minimize(problem, pop_size=6, generations=0, seed=4,
                      initial_population=[warm])
    assert np.any(np.all(np.isclose(result.genomes, warm), axis=1))


def test_odd_population_size_is_rejected() -> None:
    """Fail loudly rather than silently dropping a parent during pairing."""
    with pytest.raises(ValueError):
        minimize(_zdt1(), pop_size=7, generations=1, seed=1)


def test_mismatched_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        Problem(3, np.zeros(2), np.ones(2), np.full(3, REAL),
                lambda x: (np.zeros(2), np.zeros(1)), 2, 1)


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        Problem(2, np.ones(2), np.zeros(2), np.full(2, REAL),
                lambda x: (np.zeros(2), np.zeros(1)), 2, 1)
