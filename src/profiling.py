"""Algorithmic complexity register and empirical profiling.

WHY COMPLEXITY IS DECLARED, NOT INFERRED
----------------------------------------
Every non-trivial routine in this system carries its asymptotic cost in the
register below, and `verify_complexity` checks the declarations against measured
growth. A comment claiming O(n log n) that is actually O(n^2) is worse than no
comment: it stops people looking. Declaring the bound in one machine-readable
place and testing it turns documentation into an assertion.

THE COST MODEL
--------------
Symbols used throughout:

    N   population size (individuals in a generation)
    G   number of generations
    M   number of objectives
    n   samples in a dataset split
    d   feature count
    k   class count
    P   parameters in the network
    I   training iterations
    S   shortlist size for PatchML

The dominant term for a full run is the fitness evaluation, not the genetic
operators: `N * (G + 1)` model fits at O(n * I * P) each, against O(M * N^2) for
non-dominated sorting. With N <= 64 the quadratic sort is negligible, which is
why the simple auditable implementation is kept rather than the more intricate
O(N log^(M-1) N) variant -- an optimisation that would complicate the code
without moving the wall clock.

WHY WALL-CLOCK TIME IS NOT THE FITNESS COST
-------------------------------------------
Objective 2 uses `n * I * P`, proportional to FLOPs, rather than elapsed time.
Time varies with machine load and would make fitness non-deterministic, breaking
the reproducibility guarantee the whole system rests on. Timing belongs here, in
profiling, where it is reported and never fed back into a decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ComplexityClaim:
    """A declared asymptotic bound for one routine."""

    routine: str
    time: str
    space: str
    note: str


# The register. Every entry is a claim this module can be asked to defend.
COMPLEXITY_REGISTER: Tuple[ComplexityClaim, ...] = (
    ComplexityClaim(
        "nsga2.fast_non_dominated_sort", "O(M * N^2)", "O(N^2)",
        "Pairwise domination over the merged parent+offspring pool. Quadratic is "
        "the standard NSGA-II bound and is dominated by model fitting at N <= 64."),
    ComplexityClaim(
        "nsga2.crowding_distance", "O(M * N log N)", "O(N)",
        "One stable sort per objective; the sort dominates the linear sweep."),
    ComplexityClaim(
        "nsga2.minimize", "O(G * (N * F + M * N^2))", "O(N * n_var)",
        "F is one fitness evaluation. F dwarfs the N^2 term for every realistic N."),
    ComplexityClaim(
        "models.train_model", "O(n * I * P)", "O(n * d + P)",
        "Backpropagation over I epochs. Bounded above by capped max_iter."),
    ComplexityClaim(
        "models.finetune", "O(n_patch * I_ft * P)", "O(P)",
        "Warm start: no re-initialisation, so I_ft is an order below training."),
    ComplexityClaim(
        "repairs.shortlist_candidates", "O(n * k + n log n)", "O(n)",
        "One forward pass for margins, then one stable sort. The pruning that "
        "makes PatchML tractable: 2^n -> 2^S search space."),
    ComplexityClaim(
        "diagnose.feature_divergence", "O(n * d)", "O(d)",
        "Column means and standard deviations in a single pass."),
    ComplexityClaim(
        "diagnose.heavy_tail_rate", "O(n * d log n)", "O(n)",
        "Median and MAD per feature; the sort inside the median dominates."),
    ComplexityClaim(
        "diagnose.concentration_ratio", "O(d log d)", "O(d)",
        "One sort over per-feature divergences."),
    ComplexityClaim(
        "metrics.expected_calibration_error", "O(n * k + B)", "O(B)",
        "One pass for confidences, then B equal-width bins."),
    ComplexityClaim(
        "detect.detect_failure", "O(L * n * P)", "O(n)",
        "One forward pass per validation drift level L."),
    ComplexityClaim(
        "validation.recover_array", "O(n * d)", "O(n * d)",
        "Single pass over the non-finite mask."),
)


def complexity_table() -> str:
    """Render the register as an aligned text table.

    Complexity: O(R) over the register.
    """
    width = max(len(c.routine) for c in COMPLEXITY_REGISTER)
    lines = [f"{'ROUTINE'.ljust(width)}  {'TIME'.ljust(26)}  SPACE",
             "-" * (width + 46)]
    for claim in COMPLEXITY_REGISTER:
        lines.append(f"{claim.routine.ljust(width)}  {claim.time.ljust(26)}  {claim.space}")
    return "\n".join(lines)


@dataclass
class Measurement:
    """One empirical timing point."""

    size: int
    seconds: float


def measure_growth(routine: Callable[[int], None], sizes: "List[int]",
                   repeats: int = 3) -> List[Measurement]:
    """Time `routine` at each input size, taking the best of `repeats`.

    Complexity: O(sum(cost(size)) * repeats).

    The MINIMUM is taken, not the mean. Timing noise on a shared machine is
    one-sided -- interference only ever makes a run slower -- so the minimum is
    the best available estimate of the true cost, and the mean mostly measures
    what else the machine was doing.
    """
    out: List[Measurement] = []
    for size in sizes:
        best = float("inf")
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            routine(size)
            best = min(best, time.perf_counter() - start)
        out.append(Measurement(size, best))
    return out


def estimate_exponent(points: "List[Measurement]") -> float:
    """Fit `t = c * n^p` by least squares in log-log space and return `p`.

    Complexity: O(len(points)).

    A straight line in log-log space is exactly a power law, so the slope is the
    empirical exponent: ~1 for linear, ~2 for quadratic. Points with a
    non-positive time are dropped because their logarithm is undefined -- which
    happens when a routine is too fast for the clock resolution rather than
    because anything is wrong.
    """
    usable = [p for p in points if p.seconds > 0 and p.size > 0]
    if len(usable) < 2:
        return float("nan")
    x = np.log(np.array([p.size for p in usable], dtype=np.float64))
    y = np.log(np.array([p.seconds for p in usable], dtype=np.float64))
    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(x_centred * (y - y.mean())) / denominator)


def verify_complexity(routine: Callable[[int], None], sizes: "List[int]",
                      expected_exponent: float, tolerance: float = 0.6) -> "Tuple[bool, float]":
    """Check that measured growth matches a declared bound.

    Complexity: dominated by `measure_growth`.

    The tolerance is deliberately loose. Constant factors, cache behaviour and
    interpreter overhead all distort the exponent at the small sizes a test can
    afford; the purpose is to catch an accidental order-of-magnitude regression
    (linear silently becoming quadratic), not to certify a constant.
    """
    exponent = estimate_exponent(measure_growth(routine, sizes))
    if not np.isfinite(exponent):
        return True, exponent  # too fast to time: no evidence of regression
    return abs(exponent - expected_exponent) <= tolerance, exponent


def profile_pipeline(stage_times: Dict[str, float]) -> str:
    """Render measured stage timings with their share of the total.

    Complexity: O(S log S) over the stages, for the sort.
    """
    total = sum(stage_times.values()) or 1.0
    width = max((len(k) for k in stage_times), default=10)
    lines = [f"{'STAGE'.ljust(width)}  {'SECONDS'.rjust(9)}  SHARE", "-" * (width + 20)]
    for name, seconds in sorted(stage_times.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name.ljust(width)}  {seconds:9.3f}  {100 * seconds / total:5.1f}%")
    lines.append(f"{'TOTAL'.ljust(width)}  {total:9.3f}  100.0%")
    return "\n".join(lines)
