"""The AdaptX closed loop: Detect -> Diagnose -> Select -> Evolve -> Validate.

This module is the only place the five stages meet. Every stage is imported, none
is implemented here -- so a Round-2 patch touches one leaf module and this file
keeps working unchanged.

KNEE SELECTION
--------------
NSGA-II returns a front, but a submission must name one model. We pick the knee
by a normalised weighted Chebyshev (augmented Tchebycheff) scalarisation of the
front only -- never of the search itself. Scalarising after the fact keeps the
search free of arbitrary weights (which is what makes it a genuine multi-
objective method) while still producing a single defensible deployment choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .detect import DetectionReport, detect_failure, evaluate_on_levels
from .determinism import stable_hash
from .diagnose import Diagnosis, diagnose
from .metrics import mean_over_levels, ood_recovery, prediction_stability
from .models import FittedModel, train_model
from .nsga2 import Result, minimize
from .repairs import (FAMILIES, N_OBJECTIVES, RepairCandidate, RepairContext,
                      decode_final, finalise, select_family)
from .scenarios import Scenario
from .validation import (summarise_failure, validate_array, validate_labels,
                         validate_matched_lengths)

# Deployment preference over the five objectives. Recovery dominates; the rest
# break ties toward the cheaper, smaller, steadier repair. These weights affect
# ONLY the final pick from the front, never the search.
KNEE_WEIGHTS = np.array([0.55, 0.15, 0.10, 0.08, 0.12])


@dataclass
class CaseResult:
    """Everything AdaptX concluded about one scenario."""

    scenario: str
    hidden_cause: str
    detection: DetectionReport
    diagnosis: Diagnosis
    selected_family: str
    candidate: "RepairCandidate | None"
    baseline_test: Dict[float, float]
    repaired_test: Dict[float, float]
    reference_test: Dict[float, float]
    convergence: List[dict] = field(default_factory=list)
    n_evaluations: int = 0
    considered: List[str] = field(default_factory=list)

    @property
    def diagnosis_correct(self) -> bool:
        """Healthy case: correct iff we declared no failure."""
        if self.hidden_cause == "none":
            return not self.detection.failure_detected
        return self.detection.failure_detected and self.diagnosis.primary == self.hidden_cause

    @property
    def recovery(self) -> float:
        """Fraction of lost performance restored, on the HIDDEN test levels."""
        failed = mean_over_levels(self.baseline_test)
        repaired = mean_over_levels(self.repaired_test)
        reference = mean_over_levels(self.reference_test)
        return ood_recovery(failed, repaired, reference)


def train_deployed_model(scenario: Scenario, cfg: Config) -> FittedModel:
    """Fit the model that is already 'in production' for this scenario.

    The overfitting scenario deliberately gets an over-capacity, barely-
    regularised network on a tiny training set. That is the failure under test,
    not a bug -- the benchmark must contain a genuinely sick patient.
    """
    if scenario.overparameterized:
        return train_model(scenario.train.X, scenario.train.y, cfg, cfg.seed,
                           hidden=(128, 64), alpha=1e-6, max_iter=cfg.model.max_iter)
    return train_model(scenario.train.X, scenario.train.y, cfg, cfg.seed)


def train_reference_model(scenario: Scenario, cfg: Config) -> FittedModel:
    """The ceiling: retrain on the ENTIRE new pool plus the original data.

    This is the honest upper bound for recovery -- what an engineer would get by
    throwing all available new data at the problem with no cleverness. PatchML
    claims to approach it with a fraction of the samples, so the comparison must
    be to this, not to a weaker baseline.

    Applying the same ceiling definition to every scenario (including input
    corruption) removes the inconsistency where one scenario was scored against a
    reference that was not actually achievable.
    """
    X = np.vstack([scenario.train.X, scenario.pool.X])
    y = np.concatenate([scenario.train.y, scenario.pool.y])
    # Iterations are capped below `ModelConfig.max_iter` because this fit sees
    # several times more data than the deployed model did, and it is run once per
    # scenario purely to establish the ceiling. The cap keeps a reference model
    # from costing more than the entire search it is the yardstick for.
    return train_model(X, y, cfg, cfg.seed + 1, max_iter=min(cfg.model.max_iter, 150))


def _knee(front_objectives: np.ndarray) -> int:
    """Index of the best compromise point on the Pareto front.

    Each objective is min-max normalised across the front so that objectives with
    very different units (accuracy in [0,1], log-cost in [0,20]) contribute
    comparably. Degenerate ranges collapse to zero contribution rather than
    dividing by zero.
    """
    lo = front_objectives.min(axis=0)
    hi = front_objectives.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    normalised = (front_objectives - lo) / span
    return int(np.argmin(normalised @ KNEE_WEIGHTS))


def run_search(family_name: str, context: RepairContext,
               cfg: Config) -> "Tuple[RepairCandidate | None, Result]":
    """Evolve one repair family and return its knee candidate."""
    family = FAMILIES[family_name](context)
    # Per-family operator seed via BLAKE2b, never Python's salted `hash()`:
    # the latter changes between processes and would break reproducibility.
    family_seed = (cfg.seed + stable_hash(family_name.encode("utf-8")) % 100_000)
    result = minimize(
        family.problem,
        pop_size=cfg.search.pop_size,
        generations=cfg.search.generations,
        seed=family_seed,
        crossover_prob=cfg.search.crossover_prob,
        mutation_prob_scale=cfg.search.mutation_prob_scale,
        sbx_eta=cfg.search.sbx_eta,
        poly_eta=cfg.search.poly_eta,
        tournament_size=cfg.search.tournament_size,
        initial_population=family.warm_starts or None,
    )
    genomes, objectives = result.pareto_front()
    if genomes.shape[0] == 0:
        return None, result
    # The knee is rebuilt at full fidelity so the deployed model never inherits
    # the search's reduced training budget.
    best = finalise(decode_final(family, genomes[_knee(objectives)]), context)
    return best, result


def _search_with_recovery(family_name: str, context: RepairContext,
                          cfg: Config) -> "Tuple[RepairCandidate | None, Result]":
    """Run one repair family, degrading gracefully if the search itself fails.

    ERROR RECOVERY. A repair family can fail for reasons that are data-dependent
    and not programmer error: a degenerate patch that leaves a class unseen, a
    singular covariance, an optimiser that cannot converge on corrupted input.
    Those must not abort the whole benchmark, because the remaining scenarios
    are still diagnosable and the fallback family may well succeed.

    The failure is caught, summarised into the trace so a degraded run still
    records WHAT went wrong, and reported as "no candidate" so the caller can
    try the runner-up hypothesis. Programmer errors are deliberately NOT caught:
    only `Exception` subclasses raised during the search are, and a
    `KeyboardInterrupt` or `SystemExit` still propagates.
    """
    try:
        return run_search(family_name, context, cfg)
    except Exception as exc:  # noqa: BLE001 - deliberate recovery boundary
        empty = Result(np.empty((0, 1)), np.empty((0, N_OBJECTIVES)),
                       np.empty((0, 1)), np.empty(0, dtype=np.int64),
                       [{"error": summarise_failure(exc, f"search:{family_name}")}], 0)
        return None, empty


def diagnose_and_repair(scenario: Scenario, cfg: Config) -> CaseResult:
    """Run the full closed loop on one scenario.

    Order of operations is the contract: nothing about the repair is decided
    before detection has fired, and nothing about the family is decided before
    the evidence vector exists.
    """
    # Validate before spending a single model fit. A shape error found here
    # names the offending split; found later it surfaces inside scikit-learn
    # with no indication of which input was wrong.
    validate_array(scenario.train.X, f"{scenario.name}.train.X", ndim=2, min_rows=2)
    validate_labels(scenario.train.y, f"{scenario.name}.train.y", cfg.data.n_classes)
    validate_matched_lengths(train_X=scenario.train.X, train_y=scenario.train.y)
    validate_matched_lengths(pool_X=scenario.pool.X, pool_y=scenario.pool.y)

    base = train_deployed_model(scenario, cfg)
    detection = detect_failure(base, scenario, cfg)
    diagnosis = diagnose(scenario.train, scenario.pool, detection, cfg)

    baseline_test = evaluate_on_levels(base, scenario, cfg.data.test_levels)

    if not detection.failure_detected:
        # DO NO HARM. A healthy model is returned untouched; we do not spend a
        # single evaluation, and the repaired scores are the baseline scores.
        return CaseResult(scenario.name, scenario.hidden_cause, detection, diagnosis,
                          "none", None, baseline_test, dict(baseline_test),
                          dict(baseline_test), [], 0, [])

    reference = train_reference_model(scenario, cfg)
    reference_test = evaluate_on_levels(reference, scenario, cfg.data.test_levels)

    context = RepairContext(scenario, base, cfg,
                            failed_score=mean_over_levels(
                                evaluate_on_levels(base, scenario, cfg.data.val_levels)),
                            seed=cfg.seed)

    primary, shortlist = select_family(diagnosis.confidences, cfg)
    best, result = _search_with_recovery(primary, context, cfg)

    # Fallback: if the leading hypothesis yields nothing feasible, try the next
    # one. A diagnosis is a belief, so the system must survive being wrong.
    considered = [primary]
    for alternative in shortlist[1:2]:
        if best is not None:
            break
        considered.append(alternative)
        best, result = _search_with_recovery(alternative, context, cfg)
        primary = alternative

    if best is None:
        return CaseResult(scenario.name, scenario.hidden_cause, detection, diagnosis,
                          "none", None, baseline_test, dict(baseline_test),
                          reference_test, result.history, result.n_evaluations,
                          considered)

    repaired_test = evaluate_on_levels(best.model, scenario, cfg.data.test_levels)
    return CaseResult(scenario.name, scenario.hidden_cause, detection, diagnosis,
                      primary, best, baseline_test, repaired_test, reference_test,
                      result.history, result.n_evaluations, considered)


def run_benchmark(scenarios: List[Scenario], cfg: Config) -> List[CaseResult]:
    """Run every scenario. Sequential and ordered, hence reproducible."""
    return [diagnose_and_repair(scenario, cfg) for scenario in scenarios]


def aggregate(results: List[CaseResult]) -> Dict[str, float]:
    """Headline numbers across the benchmark -- the submitted fitness score.

    The composite `fitness` mirrors the track's stated objectives: diagnosis
    correctness and OOD recovery carry the most weight, with data efficiency and
    stability as the efficiency terms. It is reported alongside its components so
    nothing is hidden inside a single number.
    """
    repaired = [r for r in results if r.candidate is not None]
    diagnosis_accuracy = float(np.mean([r.diagnosis_correct for r in results]))
    recovery = float(np.mean([r.recovery for r in results])) if results else 0.0
    before = float(np.mean([mean_over_levels(r.baseline_test) for r in results]))
    after = float(np.mean([mean_over_levels(r.repaired_test) for r in results]))
    intervention = float(np.mean([r.candidate.intervention_size for r in repaired])) if repaired else 0.0
    stability = float(np.mean([
        prediction_stability(list(r.repaired_test.values())) for r in results]))
    params = float(np.mean([r.candidate.param_ratio for r in repaired])) if repaired else 1.0

    fitness = (0.35 * diagnosis_accuracy + 0.35 * min(recovery, 1.0)
               + 0.15 * stability + 0.15 * (1.0 - min(intervention, 1.0)))
    return {
        "diagnosis_accuracy": round(diagnosis_accuracy, 4),
        "mean_ood_recovery": round(recovery, 4),
        "mean_accuracy_before": round(before, 4),
        "mean_accuracy_after": round(after, 4),
        "mean_intervention_size": round(intervention, 4),
        "mean_stability": round(stability, 4),
        "mean_parameter_ratio": round(params, 4),
        "n_repaired": len(repaired),
        "n_cases": len(results),
        "fitness": round(fitness, 4),
    }
