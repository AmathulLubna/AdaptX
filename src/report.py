"""Stage 5 -- the machine-readable repair report.

EXPLAINABILITY IS A SCORED OBJECTIVE, NOT DOCUMENTATION
-------------------------------------------------------
The track spec requires the system to state why it acted. Every field below
answers one of the three questions in the problem statement:

  WHY did it fail?   -> detection.reason + diagnosis.evidence
  WHAT was applied?  -> selected_repair + repair_parameters
  HOW little?        -> intervention_size, parameter_ratio, training_cost

Writing the report is deliberately separated from computing it: the doctor
produces data structures, this module serialises them. That keeps the pipeline
usable as a library (the sandbox may import a function rather than run a script)
and means a change to the submission schema never touches the algorithms.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from .config import Config
from .doctor import CaseResult, aggregate
from .metrics import mean_over_levels


def case_report(case: CaseResult) -> dict:
    """Serialise one scenario's outcome, matching the required output schema."""
    before = mean_over_levels(case.baseline_test)
    after = mean_over_levels(case.repaired_test)
    reference = mean_over_levels(case.reference_test)

    report: Dict[str, object] = {
        "scenario": case.scenario,
        "failure_detected": case.detection.failure_detected,
        "detection": case.detection.as_dict(),
        "diagnosis": case.diagnosis.as_dict(),
        "selected_repair": case.selected_family,
        "families_considered": case.considered,
        "performance_before": round(before, 4),
        "performance_after": round(after, 4),
        "performance_reference": round(reference, 4),
        "ood_recovery": round(case.recovery, 4),
        # Recovery saturates at 1.0 by design; this flag preserves the fact that
        # a warm-start repair sometimes exceeds the full-retrain ceiling.
        "beat_reference": bool(after > reference + 1e-9),
        "per_level_before": {str(k): round(v, 4) for k, v in case.baseline_test.items()},
        "per_level_after": {str(k): round(v, 4) for k, v in case.repaired_test.items()},
        "n_evaluations": case.n_evaluations,
        "convergence_log": case.convergence,
        # Ground truth is attached for the evaluator's convenience only, AFTER
        # all decisions are made. Nothing in the pipeline reads it.
        "ground_truth_cause": case.hidden_cause,
        "diagnosis_correct": case.diagnosis_correct,
    }

    if case.candidate is not None:
        report.update({
            "repair_parameters": case.candidate.description,
            "intervention_size": round(case.candidate.intervention_size, 5),
            "parameter_ratio": round(case.candidate.param_ratio, 4),
            "training_cost_gflops": round(case.candidate.train_cost, 5),
            "stability": round(case.candidate.stability, 4),
            "fairness_gap": round(case.candidate.fairness, 4),
        })
    else:
        report.update({
            "repair_parameters": {"repair": "none",
                                  "rationale": "no meaningful degradation detected"},
            "intervention_size": 0.0,
            "parameter_ratio": 1.0,
            "training_cost_gflops": 0.0,
            "stability": round(case.detection.stability, 4),
            "fairness_gap": round(case.detection.fairness_gap, 4),
        })
    return report


def build_report(results: List[CaseResult], cfg: Config,
                 runtime_seconds: float) -> dict:
    """Assemble the full submission payload."""
    return {
        "system": "AdaptX",
        "team_id": os.environ.get("TEAM_ID", "unset"),
        "config": {
            "seed": cfg.seed,
            "pop_size": cfg.search.pop_size,
            "generations": cfg.search.generations,
            "validation_levels": list(cfg.data.val_levels),
            "hidden_test_levels": list(cfg.data.test_levels),
            "max_patch_fraction": cfg.repair.max_patch_fraction,
            "param_budget_ratio": cfg.repair.param_budget_ratio,
            "min_accuracy_gain": cfg.repair.min_accuracy_gain,
        },
        "summary": aggregate(results),
        "runtime_seconds": round(runtime_seconds, 3),
        "cases": [case_report(case) for case in results],
    }


def write_report(payload: dict, path: "str | Path") -> Path:
    """Write the report as indented JSON and return the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return target


def print_summary(payload: dict) -> None:
    """Human-readable console summary, including the convergence trace.

    The track spec requires convergence/iteration logs to be produced, so the
    last generation of each search is echoed here rather than hidden in the JSON.
    """
    summary = payload["summary"]
    print("=" * 72)
    print("AdaptX -- benchmark summary")
    print("=" * 72)
    for case in payload["cases"]:
        mark = "OK " if case["diagnosis_correct"] else "XX "
        print(f"{mark}{case['scenario']:<18} "
              f"detected={str(case['failure_detected']):<5} "
              f"dx={case['diagnosis']['primary']:<20} "
              f"truth={case['ground_truth_cause']:<20} "
              f"repair={case['selected_repair']}")
        print(f"     acc {case['performance_before']:.3f} -> {case['performance_after']:.3f} "
              f"(ceiling {case['performance_reference']:.3f}) "
              f"recovery={case['ood_recovery']:.3f} "
              f"intervention={case['intervention_size']:.4f}")
        if case["convergence_log"]:
            last = case["convergence_log"][-1]
            print(f"     converged: gen={last['generation']} front={last['front_size']} "
                  f"best_obj={last['best_per_objective']} evals={last['n_evaluations']}")
    print("-" * 72)
    for key, value in summary.items():
        print(f"{key:<26} {value}")
    print("-" * 72)
    print(f"FINAL FITNESS SCORE: {summary['fitness']:.4f}")
    print(f"runtime_seconds:     {payload['runtime_seconds']}")
    print("=" * 72)
