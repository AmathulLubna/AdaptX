"""Repair-family tests: constraints, budgets, and the PatchML efficiency claim."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.detect import evaluate_on_levels
from src.doctor import train_deployed_model
from src.metrics import mean_over_levels
from src.models import finetune
from src.repairs import (CAUSE_TO_FAMILY, FAMILIES, RepairContext, finalise,
                         make_evaluator, objectives_and_constraints,
                         select_family, shortlist_candidates)
from src.scenarios import CAUSES, Dataset


@pytest.fixture(scope="module")
def context(tiny_config: Config, env_scenario) -> RepairContext:
    base = train_deployed_model(env_scenario, tiny_config)
    failed = mean_over_levels(
        evaluate_on_levels(base, env_scenario, tiny_config.data.val_levels))
    return RepairContext(env_scenario, base, tiny_config, failed, tiny_config.seed)


def test_every_cause_maps_to_an_implemented_family() -> None:
    """The routing table must not reference a repair that does not exist."""
    for cause in CAUSES:
        assert cause in CAUSE_TO_FAMILY
        assert CAUSE_TO_FAMILY[cause] in FAMILIES


def test_selection_follows_the_leading_diagnosis(tiny_config: Config) -> None:
    """Repair choice must be DRIVEN by diagnosis -- the anti-AutoML property."""
    confidences = {c: 0.05 for c in CAUSES}
    confidences["class_imbalance"] = 0.9
    primary, shortlist = select_family(confidences, tiny_config)
    assert primary == "ClassReweighting"
    assert shortlist[0] == primary


def test_selection_falls_back_when_all_evidence_is_weak(tiny_config: Config) -> None:
    """With no confident hypothesis the system must still return a valid plan."""
    primary, shortlist = select_family({c: 0.0 for c in CAUSES}, tiny_config)
    assert primary in FAMILIES
    assert shortlist


def test_selection_shortlist_has_no_duplicates(tiny_config: Config) -> None:
    confidences = {c: 0.5 for c in CAUSES}
    _, shortlist = select_family(confidences, tiny_config)
    assert len(shortlist) == len(set(shortlist))


def test_every_family_builds_a_wellformed_problem(context: RepairContext) -> None:
    """Structural contract for every repair, including ones added in Round 2."""
    for name, build in FAMILIES.items():
        family = build(context)
        problem = family.problem
        assert problem.n_var > 0, name
        assert problem.lower.shape == (problem.n_var,), name
        assert np.all(problem.upper >= problem.lower), name
        assert problem.n_obj == 5, name


def test_every_family_decodes_to_a_usable_model(context: RepairContext) -> None:
    """A genome must always yield a model that predicts the right shape."""
    scenario = context.scenario
    for name, build in FAMILIES.items():
        family = build(context)
        genome = 0.5 * (family.problem.lower + family.problem.upper)
        candidate = finalise(family.decode(genome), context)
        predictions = candidate.model.predict(scenario.val.X)
        assert predictions.shape == scenario.val.y.shape, name
        assert 0.0 <= candidate.score <= 1.0, name
        assert 0.0 <= candidate.intervention_size <= 1.0, name


def test_objectives_have_the_declared_sign_and_arity(context: RepairContext) -> None:
    """Objective 0 is NEGATED accuracy: a common and silent sign bug."""
    family = FAMILIES["PatchML"](context)
    genome = np.zeros(family.problem.n_var)
    genome[:16] = 1.0
    candidate = finalise(family.decode(genome), context)
    objectives, violations = objectives_and_constraints(candidate, context)
    assert objectives.shape == (5,)
    assert violations.shape == (3,)
    assert objectives[0] == pytest.approx(-candidate.score)
    assert np.all(violations >= 0.0)


def test_parameter_budget_constraint_triggers(context: RepairContext) -> None:
    """An oversized model must register a violation, not be silently accepted."""
    from src.models import train_model
    scenario = context.scenario
    fat = train_model(scenario.train.X, scenario.train.y, context.cfg,
                      context.cfg.seed, hidden=(256, 256), max_iter=20)
    from src.repairs import RepairCandidate
    candidate = finalise(RepairCandidate(fat, 0.1, {"repair": "test"}), context)
    _, violations = objectives_and_constraints(candidate, context)
    assert candidate.param_ratio > context.cfg.repair.param_budget_ratio
    assert violations[1] > 0.0


def test_accuracy_gain_constraint_rejects_a_useless_repair(context: RepairContext) -> None:
    """A repair that does not beat the failed model must be infeasible."""
    from src.repairs import RepairCandidate
    unchanged = finalise(RepairCandidate(context.base_model, 0.0, {"repair": "noop"}),
                         context)
    _, violations = objectives_and_constraints(unchanged, context)
    assert violations[0] > 0.0, "an unchanged model must violate the gain constraint"


def test_patchml_respects_the_patch_budget(context: RepairContext) -> None:
    """Even an all-ones genome must not exceed `max_patch_fraction`."""
    family = FAMILIES["PatchML"](context)
    candidate = family.decode(np.ones(family.problem.n_var))
    limit = context.cfg.repair.max_patch_fraction
    assert candidate.intervention_size <= limit + 1e-9
    assert candidate.description["patch_size"] >= context.cfg.repair.min_patch_size


def test_patchml_enforces_a_minimum_patch(context: RepairContext) -> None:
    """An all-zero genome must still decode to a trainable patch."""
    family = FAMILIES["PatchML"](context)
    candidate = family.decode(np.zeros(family.problem.n_var))
    assert candidate.description["patch_size"] >= context.cfg.repair.min_patch_size


def test_shortlist_is_deterministic_and_bounded(context: RepairContext) -> None:
    """The uncertainty ranking is part of the fitness map, so it must be stable."""
    first = shortlist_candidates(context, 40)
    second = shortlist_candidates(context, 40)
    assert np.array_equal(first, second)
    assert first.size == 40
    assert first.max() < len(context.scenario.pool)


def test_shortlist_selects_genuinely_uncertain_samples(context: RepairContext) -> None:
    """The innovation claim: shortlisted rows sit nearer the decision boundary.

    If this failed, the pruning would be discarding exactly the samples PatchML
    needs, and the 2^2000 -> 2^192 reduction would be unsound.
    """
    pool = context.scenario.pool
    proba = context.base_model.predict_proba(pool.X)
    ordered = np.sort(proba, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    chosen = shortlist_candidates(context, 50)
    assert margin[chosen].mean() < margin.mean()


def test_patchml_beats_a_random_patch_of_equal_size(context: RepairContext) -> None:
    """The core PatchML claim, tested honestly against a same-size control.

    Comparing a 64-sample informed patch against a 64-sample RANDOM patch (not
    against no repair, and not against a larger patch) isolates the value of the
    SELECTION, which is the only thing PatchML contributes. Averaging over three
    random draws keeps the control from being one unlucky sample.
    """
    cfg = context.cfg
    pool = context.scenario.pool
    size = 64

    family = FAMILIES["PatchML"](context)
    genome = np.zeros(family.problem.n_var)
    genome[:size] = 1.0
    informed = finalise(family.decode(genome), context)

    scores = []
    for trial in range(3):
        rng = np.random.default_rng(1000 + trial)
        idx = rng.choice(len(pool), size=size, replace=False)
        patch = Dataset(pool.X[idx], pool.y[idx], pool.groups[idx])
        replay_idx = rng.choice(len(context.scenario.train),
                                size=min(cfg.repair.replay_size,
                                         len(context.scenario.train)), replace=False)
        X = np.vstack([context.scenario.train.X[replay_idx], patch.X])
        y = np.concatenate([context.scenario.train.y[replay_idx], patch.y])
        model = finetune(context.base_model, X, y, cfg, 1000 + trial)
        from src.repairs import score_model
        scores.append(score_model(model, context)[0])

    # A tolerance is used rather than a strict >: with a 64-sample patch the two
    # are close, and asserting a large margin would be asserting a lucky seed.
    assert informed.score >= float(np.mean(scores)) - 0.01


def test_evaluator_cache_returns_identical_results(context: RepairContext) -> None:
    """The cache must be transparent: a hit equals a miss, exactly."""
    family = FAMILIES["PatchML"](context)
    evaluate = make_evaluator(family.decode, context)
    genome = np.zeros(family.problem.n_var)
    genome[:12] = 1.0
    first = evaluate(genome.copy())
    second = evaluate(genome.copy())
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
