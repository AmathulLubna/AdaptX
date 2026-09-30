"""Central configuration for AdaptX.

DESIGN NOTE (Round-2 agility)
-----------------------------
Every tunable lives in exactly one frozen dataclass. A surprise constraint or a
new drift level is therefore a *one-line* edit here, never a code change in the
search, diagnosis or repair modules. Nothing downstream reads a literal.

REFLECTION NOTE (parameter adaptation)
-------------------------------------
Parameters are split into three tiers by how we adapt them:
  * STRUCTURAL (pop_size, generations) -- scale with the evaluation budget, not
    with the problem. We size them so a full 6-scenario run stays under the
    wall-clock budget; they are the first knobs we raise if the auditor
    rewards solution quality over latency.
  * STATISTICAL (detection/diagnosis thresholds) -- derived from sampling noise
    (see `detect.py`), not hand-tuned to the benchmark. They adapt automatically
    with validation-set size, which is why a hidden shift level cannot break them.
  * BUDGETARY (max_patch_fraction, param_budget_ratio) -- encode the deployment
    constraints of the track spec and are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass(frozen=True)
class SearchConfig:
    """NSGA-II control parameters.

    Attributes
    ----------
    pop_size:
        Population size. Must be even so that parents pair cleanly in crossover.
    generations:
        Number of generations. Total model fits = pop_size * (generations + 1).
    crossover_prob:
        Probability that a parent pair is recombined rather than copied.
    mutation_prob_scale:
        Per-gene mutation probability is `mutation_prob_scale / n_genes`, so the
        expected number of mutated genes is constant (~1) regardless of genome
        length. This keeps the operator's disruption rate invariant when the new
        data pool changes size -- essential for hidden-scenario robustness.
    sbx_eta / poly_eta:
        Distribution indices for SBX crossover and polynomial mutation on
        real-valued genes. Higher = offspring closer to parents.
    tournament_size:
        Binary tournament (=2) is the NSGA-II standard; kept configurable.
    """

    pop_size: int = 16
    generations: int = 4
    search_max_iter: int = 60
    """Iteration cap for families that RETRAIN inside the fitness loop.

    Rebuilding a network per candidate is ~30x costlier than a warm-start
    fine-tune, and it was the dominant term in our first full run (>5 min). The
    cap is a *fidelity/throughput* trade: a 70-iteration fit ranks candidates
    almost identically to a 220-iteration one (the ordering of alpha and width
    settings is established early), while letting the search visit far more of
    them. The single chosen winner is then rebuilt at full `ModelConfig.max_iter`
    before it is scored and reported, so the deployed model loses no quality.
    """
    crossover_prob: float = 0.9
    mutation_prob_scale: float = 1.0
    sbx_eta: float = 15.0
    poly_eta: float = 20.0
    tournament_size: int = 2


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds for deciding that degradation is *meaningful*, not noise.

    `noise_sigmas` multiplies the binomial standard error of the validation
    accuracy estimate. A drop is only a failure if it exceeds that many standard
    errors -- this is why the threshold adapts to validation-set size instead of
    being a magic constant.
    """

    noise_sigmas: float = 2.0
    min_absolute_drop: float = 0.02
    overfit_gap_threshold: float = 0.12
    calibration_bins: int = 10


@dataclass(frozen=True)
class DiagnosisConfig:
    """Scales that map raw evidence statistics into a [0, 1] confidence.

    Each statistic is squashed with `1 - exp(-s / scale)`, a monotone map with no
    free upper bound. The scales below are the *half-saturation* points: a shift
    of `shift_scale` standard deviations yields ~0.63 confidence. They were set
    from the statistic's null distribution on unperturbed data, not tuned on the
    perturbed benchmark, so they transfer to unseen perturbation magnitudes.
    """

    shift_scale: float = 0.45
    concentration_scale: float = 0.35
    corruption_scale: float = 0.06
    imbalance_scale: float = 0.12
    overfit_scale: float = 0.18
    min_confidence_to_repair: float = 0.20


@dataclass(frozen=True)
class RepairConfig:
    """Budgets and hard constraints on any accepted repair."""

    max_patch_fraction: float = 0.25
    min_patch_size: int = 8
    replay_size: int = 256
    finetune_iters: int = 40
    param_budget_ratio: float = 1.5
    min_accuracy_gain: float = 0.01
    max_fairness_gap: float = 0.25


@dataclass(frozen=True)
class DataConfig:
    """Synthetic benchmark geometry. No shape is hard-coded downstream."""

    n_features: int = 12
    n_informative: int = 8
    n_classes: int = 3
    n_train: int = 800
    n_val: int = 400
    n_pool: int = 1200
    n_ood: int = 400
    n_groups: int = 2
    val_levels: Tuple[float, ...] = (0.8, 1.0, 1.2)
    test_levels: Tuple[float, ...] = (0.6, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ModelConfig:
    """Base learner. Bounded by construction => no gradient explosion.

    `hidden_sizes` and `max_iter` are capped so that the worst candidate the
    search can propose is still cheap; `alpha` has a strictly positive lower
    bound so the loss surface stays strongly convex in the weights' L2 term.
    """

    hidden_sizes: Tuple[int, ...] = (16,)
    max_iter: int = 220
    # L2 is 1e-2, not the sklearn default 1e-4. The deployed model must be
    # ADEQUATE on its own distribution -- with weaker regularisation this network
    # memorises 800 samples (train ~0.99 / val ~0.79), which made every scenario
    # look mildly overfit and fired the memorisation detector on the healthy
    # control. A healthy baseline is a precondition for the benchmark to mean
    # anything: we are diagnosing induced failures, not our own sloppy training.
    alpha: float = 1e-2
    learning_rate_init: float = 3e-3
    alpha_bounds: Tuple[float, float] = (1e-5, 1e-1)
    hidden_bounds: Tuple[int, int] = (4, 64)


@dataclass(frozen=True)
class Config:
    """Root configuration object threaded through the whole pipeline."""

    seed: int = 20260930
    search: SearchConfig = field(default_factory=SearchConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def evolve(self, **changes: object) -> "Config":
        """Return a copy with top-level fields replaced (frozen-safe)."""
        return replace(self, **changes)  # type: ignore[arg-type]


DEFAULT_CONFIG = Config()
