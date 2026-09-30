# AdaptX

**OptiForge 2026 — Track 04 (IEEE CIS): Multi-Objective Heuristics + Deep Evolutionary Networks**
Team ID: `OPT-26-5711`

Autonomous diagnosis and multi-objective evolutionary **repair** of failing ML models.

> **Principle: optimise interventions, not models.**
> Do not find a better model. Determine *why* the deployed model failed, and find the
> smallest effective change that restores reliable performance.

---

## 0. UN SDG alignment — SDG 3: Good Health and Well-Being

**Targets 3.4** (reduce premature mortality from non-communicable diseases) and
**3.8** (universal health coverage).

Clinical ML models fail silently. A sepsis-risk or triage model trained at one
hospital degrades when deployed at another — different sensors, different case
mix, different population — and the failure surfaces as worse patient outcomes
long before anyone retrains it. The barrier is not detection alone but **repair
cost**: retraining needs newly labelled clinical data, and expert annotation is
the scarcest resource in the system.

AdaptX addresses both halves:

| Algorithm output | Clinical consequence |
|---|---|
| Diagnosis evidence vector (`diagnose.py`) | Says *why* the model degraded — sensor drift vs case-mix change vs data-quality failure — so the fix targets the real cause instead of triggering a blind retrain. |
| PatchML minimal repair set (`repairs.py`) | **29 of 1200 samples** in our benchmark: a 97.6% reduction in what must be expertly labelled. That is the difference between a model being repaired and abandoned. |
| Fairness constraint (`objectives_and_constraints`) | A repair **cannot** be accepted if it widens the accuracy gap between sub-populations beyond the cap — target 3.8 expressed as a hard constraint, not a report. |
| Calibration in detection (`detect.py`) | A confidently wrong clinical model is more dangerous than an uncertain one, so expected calibration error is a first-class detection signal. |
| Repair report (`report.py`) | An auditable record of what changed and why — a precondition for any clinical deployment review. |

**Stated plainly:** the benchmark is synthetic and no clinical claim is made or
implied. What is demonstrated is the mechanism, on data whose ground-truth
failure cause is known so diagnostic accuracy can actually be measured.


---

## Quick start

```bash
pip install -r requirements.txt
python main.py                  # full benchmark, writes reports/latest.json
python -m pytest                # 80-test suite
python submission.py --test     # 58 self-tests, no pytest required
```

**Two entrypoints, one source of truth.** `main.py` runs the modular package in
`src/`. `submission.py` is that same package flattened into one self-contained
file by `tools/build_submission.py`, because the evaluation portal executes a
single file — it carries the validation suite inside it and produces
byte-identical results. Regenerate it with `python tools/build_submission.py`
after any change to `src/`.

Writes `reports/latest.json` and prints the final fitness score plus convergence logs.

```bash
python main.py --fast          # reduced budget smoke test (~1 min)
python main.py --seed 777      # reproducibility check
python -m pytest               # full test suite
```

No secrets, no network access, no frontend. Dependencies: `numpy`, `scipy`,
`scikit-learn`. **NSGA-II is implemented from scratch in `src/nsga2.py`** — there is
no `pymoo` dependency.

---

## The closed loop

```
Original model ──► New / OOD / perturbed data
                            │
                            ▼
                   [1] Failure detection        src/detect.py
                            │
                            ▼
                   [2] Failure diagnosis        src/diagnose.py
                            │      (evidence vector over 5 causes)
                            ▼
                   [3] Repair selection         src/repairs.py
                            │      (diagnosis narrows the search space)
                            ▼
                   [4] NSGA-II evolution        src/nsga2.py
                            │      (5 objectives, 3 constraints)
                            ▼
                   [5] Validation on HIDDEN     src/doctor.py
                       drift levels
                            │
                            ▼
                   [6] Repair report (JSON)     src/report.py
```

| Module | Responsibility |
|---|---|
| `src/config.py` | Every tunable, in one frozen dataclass tree |
| `src/determinism.py` | Thread pinning, stable hashing, content-addressed seeding |
| `src/scenarios.py` | Synthetic benchmark; 5 failure modes + healthy control |
| `src/models.py` | MLP wrapper, warm-start fine-tuning, cost accounting |
| `src/metrics.py` | Accuracy, calibration, fairness, stability, recovery |
| `src/detect.py` | Stage 1 — is the degradation meaningful? |
| `src/diagnose.py` | Stage 2 — evidence vector over causes |
| `src/repairs.py` | Stages 3–4 — five repair families as search problems |
| `src/nsga2.py` | The evolutionary engine (ours, not a library) |
| `src/doctor.py` | Orchestration and knee selection |
| `src/report.py` | Stage 6 — machine-readable repair report |

---

## 1. Representation (what a chromosome *is*)

Every repair family encodes a candidate as a real vector `x ∈ [l, u]^n` plus a
per-gene type tag (`BINARY` or `REAL`). One engine searches all of them.

| Family | Genome | Length | Genes |
|---|---|---|---|
| **PatchML** | subset-selection mask over shortlisted new samples | 192 | binary |
| FeatureReweighting | per-feature gain in `[0, 2]` | `n_features` | real |
| ClassReweighting | per-class importance in `[0.2, 5]` | `n_classes` | real |
| RobustPreprocessing | `[clip_σ, augment_σ, patch_fraction]` | 3 | real |
| Regularisation | `[log₁₀α, width₁, width₂, pool_fraction]` | 4 | real |

**Why binary for PatchML.** The question *"which samples repair the model?"* is
literally a subset-selection problem, so the mask is the natural encoding.
Alternatives were rejected: a real-valued per-sample score needs an arbitrary
threshold, and an integer count-plus-ranking cannot express *which* samples, only
*how many*.

---

## 2. Operators

| Operator | Real genes | Binary genes | Why |
|---|---|---|---|
| Crossover | SBX (η=15) | Uniform | A patch mask has no locus ordering — pool index *i* and *i+1* are unrelated — so a positional (one/two-point) operator would impose structure that does not exist. |
| Mutation | Polynomial (η=20) | Bit-flip | Polynomial mutation is bounded and self-adaptive in step size near the box edges. |
| Selection | Binary tournament on (violation, rank, −crowding) | | Standard NSGA-II; rank first, density second. |
| Survival | Elitist (μ+λ) truncation | | Guarantees monotone front improvement. |

**Mutation rate is `1 / n_genes`**, not a constant. The expected number of mutated
genes is therefore ≈1 regardless of genome length, so the operator's disruption
rate stays invariant when the pool size changes — a Round-2 robustness property.

---

## 3. Objectives and constraints

Five objectives, all **minimised**:

| # | Objective | Meaning |
|---|---|---|
| 0 | `−mean OOD score` | generalisation recovery |
| 1 | `intervention size` | how much of the system we touched, in [0,1] |
| 2 | `log(1 + training cost)` | FLOP-proportional, hardware-independent |
| 3 | `parameter ratio` | deployment footprint vs original |
| 4 | `1 − stability` | variance across validation drift levels |

Three hard constraints, handled by **constraint-domination** (no penalty weights):

1. must beat the failed model by ≥ `min_accuracy_gain` (0.01)
2. parameter count ≤ `param_budget_ratio` × original (1.5×)
3. sub-population accuracy gap ≤ `max_fairness_gap` (0.25)

**Why no weighted sum during search.** Scalarising early bakes in a preference the
jury can challenge, and it cannot find non-convex regions of the front. We search
the true front, then pick one deployment point from it by a normalised weighted
Chebyshev scalarisation (`doctor.KNEE_WEIGHTS`). Weights therefore affect only the
final pick, never the exploration.

**Cost is not wall-clock time.** Wall-clock varies with machine load, which would
make fitness non-deterministic. We use `n_samples × n_iterations × n_params`,
proportional to real FLOPs and a pure function of the configuration.

---

## 4. PatchML — the minimal-data repair

When diagnosis points at environment shift, PatchML answers:

> *What is the smallest subset of newly available data that repairs the model?*

**The problem.** A naive genome is one bit per pool sample — 2000 genes for a
2000-row pool. No 5-generation search explores 2^2000.

**Our contribution — diagnosis-guided candidate pruning.** We first rank the pool
by the base model's predictive **margin** (gap between its top-two class
probabilities) and keep the 192 least-confident rows. Those are the samples lying
nearest the decision boundary the shift has displaced; confidently-correct rows
carry almost no repair gradient. This cuts the space from 2^2000 to 2^192 while
retaining the highest-value samples.

Ranking uses only the model's own outputs — never labels, never the hidden cause —
so it is legitimate at deployment time. `tests/test_repairs.py` verifies that
shortlisted samples really do have below-average margin.

**Warm starts.** Generation 0 is seeded with prefix masks of the uncertainty
ranking (8, 16, 32, 64, 128 samples). The search begins from a strong front rather
than finding one, which is why a 5-generation budget suffices.

**Replay.** Every fine-tune mixes in original training samples. Without replay,
adapting on a small shifted patch causes catastrophic forgetting, and a class can
be absent from the batch entirely (which `partial_fit` cannot handle). Replay is a
correctness requirement, not just a regulariser.

---

## 5. Determinism — how it is guaranteed

A multi-objective search is reproducible only if **both** halves are: the sampler
that proposes genomes and the evaluator that scores them. Our earlier prototype
was deterministic for a single evaluation and for generation 1, then diverged
later. That signature has exactly three causes, and all three are closed:

1. **Shared global RNG stream.** If evaluation draws from the same global
   `np.random` stream as the operators, any change in how many draws an evaluation
   makes shifts the stream for every later operator call — invisible in generation
   1, fatal afterwards.
   **Fix:** `np.random.Generator` objects threaded explicitly; the global stream is
   never touched; each evaluation is seeded from a BLAKE2b hash of the *genome
   contents* (`determinism.genome_seed`).

2. **Non-associative floating-point reduction under threaded BLAS.** Multi-threaded
   OpenMP splits dot products across threads, so summation order depends on thread
   scheduling and identical inputs can give bit-different weights. SGD amplifies
   this until it flips dominance comparisons between near-tied candidates.
   **Fix:** `pin_threads(1)` runs as the first statement of `main.py`, before NumPy
   is imported.

3. **Unstable sorts and set iteration.** Ties in crowding distance resolved by an
   unstable sort give different survivors run to run.
   **Fix:** every sort passes `kind="stable"`; no set/dict iteration order reaches a
   decision.

Plus an **evaluation cache** keyed on genome bytes: an elite surviving into the
next generation is never re-fitted, so it cannot drift even by one ulp.

**Result:** `f: G → R⁵` is a genuine mathematical function on genome space, not a
random variable. Tested in `tests/test_determinism.py` over **20 generations**,
including a test that deliberately desynchronises the global RNG between runs.

---

## 6. Convergence argument (for the jury)

Let `P_t` be the population at generation *t*, and `F_0(P)` its first
non-dominated front.

1. **Elitism.** Survival truncates the combined pool `P_t ∪ Q_t` (parents plus
   offspring), never offspring alone. A member of `F_0(P_t)` can only be displaced
   by a solution that dominates it.
2. Therefore `min_i f_k(P_t)` is non-increasing in *t* for every objective *k*, and
   the front's hypervolume is monotone non-decreasing. Asserted directly in
   `tests/test_nsga2.py::test_elitism_never_loses_the_best_objective_value`.
3. **Determinism.** With `f` a fixed function (§5) and a fixed operator stream, the
   whole search is a deterministic dynamical system on population space — the
   trajectory is reproducible bit for bit.
4. **Bounded search.** Total model fits = `pop_size × (generations + 1)`, known in
   advance. No adaptive stopping, so runtime is predictable.
5. **No gradient explosion** (track hard constraint 2), by three structural guards
   rather than runtime checks: inputs standardised by a scaler fitted on the
   *original* data; `alpha` bounded strictly positive, making the objective
   strongly convex in the weight-norm term; `max_iter` and `learning_rate_init`
   capped, with Adam normalising step size by its second-moment estimate.

We claim **convergence to a stable non-dominated set under a fixed budget**, not
convergence to the true Pareto front. Claiming the latter for a finite-population
stochastic search would be indefensible.

---

## 7. Why this is not AutoML

Generic AutoML asks *"which configuration scores highest?"* AdaptX asks
*"why did this trained model fail, and what targeted intervention restores it?"*

Four enforced differences:

1. **Diagnosis drives the search space.** `CAUSE_TO_FAMILY` routes each diagnosed
   cause to one repair family. We never search all five blindly.
2. **Intervention size is a first-class objective.** AutoML has no notion of
   "change as little as possible".
3. **Repairs adapt, they do not replace.** Four of five families warm-start from the
   deployed weights. Only `Regularisation` rebuilds — because excess capacity is a
   property of the hypothesis class and no fine-tune removes it.
4. **Do no harm.** A healthy model is returned untouched, with zero evaluations
   spent. The control scenario exists to punish false positives, and
   `tests/test_pipeline.py::test_healthy_model_is_returned_unmodified` enforces it.

---

## 8. Robustness to hidden perturbations

- **Search sees levels 0.8 / 1.0 / 1.2. Scoring uses 0.6 / 1.0 / 1.5 / 2.0.** The
  fitness function never touches the hidden levels.
- Objective 4 (stability) explicitly rewards repairs that are *flat* across drift
  magnitude — that is what makes them extrapolate.
- Every diagnostic statistic is squashed by `1 − exp(−s/scale)`, a **monotone** map.
  A stronger unseen perturbation is further along the same curve, never off a
  tuned cliff. Tested in
  `test_diagnosis.py::test_monotone_confidence_in_perturbation_strength`.
- The concentration statistic is **scale-invariant**, so one threshold works at
  every magnitude.
- No data shape is hard-coded; `test_pipeline.py` runs the whole pipeline at a
  different feature count, class count and sample size.

---

## 9. Testing

```bash
python -m pytest -q
```

| File | Covers |
|---|---|
| `test_nsga2.py` | Dominance, constraint-domination, fronts, crowding, bounds, elitism, ZDT1 convergence |
| `test_determinism.py` | Multi-generation reproducibility, global-RNG isolation, content-addressed seeding |
| `test_diagnosis.py` | Each statistic against analytically known answers; do-no-harm; monotonicity |
| `test_repairs.py` | Constraint triggering, patch budgets, PatchML vs same-size random control |
| `test_pipeline.py` | Metrics, end-to-end, report schema, NaN survival, unseen data shapes |
| `test_no_leakage.py` | **AST proof** that decision modules never read `hidden_cause`; no plaintext secrets; docstring coverage |

The leakage test is static (AST), not behavioural, because a behavioural test
cannot prove absence of leakage — a system that peeked would simply score
perfectly and look excellent.

---

## 10. Security

- No secrets anywhere. The only environment variable read is `TEAM_ID`, via
  `os.environ.get` with a safe default.
- `.env` is gitignored; `.env.example` is the tracked template.
- No network calls, no `pickle`, no `eval`/`exec`, no shell-out.
- `test_no_leakage.py::test_no_plaintext_secrets_in_source` enforces this in CI.

---

## 11. Known limitations (state these before the jury asks)

1. **Synthetic benchmark.** Scenarios are generated, not real-world. The
   data-generating process is documented in `scenarios.py` so results are
   reproducible, but transfer to real drift is unproven.
2. **Classification only.** Regression and generative modelling are in the track
   theme and are not implemented.
3. **One cause per scenario.** Real failures are often compound; the evidence
   vector supports mixtures but the benchmark does not test them.
4. **Diagnostic scales are chosen, not learned.** They are set from the statistics'
   null distributions on unperturbed data and are monotone, but they are still
   design choices.
5. **Small search budget.** 16 × 6 = 96 evaluations per scenario. The front is
   stable, not provably optimal.
6. **The recovery ceiling is "retrain on everything".** A stronger method could
   exceed it; `beat_reference` in the report flags when our repair already does.

---

## 12. Reflection notes — parameter adaptation

Parameters sit in three tiers, adapted differently:

- **Structural** (`pop_size`, `generations`, `search_max_iter`): scale with the
  *evaluation budget*, not the problem. First knobs to raise if the auditor
  rewards solution quality over latency. `--fast` scales only these.
- **Statistical** (detection and diagnosis thresholds): derived from sampling noise,
  not hand-tuned. `detect.py` computes the binomial standard error of the accuracy
  estimate and requires the drop to exceed `noise_sigmas × σ`. These adapt
  automatically with validation-set size, which is why a hidden drift level cannot
  break them — and why `--fast` deliberately leaves them alone, so a fast run stays
  a valid rehearsal of the scored run.
- **Budgetary** (`max_patch_fraction`, `param_budget_ratio`, `max_fairness_gap`):
  encode the track's deployment constraints. Deliberately conservative.

**`search_max_iter` is a fidelity/throughput trade.** Rebuild-style families train a
network per candidate, ~30× the cost of a warm-start fine-tune, and dominated our
first full run. A 70-iteration fit ranks candidates almost identically to a
220-iteration one (the ordering of alpha and width settings is established early),
while letting the search visit far more of them. The chosen winner is then rebuilt
at full `max_iter` before scoring, so the deployed model loses no quality.

---

## 13. Round-2 live-patch playbook

Everything a surprise constraint could plausibly touch is reachable from the config
block at the top of `main.py`:

| Surprise constraint | One-line change |
|---|---|
| New drift magnitudes | `DataConfig.test_levels` |
| Tighter compute budget | `SearchConfig.pop_size` / `generations` |
| Smaller patch allowance | `RepairConfig.max_patch_fraction` |
| Stricter parameter budget | `RepairConfig.param_budget_ratio` |
| Stricter fairness cap | `RepairConfig.max_fairness_gap` |
| **A new objective** | one entry in `repairs.objectives_and_constraints` + bump `N_OBJECTIVES` |
| **A new constraint** | one entry in the `violations` array — constraint-domination needs no weight |
| **A new repair family** | one `build_*` function + one line in `FAMILIES` |
| **A new failure mode** | one builder in `scenarios.py` + one line in `CAUSE_TO_FAMILY` |

Then: re-run, log the score, and add the one-line *what changed and why* note to
`ATTEMPTS.md`.
