# Viva cheat sheet — AdaptX (OPT-26-5711)

**Every teammate must be able to say all of Part 1 in their own words.**
Plain language first, technical term second. If you don't understand a line, ask
before you walk in — the panel will find the one line you can't defend.

---

## PART 1 — The 60-second pitch

> A model was working. The world changed. It broke.
> Most systems can tell you *that* it broke. Ours works out **why**, then makes
> the **smallest possible change** to fix it.
>
> It's a doctor, not a factory. It doesn't build a new patient — it diagnoses
> this one and treats it.
>
> Five stages: **Detect → Diagnose → Choose treatment → Evolve the treatment →
> Check it worked.**
>
> Our headline result: one model lost 19 accuracy points to a distribution shift.
> We recovered 82% of that loss by fine-tuning on **29 out of 1200** new samples —
> 2.4% of the available data.

---

## PART 2 — The five questions they WILL ask

### Q1. "What is your chromosome / representation?"

**Plain:** A candidate is a list of numbers describing *one possible repair*.

**Technical:** A real vector `x ∈ [l,u]^n` plus a per-gene type tag.

- **PatchML** — 192 **binary** genes. Gene *i* = "include shortlisted sample *i* in
  the repair set, yes/no". A subset-selection problem, so a bitmask is the natural
  encoding.
- **FeatureReweighting** — 12 **real** genes in [0,2], one multiplier per feature.
- **ClassReweighting** — 3 real genes in [0.2,5], one per class.
- **RobustPreprocessing** — 3 real genes: clip σ, augmentation σ, patch fraction.
- **Regularisation** — 4 real genes: log₁₀α, layer 1 width, layer 2 width, pool use.

**If asked "why binary, why not a score per sample?"** → A real-valued score needs
an arbitrary cut-off to become a subset; the threshold would then be an unjustified
free parameter. A count-plus-ranking can say *how many* but not *which*.

---

### Q2. "What are your operators, and why those?"

| | Real genes | Binary genes |
|---|---|---|
| Crossover | SBX (η=15) | Uniform |
| Mutation | Polynomial (η=20) | Bit-flip |

**Selection:** binary tournament on (constraint violation, then Pareto rank, then
crowding distance).
**Survival:** elitist (μ+λ) — parents and children compete together.

**The killer detail (say this unprompted, it shows you understand):**
> "We use **uniform** crossover for the patch mask, not one-point. A one-point
> operator assumes neighbouring genes belong together. For a patch mask, sample
> #17 and sample #18 in the pool have nothing to do with each other — there's no
> locus structure to preserve, so a positional operator would be imposing
> structure that doesn't exist."

**Mutation rate = 1 / n_genes**, not a constant → expected ~1 mutated gene
regardless of genome length, so behaviour doesn't change when the pool size does.

---

### Q3. "Why NSGA-II and Pareto fronts? Why not just one score?"

**Plain:** Because "best" isn't one thing. A repair using 500 samples might be
slightly more accurate than one using 30. Which is better? That depends on what
you can afford — so we show the whole trade-off curve instead of pretending
there's one answer.

**Technical:** Five objectives, all minimised:

| # | Objective |
|---|---|
| 0 | −mean OOD score (recovery) |
| 1 | intervention size |
| 2 | log(1 + training cost) |
| 3 | parameter ratio |
| 4 | 1 − stability across drift levels |

**Why not a weighted sum:** (a) the weights would be arbitrary and the panel could
challenge any choice; (b) a weighted sum **cannot reach non-convex parts of the
front** — no weight vector selects those points, so you'd never find them.

We *do* pick one final model, by a normalised weighted Chebyshev scalarisation of
the front (`doctor.KNEE_WEIGHTS`). **Weights apply only to the final pick, never to
the search.**

**Cost is not wall-clock time** — that varies with machine load and would break
determinism. We use `n_samples × n_iters × n_params`, proportional to FLOPs.

---

### Q4. "How do you handle constraints?"

**Plain:** Some things aren't negotiable — the repair must actually help, must not
bloat the model, must be fair across sub-groups.

**Technical:** Three constraints, via **constraint-domination** (Deb):

1. must beat the failed model by ≥ 0.01 accuracy
2. parameter count ≤ 1.5× original
3. sub-population accuracy gap ≤ 0.25

Rules: feasible always beats infeasible; between two infeasible, less violation
wins; between two feasible, ordinary Pareto dominance.

**Why this and not a penalty term:** a penalty needs a weight, and that weight
silently trades accuracy against constraint violation at a rate nobody chose.
Constraint-domination needs **no weight at all** — it's a pure ordering.

---

### Q5. "Prove your search converges. And is it deterministic?"

**Convergence (say it as 3 steps):**
1. **Elitism.** We truncate parents ∪ children together, never children alone.
2. So a solution on the best front can only be replaced by one that **dominates**
   it. Therefore best-per-objective is monotone non-increasing.
3. That's asserted directly in
   `test_nsga2.py::test_elitism_never_loses_the_best_objective_value`, and
   validated on **ZDT1**, a standard benchmark with a known analytic front.

**Be honest about the limit:**
> "We claim convergence to a **stable non-dominated set under a fixed budget** —
> not convergence to the true Pareto front. Claiming the latter for a
> finite-population stochastic search would be indefensible."

**Determinism — the three root causes we closed:**
1. **Shared global RNG.** If evaluation draws from the same stream as the
   operators, an evaluation that makes more draws shifts the stream for everything
   after it. *Invisible in generation 1, fatal later — which is exactly the bug
   signature we had.* → Fixed with explicit `Generator` objects + seeding each
   evaluation from a **BLAKE2b hash of the genome's contents**.
2. **Threaded BLAS.** Multi-threaded dot products sum in scheduling-dependent
   order → bit-different weights from identical inputs. → `pin_threads(1)` is the
   **first line of `main.py`**, before NumPy is imported.
3. **Unstable sorts.** Ties in crowding distance resolved differently each run. →
   every sort is `kind="stable"`.

Plus an evaluation cache keyed on genome bytes, so a surviving elite is never
re-fitted.

**Result:** fitness is a genuine **function** `f: G → R⁵`, not a random variable.
Verified two ways: a 20-generation test that deliberately desynchronises the
global RNG between runs, and two **separate processes** producing byte-identical
reports.

---

## PART 3 — PatchML (our headline contribution)

**The question:** if the environment changed, what is the **smallest** set of new
samples that repairs the model?

**The obstacle:** one bit per pool sample = 2000 genes. Search space 2^2000. No
5-generation search touches that.

**Our answer — diagnosis-guided candidate pruning:**
1. Ask the failing model to predict on all 1200 new samples.
2. Compute each sample's **margin** = gap between its top-two class probabilities.
3. Keep the 192 **lowest-margin** samples — the ones nearest the decision boundary
   the shift has displaced.
4. Evolve a binary mask over those 192 only.

**Why it's sound:** a sample the model is confidently right about carries almost no
repair signal. The boundary is where the information is.
**Why it's legal:** uses only the model's own outputs — no labels, no ground truth.
Usable at deployment time on unlabelled data.
**Search space:** 2^2000 → 2^192.

**Warm starts:** generation 0 is seeded with prefix masks (8, 16, 32, 64, 128 of
the ranking), so we start from a good front instead of hunting for one.

**Replay is mandatory, not optional:** every fine-tune mixes in original training
samples. Without it you get catastrophic forgetting, and a class can vanish from
the batch entirely — `partial_fit` then crashes. It's a correctness requirement.

**Result to quote:** 0.608 → 0.742, **recovery 0.821, using 29 of 1200 samples
(2.4%)**.

---

## PART 4 — "How is this different from AutoML?" (they will ask)

> AutoML asks *"which configuration scores highest?"*
> We ask *"why did **this** model fail, and what's the **smallest** fix?"*

Four concrete differences:
1. **Diagnosis picks the search space.** `CAUSE_TO_FAMILY` routes each diagnosed
   cause to one repair family. We never blindly search all five.
2. **Intervention size is an objective.** AutoML has no concept of "change as
   little as possible".
3. **We adapt, we don't replace.** 4 of 5 families warm-start from the deployed
   weights. Only `Regularisation` rebuilds — because excess capacity is a property
   of the hypothesis class and no amount of fine-tuning removes it.
4. **Do no harm.** The healthy control is detected as healthy and returned
   **untouched, with zero evaluations spent**.

---

## PART 5 — Diagnosis: how each cause is identified

No classifier is trained. Each cause is scored by a **direct measurement of its
mechanism**, squashed to [0,1] by `1 − exp(−s/scale)`.

| Cause | Signature |
|---|---|
| environment_shift | mean standardised divergence across **all** features |
| feature_instability | that divergence **concentrated** in a few features |
| input_corruption | missing rate + **robust** heavy-tail rate |
| class_imbalance | label-prior shift **not explained by** covariate shift |
| overfitting | train/val gap **with no** input-space movement |

**Three disambiguations worth naming — each one fixed a real misdiagnosis:**

- **Shift vs feature instability** share the same raw divergence, so they're
  separated by its *shape*: a normalised concentration ratio. Feature instability
  is gated on high concentration; environment shift is damped by it.
- **Corruption vs rescaling.** Our first version measured outliers against the
  *original* scale — so a feature multiplied by 2.3 looked full of outliers, and
  feature-shift was misdiagnosed as corruption. Fixed by measuring heavy tails
  against each feature's **own median and MAD**: a rescaled Gaussian has no
  unusual tail relative to itself; spike noise does.
- **Imbalance vs shift.** Under covariate shift with a fixed labelling function,
  P(y) moves automatically as P(x) moves. An undamped prior statistic therefore
  fires on *every* shift scenario. We damp it by the covariate divergence to
  isolate genuine prior shift.

**Why `1 − exp(−s/scale)` specifically:** it's **monotone**. A stronger unseen
perturbation is further along the same curve, never off a tuned cliff. That is the
formal basis of our robustness claim for hidden levels 1.5 and 2.0, and it's tested
(`test_monotone_confidence_in_perturbation_strength`).

---

## PART 6 — Robustness / Round 2

- Search optimises on drift levels **0.8 / 1.0 / 1.2**.
  Scoring uses **0.6 / 1.0 / 1.5 / 2.0**. The fitness function never sees the
  hidden levels.
- Objective 4 (stability) explicitly rewards repairs that are **flat** across drift
  magnitude — that's what makes them extrapolate.
- The concentration statistic is **scale-invariant**, so one threshold works at
  every magnitude.
- Detection thresholds are derived from the **binomial standard error** of the
  accuracy estimate, not hand-tuned constants — they adapt to sample size
  automatically.
- No data shape is hard-coded; a test runs the whole pipeline at a different
  feature count, class count and sample size.

---

## PART 7 — Limitations (say these BEFORE they're extracted from you)

Volunteering limitations reads as mastery. Being caught hiding one reads as the
opposite.

1. **Synthetic benchmark.** Generated scenarios, not real-world drift. Transfer is
   unproven.
2. **Classification only.** Regression and generative modelling are in the track
   theme and are not implemented.
3. **One cause per scenario.** Real failures are often compound; the evidence
   vector supports mixtures but the benchmark doesn't test them.
4. **Weak recovery on two cases.** Overfitting recovers only 0.29 — repairing
   memorisation needs more than the 70 training samples that scenario has. Input
   corruption's full-retrain ceiling is *below* the failed model, so its recovery
   metric is degenerate and our repair is roughly neutral there.
5. **Diagnostic scales are chosen, not learned.** Set from the statistics' null
   distributions on unperturbed data, and monotone — but still design choices.
6. **Small budget.** 16 × 5 = 80 evaluations per scenario. The front is stable, not
   provably optimal.

---

## PART 8 — Numbers to have memorised

| | |
|---|---|
| Final fitness | **0.8777** |
| Diagnosis accuracy | **6/6 scenarios** |
| Mean OOD recovery | 0.78 |
| Mean accuracy | 0.675 → 0.732 |
| Mean intervention size | 0.19 |
| Runtime | ~85 s |
| Tests | **80 passing** |
| PatchML headline | 0.608 → 0.742, **29/1200 samples (2.4%)** |
| Feature-shift headline | 0.588 → 0.752, recovery 0.91, **13 samples** |
| Population × generations | 16 × 5 = 80 evaluations |

---

## PART 9 — If you get stuck

- **Don't bluff.** "I'd have to check that in the code" beats a wrong answer. The
  panel is testing whether you understand your own system, not whether you're
  infallible.
- **If asked something we didn't do:** "We didn't implement that — here's why we
  prioritised X instead, and here's how we'd add it."
- **If challenged on a threshold:** point at how it was *derived* (sampling noise,
  null distribution) rather than defending the number itself.
- **If they find a bug:** agree, say what it would affect, say how you'd verify the
  fix. That's an engineering answer.
