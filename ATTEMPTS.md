# Attempt log — AdaptX (OPT-26-5711)

Competition rule: **every attempt after the first needs a one-line
"what changed and why" note.** Record the score *before* starting the next change,
and never edit an earlier row — the history is part of what the auditor scans.

---

## Attempt 1 — Baseline AdaptX

**Status:** ready to submit
**What changed and why:** *(first attempt — no note required)*

**Scope**
- Full closed loop: Detect → Diagnose → Select → Evolve → Validate → Report.
- Own NumPy NSGA-II (`src/nsga2.py`); **no `pymoo` dependency**.
- Five repair families; PatchML with margin-based candidate pruning and warm starts.
- Determinism guaranteed by construction (thread pinning + content-addressed
  seeding + stable sorts + evaluation cache).
- 6 benchmark scenarios including a healthy control.

**Recorded results:** see `reports/latest.json` → `summary`.
Fill the table below from that file after the scored run — do not transcribe from
memory, and do not round in your favour.

| Metric | Value |
|---|---|
| `fitness` | **0.8777** |
| `diagnosis_accuracy` | 1.0 (6/6 scenarios) |
| `mean_ood_recovery` | 0.7815 |
| `mean_accuracy_before` -> `after` | 0.6749 -> 0.7316 |
| `mean_intervention_size` | 0.1869 |
| `mean_stability` | 0.8816 |
| `runtime_seconds` | 85.343 |

**Auditor feedback (7 parameters) — fill in after submission**

| Parameter | Weight | Score | Notes |
|---|---|---|---|
| Code Quality | 20% | | |
| Efficiency & Latency | 18% | | |
| Testing & Validation | 18% | | |
| Security & Secrets | 12% | | |
| Problem Alignment | 12% | | |
| Track Innovation | 10% | | |
| Accessibility & Docs | 10% | | |

---

## Attempt 2 — *(reserve for the auditor's weakest parameter)*

**What changed and why:**
> _(one line, mandatory)_

**Rule for this attempt:** read the Attempt-1 auditor breakdown, find the single
lowest-scoring parameter, and fix **only that**. Do not refactor broadly between
attempts — a wide change makes the score delta uninterpretable and you lose the
ability to reason about attempt 3.

Likely candidates and their one-line fixes:

| If weakest is… | Change |
|---|---|
| Efficiency & Latency | `SearchConfig.pop_size` / `generations` / `search_max_iter` down |
| Problem Alignment | raise `generations`; widen `min_confidence_to_repair` shortlist |
| Track Innovation | enable a second repair family per case and report both fronts |
| Testing | add the scenario-specific assertions listed in README §9 |

---

## Attempt 3 — *(Round 2: surprise constraint)*

**What changed and why:**
> _(one line, mandatory)_

**Playbook.** Do **not** rewrite the script. Identify which single module the
constraint touches, patch it, re-run, log. Mapping is in README §13:

- new objective → `repairs.objectives_and_constraints` + `N_OBJECTIVES`
- new constraint → one entry in the `violations` array (no weight needed —
  constraint-domination handles it)
- new repair family → one `build_*` + one line in `FAMILIES`
- new failure mode → one builder in `scenarios.py` + one line in `CAUSE_TO_FAMILY`
- tighter budget → one field in `RepairConfig` / `SearchConfig`

**Before re-running:** `python -m pytest -q` must still pass. A patched module that
breaks determinism costs more than the constraint does.

---

## Running notes

Use this section during the event for anything that is not an attempt: observed
leaderboard positions, auditor quirks, decisions made under time pressure.

- `main.py` pins BLAS threads to 1 **before** importing NumPy. If you ever move that
  import, determinism breaks silently — the tests will catch it, so run them.
- `--fast` changes only structural parameters, never thresholds, so a fast run
  remains a valid rehearsal of the scored run.
- `reports/latest.json` is committed deliberately: it is the evidence for the
  "final fitness score recorded per attempt" requirement.
