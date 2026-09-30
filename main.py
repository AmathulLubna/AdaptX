"""AdaptX -- entrypoint.  OptiForge 2026, Track 04.

    python main.py                 # full benchmark, writes reports/latest.json
    python main.py --fast          # reduced budget, for smoke tests
    python main.py --seed 123      # reproducibility check

=============================================================================
ROUND-2 CONFIG BLOCK -- change these, nothing else
=============================================================================
Every knob the surprise constraint could plausibly touch is reachable from here
through `Config`. Typical live patches and where they go:

  new drift magnitudes       -> DataConfig.test_levels
  tighter compute budget     -> SearchConfig.pop_size / generations
  smaller patch allowance    -> RepairConfig.max_patch_fraction
  stricter parameter budget  -> RepairConfig.param_budget_ratio
  stricter fairness cap      -> RepairConfig.max_fairness_gap
  a brand-new failure mode   -> one builder in scenarios.py + one FAMILIES entry

THREADS ARE PINNED BEFORE NUMPY LOADS
-------------------------------------
`pin_threads()` is the first executable statement on purpose. Multi-threaded BLAS
reduces dot products in a scheduling-dependent order, so identical inputs can
yield bit-different weights; that was one of the three root causes of the
non-determinism this build fixes. Setting the thread environment after NumPy is
imported has no effect, hence the import order below (and the noqa markers).
"""

from __future__ import annotations

from src.determinism import pin_threads

pin_threads(1)  # MUST precede the numpy/sklearn import chain below.

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from src.config import Config, DataConfig, SearchConfig  # noqa: E402
from src.doctor import run_benchmark  # noqa: E402
from src.report import build_report, print_summary, write_report  # noqa: E402
from src.scenarios import build_scenarios  # noqa: E402

REPORT_DIR = Path(__file__).parent / "reports"


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """CLI surface. Deliberately tiny: the sandbox may run `python main.py` bare."""
    parser = argparse.ArgumentParser(description="AdaptX benchmark runner")
    parser.add_argument("--seed", type=int, default=None,
                       help="override the master seed (default: Config.seed)")
    parser.add_argument("--fast", action="store_true",
                       help="reduced search budget for smoke tests")
    parser.add_argument("--out", type=str, default=str(REPORT_DIR / "latest.json"),
                       help="path for the JSON repair report")
    parser.add_argument("--quiet", action="store_true",
                       help="suppress the console summary")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Assemble the run configuration from defaults plus CLI overrides.

    REFLECTION NOTE (parameter adaptation)
    --------------------------------------
    `--fast` scales only the two STRUCTURAL parameters (population and
    generations) and shrinks the data. It never touches a threshold, because
    detection and diagnosis thresholds are derived from sampling statistics -- if
    a smaller budget changed them, the fast run would stop being a valid
    rehearsal of the scored run, which is the whole point of having it.
    """
    cfg = Config()
    if args.seed is not None:
        cfg = cfg.evolve(seed=int(args.seed))
    if args.fast:
        cfg = cfg.evolve(
            search=SearchConfig(pop_size=8, generations=2),
            data=DataConfig(n_train=300, n_val=200, n_pool=500, n_ood=250),
        )
    return cfg


def run(cfg: Config, out_path: "str | Path", quiet: bool = False) -> dict:
    """Execute the whole pipeline and return the report payload.

    Exposed as a plain function so the grading sandbox can import and call it
    without going through `argparse`; `main()` is only a thin CLI shim.
    """
    started = time.perf_counter()
    scenarios = build_scenarios(cfg)
    results = run_benchmark(scenarios, cfg)
    payload = build_report(results, cfg, time.perf_counter() - started)
    write_report(payload, out_path)
    if not quiet:
        print_summary(payload)
    return payload


def main(argv: "list[str] | None" = None) -> float:
    """Return the final fitness score, so callers can assert on it."""
    args = parse_args(argv)
    payload = run(build_config(args), args.out, args.quiet)
    return float(payload["summary"]["fitness"])


if __name__ == "__main__":
    main()
