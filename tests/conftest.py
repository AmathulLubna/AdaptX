"""Shared pytest fixtures.

Threads are pinned at collection time, before scikit-learn is imported by any
test module, so the determinism tests exercise the same numeric environment the
scored run uses. Without this the reproducibility assertions would be testing a
different configuration from the one we submit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.determinism import pin_threads  # noqa: E402

pin_threads(1)

import pytest  # noqa: E402

from src.config import (Config, DataConfig, ModelConfig, RepairConfig,  # noqa: E402
                        SearchConfig)
from src.scenarios import build_scenarios  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    """A deliberately small configuration so the suite runs in seconds.

    Only STRUCTURAL parameters are shrunk -- thresholds keep their production
    values, so a test that passes here is evidence about the real system rather
    than about a differently-tuned toy.
    """
    return Config(
        seed=12345,
        search=SearchConfig(pop_size=6, generations=2),
        data=DataConfig(n_train=240, n_val=160, n_pool=300, n_ood=200,
                        n_features=8, n_informative=5, n_classes=3),
        model=ModelConfig(hidden_sizes=(12, 8), max_iter=80),
        repair=RepairConfig(replay_size=96, finetune_iters=8, min_patch_size=6),
    )


@pytest.fixture(scope="session")
def scenarios(tiny_config: Config):
    return build_scenarios(tiny_config)


@pytest.fixture(scope="session")
def env_scenario(scenarios):
    """The environment-shift case, used by the PatchML and determinism tests."""
    return next(s for s in scenarios if s.hidden_cause == "environment_shift")
