"""Input validation, typed exceptions, and error-recovery routines.

WHY A DEDICATED VALIDATION LAYER
--------------------------------
A diagnosis system is handed data it did not create: a pool that may be empty, a
model whose classes do not match the labels, an array full of NaN, a
configuration whose budgets contradict each other. Left unchecked, these surface
far from their cause -- a shape error deep inside a fitness evaluation, forty
generations into a search, with no indication of which input was wrong.

Every public entry point therefore validates its inputs and raises a typed
exception naming the offending value. Two principles:

  * FAIL FAST on programmer error (a malformed configuration, mismatched array
    shapes). These cannot be recovered from and must be loud.
  * DEGRADE GRACEFULLY on data error (a corrupted pool, a class missing from a
    batch, a non-finite feature). These are the conditions the system exists to
    repair, so crashing on them would defeat its purpose.

The distinction is the whole design. `validate_dataset` raises; `recover_array`
repairs and reports. Which one a given failure gets is a deliberate choice, not
an accident of where the exception happened to be caught.

SECRET BOUNDARY
---------------
`load_env_config` is the only place the process reads environment variables. It
reads exactly three non-sensitive keys, never logs a value, and refuses to
accept anything that looks like a credential -- so a misconfigured deployment
cannot leak a secret into a report or a traceback. The system needs no secrets
to run; this exists to keep it that way.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

# Environment keys this system will read. Anything else is ignored, so a secret
# accidentally exported into the environment cannot reach the report.
ALLOWED_ENV_KEYS: Tuple[str, ...] = ("TEAM_ID", "ADAPTX_SEED", "ADAPTX_OUT")

# Substrings that mark a value as credential-shaped. Matching keys are refused
# outright rather than sanitised, because a redacted secret is still a secret
# that was read into memory.
SECRET_PATTERNS: Tuple[str, ...] = (
    "key", "secret", "token", "password", "passwd", "credential",
    "auth", "session", "cookie", "private",
)


class AdaptXError(Exception):
    """Base class for every error this system raises deliberately.

    Catching `AdaptXError` catches our errors and nothing else, so a caller can
    distinguish a rejected input from a genuine bug in NumPy or scikit-learn.
    """


class ValidationError(AdaptXError, ValueError):
    """An input failed validation. Subclasses ValueError for compatibility."""


class ConfigurationError(AdaptXError, ValueError):
    """A configuration is internally inconsistent (contradictory budgets)."""


class SecurityError(AdaptXError, RuntimeError):
    """A credential-shaped value was encountered where none is permitted."""


class SearchError(AdaptXError, RuntimeError):
    """The evolutionary search could not produce a usable candidate."""


def validate_array(array: Any, name: str, *, ndim: "int | None" = None,
                   min_rows: int = 1, allow_nonfinite: bool = True) -> np.ndarray:
    """Coerce to a float64 array and check its shape.

    Complexity: O(n) in the number of elements, dominated by the finiteness scan
    (skipped entirely when `allow_nonfinite` is True).

    Parameters
    ----------
    allow_nonfinite:
        True for observation matrices, which legitimately contain NaN under the
        input-corruption failure mode -- rejecting those would make the system
        unable to diagnose the very thing it is built for. False for arrays that
        must be clean, such as computed objective vectors.

    Raises
    ------
    ValidationError
        If the value is not array-like, has the wrong rank, has too few rows, or
        contains non-finite entries when those are disallowed.
    """
    try:
        out = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name}: not convertible to a float array ({exc})") from exc

    if ndim is not None and out.ndim != ndim:
        raise ValidationError(f"{name}: expected {ndim}-D, got {out.ndim}-D with shape {out.shape}")
    if out.ndim >= 1 and out.shape[0] < min_rows:
        raise ValidationError(f"{name}: needs at least {min_rows} row(s), got {out.shape[0]}")
    if not allow_nonfinite and not np.all(np.isfinite(out)):
        bad = int(np.count_nonzero(~np.isfinite(out)))
        raise ValidationError(f"{name}: contains {bad} non-finite value(s)")
    return out


def validate_labels(labels: Any, name: str, n_classes: "int | None" = None) -> np.ndarray:
    """Coerce labels to int64 and check they index valid classes.

    Complexity: O(n).

    Raises
    ------
    ValidationError
        If labels are not integral, are negative, or exceed `n_classes - 1`.
    """
    try:
        out = np.asarray(labels, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name}: not convertible to integer labels ({exc})") from exc

    if out.ndim != 1:
        raise ValidationError(f"{name}: expected 1-D, got shape {out.shape}")
    if out.size and out.min() < 0:
        raise ValidationError(f"{name}: negative class index {int(out.min())}")
    if n_classes is not None and out.size and out.max() >= n_classes:
        raise ValidationError(
            f"{name}: class index {int(out.max())} exceeds n_classes={n_classes}")
    return out


def validate_matched_lengths(**arrays: Any) -> None:
    """Check that every named array has the same leading dimension.

    Complexity: O(k) in the number of arrays -- lengths only, no element access.

    Mismatched X and y is the single most common data error and produces a
    baffling error deep inside scikit-learn; naming both lengths here turns it
    into a one-line diagnosis.
    """
    lengths: Dict[str, int] = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        lengths[name] = int(arr.shape[0]) if arr.ndim else 0
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(lengths.items()))
        raise ValidationError(f"length mismatch: {detail}")


def validate_probability_vector(values: Any, name: str, tol: float = 1e-6) -> np.ndarray:
    """Check that rows are non-negative and sum to one.

    Complexity: O(n * k).

    Used on predicted probabilities before calibration is computed, because a
    calibration error derived from unnormalised scores is silently meaningless.
    """
    out = validate_array(values, name, ndim=2, allow_nonfinite=False)
    if np.any(out < -tol):
        raise ValidationError(f"{name}: contains negative probabilities")
    sums = out.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1e-3):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise ValidationError(f"{name}: rows do not sum to 1 (worst deviation {worst:.3g})")
    return out


def validate_bounds(lower: Any, upper: Any, name: str = "bounds") -> "Tuple[np.ndarray, np.ndarray]":
    """Check that a search box is well formed.

    Complexity: O(n).

    Raises
    ------
    ValidationError
        If the two arrays differ in shape or any upper bound is below its lower.
    """
    lo = validate_array(lower, f"{name}.lower", ndim=1, allow_nonfinite=False)
    hi = validate_array(upper, f"{name}.upper", ndim=1, allow_nonfinite=False)
    if lo.shape != hi.shape:
        raise ValidationError(f"{name}: shape mismatch {lo.shape} vs {hi.shape}")
    if np.any(hi < lo):
        idx = int(np.argmax(hi < lo))
        raise ValidationError(f"{name}: upper < lower at index {idx} ({hi[idx]} < {lo[idx]})")
    return lo, hi


def validate_fraction(value: float, name: str, *, low: float = 0.0,
                      high: float = 1.0) -> float:
    """Check that a scalar lies within an inclusive range.

    Complexity: O(1).
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name}: not a number ({value!r})") from exc
    if not np.isfinite(out):
        raise ValidationError(f"{name}: must be finite, got {out}")
    if not (low <= out <= high):
        raise ValidationError(f"{name}: must lie in [{low}, {high}], got {out}")
    return out


def validate_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    """Check that a value is an integer at or above a floor.

    Complexity: O(1).
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name}: must be an integer, got {type(value).__name__}")
    if int(value) < minimum:
        raise ValidationError(f"{name}: must be >= {minimum}, got {value}")
    return int(value)


def recover_array(array: Any, fallback: "np.ndarray | float" = 0.0,
                  name: str = "array") -> "Tuple[np.ndarray, Dict[str, int]]":
    """Repair a damaged observation matrix instead of rejecting it.

    Complexity: O(n) with a single pass over the non-finite mask.

    This is the graceful-degradation half of the design. A pool arriving with
    NaN or infinities is not a programmer error -- it is the input-corruption
    failure mode, the thing the system is built to diagnose. Crashing on it
    would be a category error, so the damage is repaired and *reported*: the
    returned counts feed the corruption evidence, so nothing is hidden by being
    fixed.

    Returns
    -------
    (repaired, report)
        `report` counts `nan`, `posinf`, `neginf` and `total` replacements.
    """
    out = np.array(array, dtype=np.float64, copy=True)
    nan_mask = np.isnan(out)
    pos_mask = np.isposinf(out)
    neg_mask = np.isneginf(out)
    bad = nan_mask | pos_mask | neg_mask

    report = {
        "nan": int(np.count_nonzero(nan_mask)),
        "posinf": int(np.count_nonzero(pos_mask)),
        "neginf": int(np.count_nonzero(neg_mask)),
        "total": int(np.count_nonzero(bad)),
    }
    if report["total"]:
        if isinstance(fallback, np.ndarray) and out.ndim == 2:
            # Column-wise fallback: replace each damaged entry with its own
            # feature's reference value, never a single global constant, which
            # would collapse the feature geometry the diagnosis depends on.
            out[bad] = np.take(fallback, np.nonzero(bad)[1])
        else:
            out[bad] = float(fallback) if not isinstance(fallback, np.ndarray) else 0.0
    return out, report


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0,
                epsilon: float = 1e-12) -> np.ndarray:
    """Element-wise division that never raises or returns NaN.

    Complexity: O(n).

    Every ratio in the diagnosis layer (divergence over standard deviation,
    concentration over total) can meet a zero denominator on degenerate data --
    a constant feature, an empty class. Returning `default` keeps the statistic
    defined rather than poisoning every downstream comparison with NaN.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, float(default), dtype=np.float64)
    ok = np.abs(den) > epsilon
    np.divide(num, den, out=out, where=ok)
    return out


def load_env_config() -> Dict[str, str]:
    """Read the small set of permitted environment keys.

    Complexity: O(k) over `ALLOWED_ENV_KEYS`, independent of environment size.

    SECURITY BOUNDARY. This is the only function in the system that reads
    `os.environ`. It is an allowlist, not a filter: a key outside
    `ALLOWED_ENV_KEYS` is never read at all, so a secret exported into the
    process environment cannot reach a log, a report or a traceback. No value is
    ever printed by this module.

    Raises
    ------
    SecurityError
        If a permitted key nonetheless holds a credential-shaped value, which
        means the deployment is misconfigured and should stop rather than
        continue with a secret in memory.
    """
    config: Dict[str, str] = {}
    for key in ALLOWED_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is None:
            continue
        if contains_secret_marker(key):
            raise SecurityError(f"refusing to read credential-shaped key {key!r}")
        config[key] = raw.strip()
    return config


def contains_secret_marker(text: str) -> bool:
    """True when a name looks like it holds a credential.

    Complexity: O(p * m) over the pattern list and name length -- both tiny.

    Word-boundary matching avoids false positives on legitimate names: `monkey`
    contains "key" but is not a credential, while `API_KEY` is.
    """
    lowered = str(text).lower()
    return any(re.search(rf"(^|[^a-z]){p}([^a-z]|$)", lowered) for p in SECRET_PATTERNS)


def assert_no_secrets(mapping: "Dict[str, Any] | Iterable[Tuple[str, Any]]") -> None:
    """Raise if any key in a payload looks like a credential.

    Complexity: O(k) over the mapping.

    Called on the report payload before it is written, so a credential can never
    be serialised into the submitted JSON even if some future field is added
    carelessly.
    """
    items = mapping.items() if isinstance(mapping, dict) else mapping
    offenders = [k for k, _ in items if contains_secret_marker(k)]
    if offenders:
        raise SecurityError(f"credential-shaped keys in payload: {sorted(offenders)}")


def clamp(value: float, low: float, high: float) -> float:
    """Constrain a scalar to a range, tolerating reversed bounds.

    Complexity: O(1).
    """
    if low > high:
        low, high = high, low
    return float(min(max(float(value), low), high))


def summarise_failure(exc: BaseException, context: str) -> str:
    """One-line, type-qualified description of a caught exception.

    Complexity: O(1).

    Used by recovery paths so a degraded run still records *what* went wrong.
    A recovery that logs nothing is indistinguishable from one that never fired.
    """
    return f"{context}: {type(exc).__name__}: {exc}"


def coerce_sequence(values: Any, name: str, length: "int | None" = None) -> Sequence[float]:
    """Coerce to a flat float sequence and optionally check its length.

    Complexity: O(n).
    """
    out = validate_array(values, name, ndim=1, min_rows=0)
    if length is not None and out.shape[0] != length:
        raise ValidationError(f"{name}: expected length {length}, got {out.shape[0]}")
    return out
