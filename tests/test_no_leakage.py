"""Static proof that the hidden ground truth never reaches a decision.

WHY A STATIC TEST
-----------------
A behavioural test cannot prove the absence of leakage: a system that peeked at
`hidden_cause` would simply score perfectly and look excellent. The only sound
check is to inspect the source and assert that the decision modules never read
the attribute at all. We parse the AST rather than grepping, so a comment
mentioning the field does not trip the test and an obfuscated access
(`getattr(s, "hidden" + "_cause")`) still does.

This is also the test that defends our benchmark's credibility to the jury: it is
the reason our diagnosis accuracy is a real measurement rather than a tautology.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

SRC = Path(__file__).resolve().parents[1] / "src"

# Modules that make decisions. They must be blind to the ground truth.
DECISION_MODULES = ("detect.py", "diagnose.py", "repairs.py", "models.py",
                    "metrics.py", "nsga2.py")

# Modules allowed to touch it: the generator that defines it, the orchestrator
# that records it for scoring after all decisions are made, and the reporter.
ALLOWED = ("scenarios.py", "doctor.py", "report.py")

FORBIDDEN_NAMES = ("hidden_cause",)


def _module_paths(names) -> List[Path]:
    return [SRC / name for name in names]


def _attribute_reads(tree: ast.AST) -> List[str]:
    """Every attribute name read anywhere in the module."""
    return [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]


def _docstrings(tree: ast.AST) -> set:
    """Every docstring node in the module, by object identity.

    Docstrings are excluded from the indirect-access scan: a comment explaining
    that a module must NOT read the ground truth is evidence of compliance, not a
    violation, and flagging it would push us to delete the documentation that
    makes the guarantee auditable.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(id(first.value))
    return found


def _string_constants(tree: ast.AST) -> List[str]:
    """Executable string literals, to catch getattr-style indirection."""
    docstrings = _docstrings(tree)
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings]


def test_decision_modules_never_read_the_hidden_cause() -> None:
    for path in _module_paths(DECISION_MODULES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        attributes = _attribute_reads(tree)
        for forbidden in FORBIDDEN_NAMES:
            assert forbidden not in attributes, f"{path.name} reads .{forbidden}"


def test_decision_modules_contain_no_indirect_access() -> None:
    """Catches `getattr(scenario, "hidden_cause")` and dictionary lookups."""
    for path in _module_paths(DECISION_MODULES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals = _string_constants(tree)
        for forbidden in FORBIDDEN_NAMES:
            offenders = [s for s in literals if forbidden in s]
            assert not offenders, f"{path.name} references {forbidden} as a string"


def test_decision_modules_do_not_import_the_cause_vocabulary_for_branching() -> None:
    """`CAUSES` may be imported (to shape the output vector) but never compared.

    Diagnosis must emit a confidence for every cause; what it must not do is
    branch on which cause is 'the' answer before the evidence is computed.
    """
    diagnose = ast.parse((SRC / "diagnose.py").read_text(encoding="utf-8"))
    for node in ast.walk(diagnose):
        if isinstance(node, ast.Compare):
            literals = _string_constants(node)
            assert not any(s in ("environment_shift", "feature_instability",
                                 "input_corruption", "class_imbalance",
                                 "overfitting") for s in literals), \
                "diagnose.py branches on a cause name"


def test_allowed_modules_exist() -> None:
    """Guards against the allowlist silently drifting from the file layout."""
    for path in _module_paths(ALLOWED + DECISION_MODULES):
        assert path.exists(), f"missing module {path.name}"


# `CaseResult.diagnosis_correct` is the SCORING property: comparing our answer
# against the ground truth is its entire job, and it runs after every decision has
# been made. Everything else in doctor.py must stay blind.
SCORING_ONLY = ("diagnosis_correct",)


def test_scoring_happens_after_decisions_in_doctor() -> None:
    """`hidden_cause` may only be stored or scored, never used to decide.

    In `doctor.py` the ground truth is passed straight into `CaseResult` for the
    evaluator. It must not appear in the test of any `if` outside the scoring
    property, which would mean a decision was being conditioned on it.
    """
    tree = ast.parse((SRC / "doctor.py").read_text(encoding="utf-8"))
    exempt = {id(inner)
              for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name in SCORING_ONLY
              for inner in ast.walk(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and id(node) not in exempt:
            assert "hidden_cause" not in _attribute_reads(node.test), \
                "doctor.py branches on the hidden cause outside scoring"


def test_the_scoring_exemption_is_narrow() -> None:
    """The allowlist above must not quietly grow into a loophole.

    If someone adds a decision function to SCORING_ONLY, this test makes that
    visible rather than letting the leakage guarantee erode silently.
    """
    assert len(SCORING_ONLY) == 1
    assert SCORING_ONLY == ("diagnosis_correct",)


def test_no_plaintext_secrets_in_source() -> None:
    """Security parameter: no credentials anywhere in the tree.

    The system needs no secrets at all; the only environment variable read is
    TEAM_ID, and it is read via `os.environ.get` with a safe default.
    """
    suspicious = ("api_key", "apikey", "password", "secret_key", "aws_access",
                  "token =", "bearer ")
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for needle in suspicious:
            assert needle not in text, f"{path.name} contains {needle!r}"


def test_every_public_function_has_a_docstring() -> None:
    """Accessibility & docs parameter, enforced rather than assumed."""
    missing: List[str] = []
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"{path.name} has no module docstring"
        # Only module- and class-level definitions are required to carry a
        # docstring. Closures (the per-family `decode`/`evaluate` functions) are
        # implementation detail of the factory that documents them, and requiring
        # a docstring on each would duplicate the factory's own explanation.
        top_level = list(tree.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                top_level.extend(node.body)
        for node in top_level:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                missing.append(f"{path.name}:{node.name}")
    assert not missing, f"undocumented public definitions: {missing}"
