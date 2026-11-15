"""
tests/test_pipeline.py — Pytest unit tests for the hybrid math solver.

Test groups:
  1. AST Safety Filter (TestASTSafetyFilter): checks blocking of unsafe code.
  2. Canonical Normalization (TestCanonicalNormalization): tests algebraic equivalence.
  3. Integral Backward Verification (TestIntegralVerification): verifies differentiation.
  4. Pipeline Integration (TestPipelineIntegration): runs pipeline in MOCK_MODE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import sympy as sp

# Ensure src/ and project root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set MOCK_MODE environment variable to "true" before pipeline tests
os.environ["MOCK_MODE"] = "true"

from src.parser import CodeParser
from src.verifier import SymPyVerifier
from src.pipeline import MathPipeline, PipelineConfig, TaskResult


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def setup_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MOCK_MODE environment variable is set to 'true'."""
    monkeypatch.setenv("MOCK_MODE", "true")


@pytest.fixture
def parser() -> CodeParser:
    return CodeParser()


@pytest.fixture
def verifier() -> SymPyVerifier:
    return SymPyVerifier()


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> MathPipeline:
    monkeypatch.setenv("MOCK_MODE", "true")
    config = PipelineConfig(mock_mode=True, max_retries=3)
    return MathPipeline(config=config)


# ===========================================================================
# GROUP 1 - AST Safety Filter (test_parser.py tests in this file)
# ===========================================================================

class TestASTSafetyFilter:
    """Group 1: AST Safety Filter tests."""

    def test_safe_code_passes(self, parser: CodeParser) -> None:
        """Code with only sympy imports and operations passes."""
        code = (
            "import sympy as sp\n"
            "from sympy import symbols, sin, cos, diff, integrate\n"
            "x = symbols('x')\n"
            "result = integrate(sin(x) + cos(x), x)\n"
        )
        is_safe, error = parser.validate_safety(code)
        assert is_safe is True
        assert error == ""

    def test_import_os_blocked(self, parser: CodeParser) -> None:
        """`import os` raises/returns unsafe."""
        code = "import os\nresult = os.getcwd()"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "os" in error.lower() or "forbidden" in error.lower()

    def test_import_sys_blocked(self, parser: CodeParser) -> None:
        """`import sys` is blocked."""
        code = "import sys\nresult = sys.version"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "sys" in error.lower() or "forbidden" in error.lower()

    def test_exec_blocked(self, parser: CodeParser) -> None:
        """Code containing `exec(...)` is blocked."""
        code = "exec('x = 1')\nresult = 1"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "exec" in error.lower() or "forbidden" in error.lower()

    def test_eval_blocked(self, parser: CodeParser) -> None:
        """Code containing `eval(...)` is blocked."""
        code = "result = eval('1 + 1')"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "eval" in error.lower() or "forbidden" in error.lower()

    def test_open_blocked(self, parser: CodeParser) -> None:
        """Code containing `open(...)` is blocked."""
        code = "f = open('test.txt')\nresult = f.read()"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "open" in error.lower() or "forbidden" in error.lower()

    def test_subprocess_blocked(self, parser: CodeParser) -> None:
        """`import subprocess` is blocked."""
        code = "import subprocess\nresult = subprocess.run(['ls'])"
        is_safe, error = parser.validate_safety(code)
        assert is_safe is False
        assert "subprocess" in error.lower() or "forbidden" in error.lower()


# ===========================================================================
# GROUP 2 - Canonical Normalization (different algebraic forms are equivalent)
# ===========================================================================

class TestCanonicalNormalization:
    """Group 2: Canonical Normalization and algebraic equivalence tests."""

    def test_equivalent_polynomial_forms(self, verifier: SymPyVerifier) -> None:
        """(x+1)**2 == x**2 + 2*x + 1."""
        x = sp.Symbol("x")
        expr1 = (x + 1) ** 2
        expr2 = x**2 + 2 * x + 1
        assert verifier.are_equivalent(expr1, expr2)
        assert verifier.normalize(expr1) == verifier.normalize(expr2)

    def test_equivalent_trig_forms(self, verifier: SymPyVerifier) -> None:
        """sin(x)**2 + cos(x)**2 == 1."""
        x = sp.Symbol("x")
        expr1 = sp.sin(x) ** 2 + sp.cos(x) ** 2
        expr2 = sp.Integer(1)
        assert verifier.are_equivalent(expr1, expr2)
        assert verifier.normalize(expr1) == 1

    def test_different_expressions_not_equivalent(self, verifier: SymPyVerifier) -> None:
        """x**2 != x**3."""
        x = sp.Symbol("x")
        expr1 = x**2
        expr2 = x**3
        assert not verifier.are_equivalent(expr1, expr2)

    def test_zero_simplification(self, verifier: SymPyVerifier) -> None:
        """(x**2 - 1)/(x-1) simplified at x=2 gives 3."""
        x = sp.Symbol("x")
        expr = (x**2 - 1) / (x - 1)
        simplified = sp.simplify(expr)
        assert simplified == x + 1
        assert simplified.subs(x, 2) == 3
        assert verifier.are_equivalent(expr, x + 1)


# ===========================================================================
# GROUP 3 - Integral Backward Verification
# ===========================================================================

class TestIntegralVerification:
    """Group 3: Integral backward verification via differentiation."""

    def test_integral_verification_correct(self, verifier: SymPyVerifier) -> None:
        """diff(x**3/3, x) - x**2 == 0."""
        x = sp.Symbol("x")
        integral_result = x**3 / 3
        integrand = x**2
        assert sp.diff(integral_result, x) - integrand == 0
        is_valid, msg = verifier.verify_integral(integral_result, integrand, x)
        assert is_valid is True
        assert msg == ""

    def test_integral_verification_wrong(self, verifier: SymPyVerifier) -> None:
        """diff(x**2, x) - x**2 != 0."""
        x = sp.Symbol("x")
        wrong_result = x**2
        integrand = x**2
        assert sp.diff(wrong_result, x) - integrand != 0
        is_valid, msg = verifier.verify_integral(wrong_result, integrand, x)
        assert is_valid is False
        assert msg != ""


# ===========================================================================
# GROUP 4 - Pipeline Integration (using MOCK_MODE)
# ===========================================================================

class TestPipelineIntegration:
    """Group 4: Pipeline integration tests in MOCK_MODE."""

    def test_pipeline_mock_limit(
        self, pipeline: MathPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Solve a limit problem in mock mode, check result is not error."""
        monkeypatch.setenv("MOCK_MODE", "true")
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            result: TaskResult = pipeline.solve_hybrid(
                problem="Find the limit of sin(x)/x as x approaches 0",
                task_type="limit",
            )
            assert result.error == ""
            assert result.success is True
            assert result.sympy_result is not None
            assert result.mode == "hybrid"

    def test_pipeline_mock_integral(
        self, pipeline: MathPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Solve an integral problem in mock mode."""
        monkeypatch.setenv("MOCK_MODE", "true")
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            result: TaskResult = pipeline.solve_hybrid(
                problem="Find the indefinite integral of x**2 * exp(x) with respect to x",
                task_type="integral",
            )
            assert result.error == ""
            assert result.success is True
            assert result.sympy_result is not None
            assert result.mode == "hybrid"

    def test_pipeline_baseline_mock(
        self, pipeline: MathPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Baseline mode returns a result."""
        monkeypatch.setenv("MOCK_MODE", "true")
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            result: TaskResult = pipeline.solve_baseline(
                problem="Find the derivative of x**3"
            )
            assert result.mode == "baseline"
            assert result.answer != ""
            assert result.success is True
            assert result.error == ""
