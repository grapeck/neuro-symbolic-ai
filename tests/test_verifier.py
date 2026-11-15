import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import sympy
from sympy.abc import x, y, z

from verifier import SymPyVerifier, ExecutionResult


@pytest.fixture
def verifier():
    return SymPyVerifier()


def test_execution_result_dataclass():
    res = ExecutionResult(result=42, error="", success=True)
    assert res.result == 42
    assert res.error == ""
    assert res.success is True

    res2 = ExecutionResult(result=None, error="Failed", success=False)
    assert res2.result is None
    assert res2.error == "Failed"
    assert res2.success is False


def test_execute_basic_expression(verifier):
    res, err = verifier.execute("integrate(x**2, x)")
    assert err == ""
    assert res == x**3 / 3


def test_execute_multiline_assignment(verifier):
    code = """
x = symbols('x')
res = diff(sin(x), x)
"""
    res, err = verifier.execute(code)
    assert err == ""
    assert res == sympy.cos(x)


def test_execute_markdown_wrapped(verifier):
    code = """```python
integrate(exp(x), x)
```"""
    res, err = verifier.execute(code)
    assert err == ""
    assert res == sympy.exp(x)


def test_execute_sandbox_restrictions(verifier):
    res, err = verifier.execute("import os")
    assert err != ""

    res, err = verifier.execute("open('some_file.txt')")
    assert err != ""

    res, err = verifier.execute("__import__('sys')")
    assert err != ""


def test_normalize(verifier):
    expr = sympy.sin(x)**2 + sympy.cos(x)**2
    norm = verifier.normalize(expr)
    assert norm == 1

    expr2 = (x + 1)**2 - (x**2 + 2*x + 1)
    norm2 = verifier.normalize(expr2)
    assert norm2 == 0


def test_are_equivalent_symbolic(verifier):
    assert verifier.are_equivalent("sin(x)**2 + cos(x)**2", 1)
    assert verifier.are_equivalent("tan(x)", "sin(x)/cos(x)")
    assert verifier.are_equivalent("(x + 2)*(x - 2)", "x**2 - 4")
    assert not verifier.are_equivalent("x", "x + 1")


def test_are_equivalent_numerical_fallback(verifier):
    # log(x) + log(y) vs log(x*y) requires positive assumptions to simplify symbolically
    assert verifier.are_equivalent("log(x) + log(y)", "log(x*y)")


def test_verify_integral(verifier):
    ok, err = verifier.verify_integral("x**3/3 + 10", "x**2", "x")
    assert ok is True
    assert err == ""

    ok, err = verifier.verify_integral("-cos(x) + C", "sin(x)", "x")
    assert ok is True
    assert err == ""

    ok, err = verifier.verify_integral("x**2", "x**2", "x")
    assert ok is False
    assert "Derivative of result" in err


def test_verify_equation_roots(verifier):
    ok, err = verifier.verify_equation_roots([2, -2], "x**2 - 4 = 0", "x")
    assert ok is True
    assert err == ""

    ok, err = verifier.verify_equation_roots(
        [sympy.I, -sympy.I], sympy.Eq(x**2, -1), "x"
    )
    assert ok is True
    assert err == ""

    ok, err = verifier.verify_equation_roots([2, 3], "x**2 - 4 = 0", "x")
    assert ok is False
    assert "Root 3 does not satisfy equation" in err


def test_verify_result_dispatch(verifier):
    # Integral dispatch
    ok, err = verifier.verify_result("x**3/3", "integral", integrand="x**2", variable="x")
    assert ok is True

    # Derivative dispatch
    ok, err = verifier.verify_result("cos(x)", "derivative", expected="cos(x)")
    assert ok is True

    ok, err = verifier.verify_result("cos(x)", "derivative", integrand="sin(x)", variable="x")
    assert ok is True

    # Limit dispatch
    ok, err = verifier.verify_result("1", "limit", expected=1)
    assert ok is True

    # Series dispatch
    ok, err = verifier.verify_result("1 + x", "series", expected="1 + x")
    assert ok is True

    # Equation dispatch
    ok, err = verifier.verify_result([2, -2], "equation", equation="x**2 - 4 = 0", variable="x")
    assert ok is True

    ok, err = verifier.verify_result([2, -2], "equation", expected=[-2, 2])
    assert ok is True

    # ExecutionResult dispatch
    exec_res = ExecutionResult(result=x**3/3, success=True)
    ok, err = verifier.verify_result(exec_res, "integral", integrand="x**2", variable="x")
    assert ok is True

    # Unsupported dispatch
    ok, err = verifier.verify_result("foo", "unsupported_task")
    assert ok is False
    assert "Unsupported task type" in err
