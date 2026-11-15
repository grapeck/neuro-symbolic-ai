"""SymPy-based verification engine for hybrid LLM + SymPy mathematical solvers.

This module provides sandboxed execution of generated SymPy code, symbolic and
numerical equivalence checking, expression normalization, and specialized verification
routines for integrals, equation roots, derivatives, limits, and series expansions.
"""

from __future__ import annotations

import ast
import random
import types
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import sympy
import sympy.abc

# Safe built-in functions allowed during sandboxed code execution
SAFE_MATH_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "complex": complex,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class ExecutionResult:
    """Dataclass holding the result, error message, and status of code execution.

    Attributes:
        result: The computation output (SymPy expression, numeric value, list of roots, etc.),
            or None if execution failed.
        error: The error message string if an error occurred, otherwise an empty string.
        success: Boolean flag indicating whether the execution succeeded without errors.
    """

    result: Any = None
    error: str = ""
    success: bool = True


class SymPyVerifier:
    """Mathematical verifier for hybrid LLM + SymPy solver systems.

    Provides isolated code execution in a restricted namespace, algebraic normalization,
    heuristic and symbolic equivalence checking, and domain-specific verification methods
    for integrals, equations, derivatives, limits, and series.
    """

    def __init__(self) -> None:
        """Initializes the SymPyVerifier with an isolated global execution namespace."""
        self._namespace: dict[str, Any] = self._build_isolated_namespace()

    @staticmethod
    def _build_isolated_namespace() -> dict[str, Any]:
        """Constructs a secure global namespace containing SymPy symbols and safe math builtins.

        Returns:
            A dictionary mapping allowed identifiers to their corresponding SymPy
            or safe Python math objects.
        """
        ns: dict[str, Any] = {
            "__builtins__": SAFE_MATH_BUILTINS,
        }

        # Populate all public non-module attributes from SymPy
        for name in getattr(sympy, "__all__", []):
            if not name.startswith("_"):
                val = getattr(sympy, name, None)
                if val is not None and not isinstance(val, types.ModuleType):
                    ns[name] = val

        # Ensure all standard sympy.abc symbols (e.g. x, y, z, t, n, k) are available
        for sym_name in dir(sympy.abc):
            if not sym_name.startswith("_") and sym_name not in ns:
                val = getattr(sympy.abc, sym_name, None)
                if isinstance(val, sympy.Symbol):
                    ns[sym_name] = val

        # Guarantee critical symbols and constants are present
        explicit_symbols = {
            "symbols": sympy.symbols,
            "Symbol": sympy.Symbol,
            "integrate": sympy.integrate,
            "diff": sympy.diff,
            "limit": sympy.limit,
            "Sum": sympy.Sum,
            "oo": sympy.oo,
            "E": sympy.E,
            "pi": sympy.pi,
            "I": sympy.I,
            "sqrt": sympy.sqrt,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "exp": sympy.exp,
            "log": sympy.log,
            "ln": sympy.ln,
            "solve": sympy.solve,
            "solveset": sympy.solveset,
            "simplify": sympy.simplify,
            "expand": sympy.expand,
            "trigsimp": sympy.trigsimp,
            "Eq": sympy.Eq,
            "Rational": sympy.Rational,
            "Integer": sympy.Integer,
            "Float": sympy.Float,
        }
        for k, v in explicit_symbols.items():
            ns[k] = v

        return ns

    def execute(self, code: str) -> tuple[Any, str]:
        """Executes SymPy code in an ISOLATED namespace.

        Only SymPy symbols and functions are allowed in globals, and no builtins
        are accessible except safe mathematical builtins.

        Args:
            code: Python/SymPy source code string to execute.

        Returns:
            A tuple of (result, error_message). On success, error_message is empty.
            On failure, result is None and error_message contains the exception details.
        """
        clean_code = code.strip()

        # Strip markdown code blocks if provided by an LLM
        if clean_code.startswith("```"):
            lines = clean_code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_code = "\n".join(lines).strip()

        if not clean_code:
            return None, "Empty code provided"

        # Create isolated execution contexts
        exec_globals: dict[str, Any] = dict(self._namespace)
        exec_locals: dict[str, Any] = {}

        try:
            tree = ast.parse(clean_code)
            if not tree.body:
                return None, "Empty AST body"

            last_stmt = tree.body[-1]

            # If the last statement is an expression, evaluate it to return its value
            if isinstance(last_stmt, ast.Expr):
                if len(tree.body) > 1:
                    preceding_module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    code_obj = compile(preceding_module, "<string>", "exec")
                    exec(code_obj, exec_globals, exec_locals)

                eval_globals = {**exec_globals, **exec_locals}
                expr_obj = compile(ast.Expression(body=last_stmt.value), "<string>", "eval")
                res = eval(expr_obj, eval_globals, exec_locals)
                return res, ""

            # Otherwise, execute all statements
            code_obj = compile(tree, "<string>", "exec")
            exec(code_obj, exec_globals, exec_locals)

            # Check if the last assignment defined a variable
            if isinstance(last_stmt, ast.Assign) and last_stmt.targets:
                target = last_stmt.targets[-1]
                if isinstance(target, ast.Name) and target.id in exec_locals:
                    return exec_locals[target.id], ""
            elif isinstance(last_stmt, ast.AnnAssign) and isinstance(last_stmt.target, ast.Name):
                if last_stmt.target.id in exec_locals:
                    return exec_locals[last_stmt.target.id], ""

            # Check common variable names where solver results are stored
            for var_name in ("result", "ans", "sol", "solution", "res", "roots", "output"):
                if var_name in exec_locals:
                    return exec_locals[var_name], ""

            # Fallback to single defined variable if present
            if len(exec_locals) == 1:
                return next(iter(exec_locals.values())), ""

            return exec_locals, ""

        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)}"

    def normalize(self, expr: Any) -> Any:
        """Brings a SymPy expression to canonical form using simplify(expand(expr)) and trigsimp.

        Args:
            expr: SymPy expression, string, numeric value, or collection of expressions.

        Returns:
            The normalized canonical SymPy expression.
        """
        if isinstance(expr, str):
            try:
                expr = sympy.sympify(expr)
            except Exception:
                return expr
        elif isinstance(expr, (list, tuple)):
            return type(expr)(self.normalize(item) for item in expr)
        elif isinstance(expr, set):
            return {self.normalize(item) for item in expr}

        try:
            expanded = sympy.expand(expr)
            simplified = sympy.simplify(expanded)
            return sympy.trigsimp(simplified)
        except Exception:
            try:
                return sympy.simplify(expr)
            except Exception:
                return expr

    def are_equivalent(self, expr1: Any, expr2: Any) -> bool:
        """Checks if two SymPy expressions are symbolically equivalent.

        Evaluation strategy:
        1. Direct equality check.
        2. Symbolic simplification: sympy.simplify(sympy.expand(expr1 - expr2)) == 0.
        3. Trigonometric simplification: trigsimp.
        4. Canonical normalization comparison.
        5. Numerical evaluation at random points as a fallback.

        Args:
            expr1: First SymPy expression or string representation.
            expr2: Second SymPy expression or string representation.

        Returns:
            True if expressions are equivalent, False otherwise.
        """
        if expr1 == expr2:
            return True

        try:
            s1 = sympy.sympify(expr1) if isinstance(expr1, str) else expr1
            s2 = sympy.sympify(expr2) if isinstance(expr2, str) else expr2
        except Exception:
            return False

        if s1 == s2:
            return True

        try:
            diff_expr = s1 - s2
        except Exception:
            return False

        # 1. sympy.simplify(sympy.expand(expr1 - expr2)) == 0
        try:
            simplified_diff = sympy.simplify(sympy.expand(diff_expr))
            if simplified_diff == 0 or getattr(simplified_diff, "is_zero", False):
                return True
        except Exception:
            pass

        # 2. Try trigsimp
        try:
            trig_diff = sympy.trigsimp(diff_expr)
            if trig_diff == 0 or getattr(trig_diff, "is_zero", False):
                return True
            if sympy.simplify(trig_diff) == 0:
                return True
            if sympy.trigsimp(sympy.simplify(diff_expr)) == 0:
                return True
        except Exception:
            pass

        # Compare canonical normalized forms
        try:
            if self.normalize(s1) == self.normalize(s2):
                return True
        except Exception:
            pass

        # Try SymPy's builtin equals method
        try:
            if hasattr(diff_expr, "equals") and diff_expr.equals(0):
                return True
        except Exception:
            pass

        # 3. Numerical evaluation at random points as fallback
        try:
            free_symbols = list(diff_expr.free_symbols) if hasattr(diff_expr, "free_symbols") else []
            if not free_symbols:
                val = complex(sympy.N(diff_expr))
                return abs(val) < 1e-9

            rng = random.Random(42)
            successful_trials = 0
            num_trials = 5

            for _ in range(num_trials):
                # Pick positive random values away from zero and poles
                point = {sym: rng.uniform(0.7, 4.3) for sym in free_symbols}
                try:
                    substituted = diff_expr.subs(point)
                    val = complex(sympy.N(substituted))
                    if abs(val) > 1e-7:
                        return False
                    successful_trials += 1
                except Exception:
                    continue

            if successful_trials >= 2:
                return True
        except Exception:
            pass

        return False

    def verify_integral(
        self, result: Any, integrand: Any, variable: Any = None
    ) -> tuple[bool, str]:
        """Verifies an indefinite integral by differentiating result and checking simplify(diff(result, var) - integrand) == 0.

        Args:
            result: Proposed antiderivative expression or string.
            integrand: The original integrand expression or string.
            variable: The integration variable (Symbol or string). Inferred if None.

        Returns:
            A tuple of (is_valid, error_message). On success, error_message is empty.
        """
        try:
            res_expr = sympy.sympify(result) if isinstance(result, str) else result
            integ_expr = sympy.sympify(integrand) if isinstance(integrand, str) else integrand

            if isinstance(variable, str):
                var = sympy.Symbol(variable)
            elif variable is not None and not callable(variable):
                var = variable
            else:
                free = set()
                if hasattr(integ_expr, "free_symbols"):
                    free.update(integ_expr.free_symbols)
                if hasattr(res_expr, "free_symbols"):
                    free.update(res_expr.free_symbols)
                candidates = [s for s in free if s.name not in ("C", "c")]
                if candidates:
                    var = candidates[0]
                elif free:
                    var = next(iter(free))
                else:
                    var = sympy.Symbol("x")

            diff_res = sympy.diff(res_expr, var)
            diff_diff = sympy.simplify(diff_res - integ_expr)
            if diff_diff == 0 or getattr(diff_diff, "is_zero", False):
                return True, ""

            if self.are_equivalent(diff_res, integ_expr):
                return True, ""

            return (
                False,
                f"Derivative of result ({diff_res}) does not equal integrand ({integ_expr})",
            )
        except Exception as e:
            return False, f"Integral verification error: {str(e)}"

    def verify_equation_roots(
        self, roots: Any, equation: Any, variable: Any = None
    ) -> tuple[bool, str]:
        """Verifies equation roots by substituting back into the equation.

        Args:
            roots: Proposed root(s) - can be a single root, list/tuple/set of roots,
                or dictionary mapping variables to values.
            equation: SymPy Eq, expression equal to 0, or equation string.
            variable: Variable to solve for (Symbol or string). Inferred if None.

        Returns:
            A tuple of (is_valid, error_message). On success, error_message is empty.
        """
        try:
            if isinstance(equation, str):
                eq_str = equation.strip()
                if eq_str.startswith("Eq(") and eq_str.endswith(")"):
                    eq_obj = sympy.sympify(eq_str)
                    expr_to_check = eq_obj.lhs - eq_obj.rhs
                elif "=" in eq_str:
                    lhs_str, rhs_str = eq_str.split("=", 1)
                    expr_to_check = sympy.sympify(lhs_str) - sympy.sympify(rhs_str)
                else:
                    expr_to_check = sympy.sympify(eq_str)
            elif hasattr(equation, "lhs") and hasattr(equation, "rhs"):
                expr_to_check = equation.lhs - equation.rhs
            else:
                expr_to_check = sympy.sympify(equation)

            if isinstance(variable, str):
                var = sympy.Symbol(variable)
            elif variable is not None and not callable(variable):
                var = variable
            else:
                free = getattr(expr_to_check, "free_symbols", set())
                if len(free) == 1:
                    var = next(iter(free))
                elif sympy.Symbol("x") in free:
                    var = sympy.Symbol("x")
                elif free:
                    var = sorted(list(free), key=lambda s: s.name)[0]
                else:
                    var = sympy.Symbol("x")

            if isinstance(roots, (list, tuple, set, sympy.FiniteSet)):
                root_list = list(roots)
            elif isinstance(roots, str):
                try:
                    parsed = sympy.sympify(roots)
                    if isinstance(parsed, (list, tuple, set, sympy.FiniteSet)):
                        root_list = list(parsed)
                    else:
                        root_list = [parsed]
                except Exception:
                    root_list = [roots]
            else:
                root_list = [roots]

            if not root_list:
                return False, "No roots provided to verify"

            for r in root_list:
                if isinstance(r, dict):
                    subs_dict = r
                elif isinstance(r, sympy.Equality):
                    subs_dict = {r.lhs: r.rhs}
                else:
                    subs_val = sympy.sympify(r) if isinstance(r, str) else r
                    subs_dict = {var: subs_val}

                try:
                    sub_val = expr_to_check.subs(subs_dict)
                except Exception as e:
                    return False, f"Substitution error for root {r}: {str(e)}"

                try:
                    simplified = sympy.simplify(sympy.expand(sub_val))
                    if simplified == 0 or getattr(simplified, "is_zero", False):
                        continue
                except Exception:
                    pass

                if self.are_equivalent(sub_val, 0):
                    continue

                return False, f"Root {r} does not satisfy equation {equation} (residual: {sub_val})"

            return True, ""
        except Exception as e:
            return False, f"Equation verification error: {str(e)}"

    def verify_result(
        self,
        result: Any,
        task_type: str,
        expected: Any = None,
        integrand: Any = None,
        variable: Any = None,
        equation: Any = None,
    ) -> tuple[bool, str]:
        """Dispatch method that calls appropriate verification based on task_type.

        Supported task types:
        - 'limit': Verifies computed limit against expected limit.
        - 'derivative': Verifies computed derivative against expected or by differentiating integrand.
        - 'integral': Verifies antiderivative via differentiation against integrand or expected.
        - 'series': Verifies series expansion against expected series.
        - 'equation': Verifies equation roots by substituting back into equation or checking against expected.

        Args:
            result: Computed solver result or ExecutionResult.
            task_type: Type of mathematical task ('limit', 'derivative', 'integral', 'series', 'equation').
            expected: Optional expected result expression or root set.
            integrand: Integrand or source expression for integral and derivative verification.
            variable: Variable with respect to which operation is performed.
            equation: Target equation for equation root verification.

        Returns:
            A tuple of (is_valid, error_message). On success, error_message is empty.
        """
        if isinstance(result, ExecutionResult):
            if not result.success:
                return False, f"Execution failed: {result.error}"
            result = result.result

        task = task_type.lower().strip()

        if task in ("integral", "integrals"):
            if integrand is not None:
                return self.verify_integral(result=result, integrand=integrand, variable=variable)
            if expected is not None:
                if self.are_equivalent(result, expected):
                    return True, ""
                var = (
                    sympy.Symbol(variable)
                    if isinstance(variable, str)
                    else (variable or sympy.Symbol("x"))
                )
                try:
                    diff_diff = sympy.diff(sympy.sympify(result) - sympy.sympify(expected), var)
                    if self.are_equivalent(diff_diff, 0):
                        return True, ""
                except Exception:
                    pass
                return False, f"Result {result} does not match expected integral {expected}"
            return False, "Missing integrand or expected result for integral verification"

        elif task == "equation":
            if equation is not None:
                return self.verify_equation_roots(
                    roots=result, equation=equation, variable=variable
                )
            if expected is not None:
                res_list = (
                    list(result)
                    if isinstance(result, (list, tuple, set, sympy.FiniteSet))
                    else [result]
                )
                exp_list = (
                    list(expected)
                    if isinstance(expected, (list, tuple, set, sympy.FiniteSet))
                    else [expected]
                )
                if len(res_list) != len(exp_list):
                    return (
                        False,
                        f"Number of roots ({len(res_list)}) does not match expected ({len(exp_list)})",
                    )
                matched = [False] * len(exp_list)
                for r in res_list:
                    found = False
                    for i, exp_r in enumerate(exp_list):
                        if not matched[i] and self.are_equivalent(r, exp_r):
                            matched[i] = True
                            found = True
                            break
                    if not found:
                        return False, f"Root {r} not found in expected roots {expected}"
                return True, ""
            return False, "Missing equation or expected roots for equation verification"

        elif task in ("derivative", "derivatives"):
            if expected is not None:
                if self.are_equivalent(result, expected):
                    return True, ""
                return False, f"Derivative result {result} does not match expected {expected}"
            if integrand is not None:
                var = (
                    sympy.Symbol(variable)
                    if isinstance(variable, str)
                    else (variable or sympy.Symbol("x"))
                )
                try:
                    target_deriv = sympy.diff(sympy.sympify(integrand), var)
                    if self.are_equivalent(result, target_deriv):
                        return True, ""
                    return (
                        False,
                        f"Result {result} does not match derivative of {integrand} ({target_deriv})",
                    )
                except Exception as e:
                    return False, f"Derivative calculation error: {str(e)}"
            return False, "Missing expected derivative or target expression for derivative verification"

        elif task in ("limit", "limits"):
            if expected is not None:
                if self.are_equivalent(result, expected):
                    return True, ""
                return False, f"Limit result {result} does not match expected {expected}"
            return False, "Missing expected value for limit verification"

        elif task == "series":
            if expected is not None:
                if self.are_equivalent(result, expected):
                    return True, ""
                return False, f"Series result {result} does not match expected {expected}"
            return False, "Missing expected series for series verification"

        else:
            return (
                False,
                f"Unsupported task type: '{task_type}'. Supported types: 'limit', 'derivative', 'integral', 'series', 'equation'",
            )
