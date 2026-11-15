"""Hybrid LLM + SymPy mathematical reasoning and solving pipeline.

This module provides a neuro-symbolic mathematical solver combining Large Language
Models (LLMs) with the SymPy computer algebra system. It features safe code
sandboxing, AST-based security checks, iterative self-correction, symbolic
equivalence checking, and backward verification for calculus tasks.
"""

from __future__ import annotations

import ast
import concurrent.futures
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy as sp
from sympy import (
    Eq,
    Integer,
    Matrix,
    Rational,
    Symbol,
    diff,
    integrate,
    latex,
    simplify,
    symbols,
    sympify,
)

try:
    from src.llm_client import LLMClient
except ImportError:
    try:
        from llm_client import LLMClient
    except ImportError:
        LLMClient = None  # type: ignore[assignment, misc]

try:
    from src.parser import CodeParser, ParseResult
except ImportError:
    try:
        from parser import CodeParser, ParseResult
    except ImportError:
        CodeParser = None  # type: ignore[assignment, misc]
        ParseResult = None  # type: ignore[assignment, misc]

try:
    from src.verifier import SymPyVerifier, ExecutionResult
except ImportError:
    try:
        from verifier import SymPyVerifier, ExecutionResult
    except ImportError:
        SymPyVerifier = None  # type: ignore[assignment, misc]
        ExecutionResult = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration settings for MathPipeline.

    Attributes:
        max_retries: Maximum number of self-correction retry attempts. Defaults to 3.
        timeout: Execution timeout in seconds for code evaluation. Defaults to 30.
        model_name: Model identifier for LLM queries. Defaults to 'gemini-2.5-pro'.
        llm_api_key: Optional API key for LLM services. Defaults to None.
        llm_base_url: Optional base URL for OpenAI-compatible endpoints. Defaults to None.
        mock_mode: When True, uses mock responses rather than calling remote APIs. Defaults to False.
        temperature: Sampling temperature for LLM generation. Defaults to 0.0.
    """

    max_retries: int = 3
    timeout: int = 30
    model_name: str = "gemini-2.5-pro"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    mock_mode: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Initialize configuration from environment variables if not provided."""
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("LLM_API_KEY")
        if self.llm_base_url is None:
            self.llm_base_url = os.getenv("LLM_BASE_URL")
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            self.model_name = env_model
        env_mock = os.getenv("MOCK_MODE")
        if env_mock is not None:
            self.mock_mode = env_mock.lower() in ("true", "1", "yes")
        env_retries = os.getenv("MAX_RETRIES")
        if env_retries is not None:
            try:
                self.max_retries = int(env_retries)
            except ValueError:
                pass


@dataclass
class TaskResult:
    """Encapsulates the final outcome of a mathematical solving task.

    Attributes:
        success: Whether the pipeline successfully solved the problem.
        answer: Formatted final answer string (e.g. LaTeX representation).
        sympy_result: The actual SymPy object produced, or None if in baseline mode.
        attempts: Total number of attempts made.
        error: Error message string if any error occurred, otherwise empty.
        execution_rate: True if code executed successfully and defined 'result'.
        symbolic_match: True if produced result matched expected answer symbolically.
        mode: The solving mode used ('hybrid' or 'baseline').
    """

    success: bool
    answer: str
    sympy_result: Any = None
    attempts: int = 1
    error: str = ""
    execution_rate: bool = False
    symbolic_match: bool = False
    mode: str = "hybrid"
    pass_at_1: bool = False
    pass_at_k: bool = False
    generated_code: str = ""


class _FallbackLLMClient:
    """Internal fallback client when LLMClient module is not imported."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.mock_mode = config.mock_mode

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response via mock solver."""
        text = (user_message or "").lower()
        if "limit" in text:
            return (
                "```python\n"
                "from sympy import *\n"
                "x = symbols('x')\n"
                "result = limit(sin(x)/x, x, 0)\n"
                "```"
            )
        if "integral" in text or "integrate" in text:
            return (
                "```python\n"
                "from sympy import *\n"
                "x = symbols('x')\n"
                "integrand = x**2\n"
                "result = integrate(integrand, x)\n"
                "```"
            )
        if "derivative" in text or "diff" in text:
            return (
                "```python\n"
                "from sympy import *\n"
                "x = symbols('x')\n"
                "result = diff(x**3, x)\n"
                "```"
            )
        return (
            "```python\n"
            "from sympy import *\n"
            "result = Integer(42)\n"
            "```"
        )


class MathPipeline:
    """Hybrid neuro-symbolic mathematical solver using LLM code generation and SymPy.

    Combines the language understanding of LLMs with the formal verification
    and exact precision of the SymPy computer algebra engine.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        llm: Optional[Any] = None,
        parser: Optional[Any] = None,
        verifier: Optional[Any] = None,
    ) -> None:
        """Initialize the MathPipeline.

        Args:
            config: Pipeline configuration settings. If None, default PipelineConfig is used.
            llm: Optional custom LLM client or mock. If None, instantiates LLMClient.
            parser: Optional code parser. If None, instantiates CodeParser.
            verifier: Optional SymPy verifier. If None, instantiates SymPyVerifier.
        """
        self.config = config if config is not None else PipelineConfig()

        if llm is not None:
            self.llm = llm
        elif LLMClient is not None:
            self.llm = LLMClient(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.model_name,
                mock_mode=self.config.mock_mode,
            )
        else:
            self.llm = _FallbackLLMClient(self.config)

        if parser is not None:
            self.parser = parser
        elif CodeParser is not None:
            self.parser = CodeParser()
        else:
            self.parser = None

        if verifier is not None:
            self.verifier = verifier
        elif SymPyVerifier is not None:
            self.verifier = SymPyVerifier()
        else:
            self.verifier = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for hybrid LLM+SymPy solving mode.

        Returns:
            The system prompt instructing the model to output only SymPy Python code.
        """
        return (
            "You are an expert mathematical problem solver that outputs executable Python code using SymPy.\n"
            "Given a math problem, you must solve it strictly by writing Python code.\n\n"
            "Strict Instructions:\n"
            "1. Output ONLY a valid Python code block enclosed in ```python and ```.\n"
            "2. Do NOT write any conversational text, explanations, or commentary outside the code block.\n"
            "3. Use SymPy for all symbolic math operations (e.g., symbols, solve, diff, integrate, Matrix, simplify).\n"
            "4. The final mathematical answer MUST be assigned to a variable named `result`.\n"
            "5. The `result` variable must contain the exact symbolic SymPy expression, value, list, or matrix.\n"
            "6. The code must be self-contained and execute cleanly without external input or file access."
        )

    def _build_correction_prompt(self, error: str, previous_code: str) -> str:
        """Build a correction prompt for self-correction when parsing or execution fails.

        Args:
            error: The error message or safety violation encountered.
            previous_code: The Python code from the failed attempt.

        Returns:
            A prompt explaining the error and requesting corrected Python code.
        """
        return (
            f"The previous Python code failed with the following error:\n"
            f"```\n{error}\n```\n\n"
            f"Previous code that failed:\n"
            f"```python\n{previous_code}\n```\n\n"
            f"Please identify the issue, fix it, and return a corrected Python code block.\n"
            f"Remember:\n"
            f"- Output ONLY a Python code block enclosed in ```python ... ```.\n"
            f"- The final answer MUST be stored in the variable `result`."
        )

    def _format_result(self, sympy_result: Any) -> str:
        """Format the SymPy result as a LaTeX string using sympy.latex().

        Args:
            sympy_result: The SymPy expression, value, or collection to format.

        Returns:
            A LaTeX representation of the result, or string representation as fallback.
        """
        if sympy_result is None:
            return ""
        try:
            return latex(sympy_result)
        except Exception:
            return str(sympy_result)

    def _extract_code(self, response_text: str) -> str:
        """Extract Python code block from LLM response text.

        Args:
            response_text: Raw output string from LLM.

        Returns:
            Extracted Python code string.

        Raises:
            ValueError: If no valid code block is found.
        """
        if not response_text or not response_text.strip():
            raise ValueError("Empty response received from LLM.")

        if self.parser is not None and hasattr(self.parser, "extract_code"):
            code = self.parser.extract_code(response_text)
            if code:
                return code

        # Regex fallback
        match = re.search(r"```(?:python|py)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        stripped = response_text.strip().strip("`").strip()
        if any(keyword in stripped for keyword in ("import sympy", "result =", "result=")):
            return stripped

        raise ValueError("No Python code block found in LLM response.")

    def _check_code_safety(self, code: str) -> None:
        """Perform static analysis on code to verify safety before execution.

        Args:
            code: Python code string to inspect.

        Raises:
            SyntaxError: If the code cannot be parsed by ast.parse.
            ValueError: If the code contains forbidden imports, calls, or attributes.
        """
        if self.parser is not None and hasattr(self.parser, "validate_safety"):
            is_safe, error = self.parser.validate_safety(code)
            if not is_safe:
                raise ValueError(f"Safety violation: {error}")
            return

        tree = ast.parse(code)
        allowed_modules = {"sympy", "math", "cmath", "fractions", "decimal", "typing"}
        forbidden_calls = {
            "eval", "exec", "open", "compile", "breakpoint", "__import__",
            "exit", "quit", "input", "system"
        }
        forbidden_attrs = {
            "__subclasses__", "__globals__", "__code__", "__bases__",
            "__mro__", "__builtins__"
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg not in allowed_modules:
                        raise ValueError(f"Forbidden import: module '{alias.name}' is not allowed.")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg not in allowed_modules:
                        raise ValueError(f"Forbidden import: module '{node.module}' is not allowed.")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in forbidden_calls:
                    raise ValueError(f"Forbidden function call: '{func.id}()' is not permitted.")
                elif isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                    raise ValueError(f"Forbidden method call: '{func.attr}()' is not permitted.")
            elif isinstance(node, ast.Attribute):
                if node.attr in forbidden_attrs:
                    raise ValueError(f"Forbidden attribute access: '{node.attr}' is not permitted.")

    def _create_isolated_namespace(self) -> Dict[str, Any]:
        """Create an isolated execution namespace pre-populated with SymPy symbols and safe builtins.

        Returns:
            Dictionary mapping identifiers to safe objects and SymPy functions.
        """
        def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
            root_pkg = name.split(".")[0]
            if root_pkg in ("sympy", "math", "cmath"):
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of module '{name}' is forbidden.")

        safe_builtins: Dict[str, Any] = {
            "__import__": safe_import,
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
            "print": lambda *args, **kwargs: None,
        }

        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "sympy": sp,
            "sp": sp,
        }
        for name in dir(sp):
            if not name.startswith("_"):
                namespace[name] = getattr(sp, name)

        return namespace

    def _execute_code(self, code: str) -> Tuple[Any, Dict[str, Any]]:
        """Execute Python code in an isolated SymPy namespace with a timeout.

        Args:
            code: Python code string to execute.

        Returns:
            Tuple of (extracted 'result' variable, local namespace dictionary).

        Raises:
            TimeoutError: If execution exceeds configured timeout.
            KeyError: If 'result' variable was not defined by code.
            Exception: Any runtime exception during execution.
        """
        namespace = self._create_isolated_namespace()

        def _target() -> Any:
            compiled = compile(code, "<sympy_exec>", "exec")
            exec(compiled, namespace, namespace)
            if "result" not in namespace:
                raise KeyError("Code executed successfully but did not define the required variable 'result'.")
            return namespace["result"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_target)
            try:
                res = future.result(timeout=self.config.timeout)
                return res, namespace
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Execution timed out after {self.config.timeout} seconds.")

    def _evaluate_expected_answer(self, expected_code: str) -> Any:
        """Evaluate expected answer code to obtain the ground truth SymPy object.

        Args:
            expected_code: Code or expression string representing expected answer.

        Returns:
            The evaluated SymPy object or value.
        """
        namespace = self._create_isolated_namespace()
        expected_code_clean = expected_code.strip()
        if "result" in expected_code_clean and "=" in expected_code_clean:
            exec(expected_code_clean, namespace, namespace)
            return namespace.get("result")
        try:
            return eval(expected_code_clean, namespace, namespace)
        except Exception:
            return sympify(expected_code_clean, locals=namespace)

    def _check_symbolic_equivalence(self, res: Any, expected: Any) -> bool:
        """Check if two symbolic or numerical results are mathematically equivalent.

        Args:
            res: Produced result object.
            expected: Expected ground truth object.

        Returns:
            True if symbolically or numerically equivalent, False otherwise.
        """
        if res == expected:
            return True

        if self.verifier is not None and hasattr(self.verifier, "are_equivalent"):
            try:
                return bool(self.verifier.are_equivalent(res, expected))
            except Exception:
                pass

        if isinstance(res, (list, tuple, set)) and isinstance(expected, (list, tuple, set)):
            if len(res) != len(expected):
                return False
            res_items = list(res)
            exp_items = list(expected)
            matched = [False] * len(exp_items)
            for r in res_items:
                found = False
                for j, e in enumerate(exp_items):
                    if not matched[j] and self._check_symbolic_equivalence(r, e):
                        matched[j] = True
                        found = True
                        break
                if not found:
                    return False
            return True

        if isinstance(res, Matrix) and isinstance(expected, Matrix):
            if res.shape != expected.shape:
                return False
            diff_mat = simplify(res - expected)
            return diff_mat == Matrix.zeros(*res.shape)

        if isinstance(res, Eq) and isinstance(expected, Eq):
            diff_eq = simplify((res.lhs - res.rhs) - (expected.lhs - expected.rhs))
            if diff_eq == 0:
                return True

        try:
            r_sym = sympify(res)
            e_sym = sympify(expected)
            if r_sym == e_sym:
                return True
            if hasattr(r_sym, "equals") and r_sym.equals(e_sym):
                return True
            diff_expr = simplify(r_sym - e_sym)
            if diff_expr == 0:
                return True
            val = complex(diff_expr.evalf())
            if abs(val) < 1e-7:
                return True
        except Exception:
            pass

        return False

    def _verify_integral(
        self, sympy_result: Any, code: str, namespace: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Perform backward verification for an integration task by differentiation.

        Args:
            sympy_result: The candidate integral result.
            code: The Python code that produced the result.
            namespace: The execution namespace.

        Returns:
            Tuple of (is_verified, message).
        """
        integrand = namespace.get("integrand") or namespace.get("expr")
        if callable(integrand):
            integrand = None
        var_cand = namespace.get("var")
        var = var_cand if isinstance(var_cand, (Symbol, str)) else namespace.get("x")
        if callable(var) or not isinstance(var, (Symbol, str)):
            var = None

        if integrand is None or var is None:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func_name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                        if func_name == "integrate" and node.args:
                            integrand_code = ast.unparse(node.args[0])
                            integrand = eval(integrand_code, namespace, namespace)
                            if len(node.args) > 1:
                                var_code = ast.unparse(node.args[1])
                                var_val = eval(var_code, namespace, namespace)
                                if isinstance(var_val, tuple) and len(var_val) >= 1:
                                    var = var_val[0]
                                else:
                                    var = var_val
                            break
            except Exception as exc:
                logger.debug(f"AST extraction of integrand failed: {exc}")

        if var is None:
            if hasattr(integrand, "free_symbols") and integrand.free_symbols:
                var = next(iter(integrand.free_symbols))
            elif hasattr(sympy_result, "free_symbols") and sympy_result.free_symbols:
                var = next(iter(sympy_result.free_symbols))
            else:
                var = Symbol("x")

        if integrand is None:
            return True, "No integrand detected for backward verification."

        if self.verifier is not None and hasattr(self.verifier, "verify_integral"):
            try:
                is_valid, msg = self.verifier.verify_integral(sympy_result, integrand, var)
                return bool(is_valid), str(msg)
            except Exception:
                pass

        try:
            derivative = diff(sympy_result, var)
            diff_check = simplify(derivative - integrand)
            if diff_check == 0 or (hasattr(derivative, "equals") and derivative.equals(integrand)):
                return True, "Backward verification succeeded: diff(result) == integrand."
            try:
                val = complex(diff_check.evalf())
                if abs(val) < 1e-7:
                    return True, "Backward verification succeeded: numerical diff(result) matches integrand."
            except Exception:
                pass

            return False, f"Backward verification failed: diff(result) = {derivative} != {integrand}"
        except Exception as exc:
            return False, f"Backward verification error: {exc}"

    def solve_hybrid(
        self,
        problem: str,
        expected_answer_code: Optional[str] = None,
        task_type: str = "general",
    ) -> TaskResult:
        """Solve a mathematical problem using the full hybrid LLM+SymPy pipeline with self-correction.

        Pipeline Steps:
        1. Build system prompt explaining that LLM must output ONLY a SymPy code block
           defining a variable named `result`.
        2. Send problem and history to LLM to obtain code response.
        3. Parse code (extract python block and perform AST safety checks).
        4. If parsing fails: append error to conversation history and retry (up to max_retries).
        5. Execute code in an isolated SymPy namespace with timeout enforcement.
        6. If execution fails: append error to history and retry (up to max_retries).
        7. If expected_answer_code is given: evaluate it and check symbolic equivalence.
        8. If integral task: run backward verification by differentiating result.
        9. Format result nicely as a LaTeX string.
        10. Return a TaskResult containing all evaluation metrics and answers.

        Args:
            problem: The mathematical problem description.
            expected_answer_code: Optional Python/SymPy code defining the ground truth answer.
            task_type: Type of math task (e.g., 'general', 'integral', 'algebra').

        Returns:
            TaskResult object with detailed solving status and formatted answer.
        """
        system_prompt = self._build_system_prompt()
        conversation_history: List[Dict[str, str]] = []

        attempts = 0
        max_attempts = max(1, self.config.max_retries)
        last_error = ""
        last_code = ""
        execution_rate = False
        symbolic_match = False
        sympy_result: Any = None
        formatted_answer = ""
        namespace: Dict[str, Any] = {}

        is_integral = (
            task_type.lower() in ("integral", "integration", "integrals")
            or "integrate" in problem.lower()
            or "integral" in problem.lower()
        )

        while attempts < max_attempts:
            attempts += 1
            execution_rate = False

            # Step 2: Query LLM
            try:
                raw_response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_message=problem,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                last_error = f"LLM Generation Error: {exc}"
                if attempts < max_attempts:
                    correction_prompt = self._build_correction_prompt(last_error, last_code)
                    conversation_history.append({"role": "assistant", "content": ""})
                    conversation_history.append({"role": "user", "content": correction_prompt})
                    continue
                break

            # Step 3 & 4: Parse code and safety check
            try:
                code = self._extract_code(raw_response)
                self._check_code_safety(code)
                last_code = code
            except Exception as exc:
                last_error = f"Parse Error: {exc}"
                if attempts < max_attempts:
                    correction_prompt = self._build_correction_prompt(last_error, raw_response)
                    conversation_history.append({"role": "assistant", "content": raw_response})
                    conversation_history.append({"role": "user", "content": correction_prompt})
                    continue
                break

            # Step 5 & 6: Execute code in isolated SymPy namespace
            try:
                sympy_result, namespace = self._execute_code(code)
                execution_rate = True
                last_error = ""
            except Exception as exc:
                execution_rate = False
                last_error = f"Execution Error: {exc}"
                if attempts < max_attempts:
                    correction_prompt = self._build_correction_prompt(last_error, code)
                    conversation_history.append({"role": "assistant", "content": raw_response})
                    conversation_history.append({"role": "user", "content": correction_prompt})
                    continue
                break

            # Step 8: Backward verification for integral task
            if is_integral:
                verified, verify_msg = self._verify_integral(sympy_result, code, namespace)
                if not verified:
                    last_error = verify_msg
                    execution_rate = False
                    if attempts < max_attempts:
                        correction_prompt = self._build_correction_prompt(last_error, code)
                        conversation_history.append({"role": "assistant", "content": raw_response})
                        conversation_history.append({"role": "user", "content": correction_prompt})
                        continue
                    break

            # If execution and verification succeeded, break retry loop
            break

        # Step 7: Evaluate expected answer and check symbolic equivalence
        if execution_rate and expected_answer_code is not None:
            try:
                expected_obj = self._evaluate_expected_answer(expected_answer_code)
                symbolic_match = self._check_symbolic_equivalence(sympy_result, expected_obj)
            except Exception as exc:
                symbolic_match = False
                if not last_error:
                    last_error = f"Expected Answer Check Error: {exc}"

        # Step 9: Format result nicely
        if sympy_result is not None:
            formatted_answer = self._format_result(sympy_result)

        # Determine overall success
        if expected_answer_code is not None:
            success = execution_rate and symbolic_match
        else:
            success = execution_rate and (not last_error)

        # Step 10: Return TaskResult
        return TaskResult(
            success=success,
            answer=formatted_answer,
            sympy_result=sympy_result,
            attempts=attempts,
            error=last_error,
            execution_rate=execution_rate,
            symbolic_match=symbolic_match,
            mode="hybrid",
            pass_at_1=(success and attempts == 1),
            pass_at_k=success,
            generated_code=last_code if last_code else "",
        )

    def solve_baseline(self, problem: str) -> TaskResult:
        """Solve a mathematical problem directly using the LLM without SymPy.

        Direct baseline solution asking the LLM to solve the problem and return
        the answer as text.

        Args:
            problem: Mathematical problem description.

        Returns:
            TaskResult representing the baseline solution.
        """
        system_prompt = (
            "You are an expert mathematical problem solver. "
            "Solve the problem directly and provide the final answer as text."
        )

        try:
            answer_text = self.llm.generate(
                system_prompt=system_prompt,
                user_message=problem,
                conversation_history=None,
            )
            return TaskResult(
                success=bool(answer_text and answer_text.strip()),
                answer=answer_text.strip() if answer_text else "",
                sympy_result=None,
                attempts=1,
                error="",
                execution_rate=False,
                symbolic_match=False,
                mode="baseline",
            )
        except Exception as exc:
            return TaskResult(
                success=False,
                answer="",
                sympy_result=None,
                attempts=1,
                error=str(exc),
                execution_rate=False,
                symbolic_match=False,
                mode="baseline",
            )
