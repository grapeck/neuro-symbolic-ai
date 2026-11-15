"""Parser and AST safety validator for hybrid LLM + SymPy math solvers.

This module provides extraction of Python code blocks from LLM responses
and rigorous AST-based safety validation before execution in symbolic solvers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re


@dataclass
class ParseResult:
    """Result of parsing and validating LLM-generated code.

    Attributes:
        code: Extracted Python code, or None if extraction or validation failed.
        error: Description of any extraction, syntax, or security validation error.
        is_safe: Whether the code passed all safety checks.
    """

    code: str | None = None
    error: str = ""
    is_safe: bool = False

    @classmethod
    def from_tuple(cls, result: tuple[str | None, str]) -> ParseResult:
        """Create a ParseResult from a (code, error) tuple.

        Args:
            result: A tuple of (code_or_None, error_message).

        Returns:
            A new ParseResult instance.
        """
        code, error = result
        is_safe = (code is not None) and (not error)
        return cls(code=code, error=error, is_safe=is_safe)


class CodeParser:
    """Parser that extracts and validates Python code from LLM responses."""

    # Forbidden module imports
    FORBIDDEN_MODULES: set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "http",
        "pathlib",
    }

    # Forbidden builtin functions and identifiers
    FORBIDDEN_BUILTINS: set[str] = {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }

    # Forbidden AST node types
    FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )

    # Allowed modules for import (only sympy and math)
    ALLOWED_MODULES: set[str] = {
        "sympy",
        "math",
    }

    # Regex patterns for extracting code blocks
    _PYTHON_BLOCK_RE = re.compile(
        r"```(?:python|py)\b[^\n]*\r?\n?(.*?)\r?\n?```",
        re.DOTALL | re.IGNORECASE,
    )
    _GENERIC_BLOCK_RE = re.compile(
        r"```[^\S\r\n]*\r?\n?(.*?)\r?\n?```",
        re.DOTALL,
    )

    def __init__(
        self,
        allowed_modules: set[str] | None = None,
        forbidden_modules: set[str] | None = None,
        forbidden_builtins: set[str] | None = None,
    ) -> None:
        """Initialize CodeParser with optional custom rule overrides.

        Args:
            allowed_modules: Set of module names allowed to be imported.
            forbidden_modules: Set of module names forbidden from being imported.
            forbidden_builtins: Set of builtin functions/names forbidden from use.
        """
        self.allowed_modules: set[str] = (
            set(allowed_modules)
            if allowed_modules is not None
            else set(self.ALLOWED_MODULES)
        )
        self.forbidden_modules: set[str] = (
            set(forbidden_modules)
            if forbidden_modules is not None
            else set(self.FORBIDDEN_MODULES)
        )
        self.forbidden_builtins: set[str] = (
            set(forbidden_builtins)
            if forbidden_builtins is not None
            else set(self.FORBIDDEN_BUILTINS)
        )
        self.forbidden_nodes: tuple[type[ast.AST], ...] = self.FORBIDDEN_NODES

    def extract_code(self, llm_response: str) -> str | None:
        """Extract Python code from markdown code blocks in an LLM response.

        Searches for code wrapped in ```python ... ``` or generic ``` ... ```
        fences using regular expressions.

        Args:
            llm_response: The raw string response from an LLM.

        Returns:
            The extracted code as a string, or None if no markdown code block
            is found.
        """
        if not isinstance(llm_response, str) or not llm_response.strip():
            return None

        # 1. Look for explicit Python code blocks first (```python or ```py)
        match = self._PYTHON_BLOCK_RE.search(llm_response)
        if match:
            return match.group(1).strip()

        # 2. Fall back to generic code blocks (``` ... ```)
        match = self._GENERIC_BLOCK_RE.search(llm_response)
        if match:
            return match.group(1).strip()

        return None

    def validate_safety(self, code: str) -> tuple[bool, str]:
        """Validate code safety using Python AST inspection.

        Checks for forbidden module imports, forbidden builtins, forbidden
        AST nodes (Delete, Global, Nonlocal), unauthorized imports (only sympy
        and math are allowed), and forbidden function calls.

        Args:
            code: The Python code string to validate.

        Returns:
            A tuple of (is_safe, error_message). If the code is safe, returns
            (True, ""). If unsafe or invalid, returns (False, error_description).
        """
        if not isinstance(code, str):
            return False, "Code must be a string."

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc.msg} at line {exc.lineno}"
        except Exception as exc:
            return False, f"Failed to parse code: {exc}"

        for node in ast.walk(tree):
            # Check forbidden AST nodes
            if isinstance(node, self.forbidden_nodes):
                node_type = type(node).__name__
                return False, f"Forbidden AST node: '{node_type}'"

            # Check direct module imports (import x, import x as y, import x.y)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.forbidden_modules:
                        return False, f"Forbidden module import: '{alias.name}'"
                    if root not in self.allowed_modules:
                        allowed = ", ".join(sorted(self.allowed_modules))
                        return (
                            False,
                            f"Module '{alias.name}' is not allowed. Only {allowed} are permitted.",
                        )

            # Check from-imports (from x import y)
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    return False, "Relative imports are not allowed."
                root = node.module.split(".")[0]
                if root in self.forbidden_modules:
                    return False, f"Forbidden module import: '{node.module}'"
                if root not in self.allowed_modules:
                    allowed = ", ".join(sorted(self.allowed_modules))
                    return (
                        False,
                        f"Module '{node.module}' is not allowed. Only {allowed} are permitted.",
                    )
                for alias in node.names:
                    if alias.name in self.forbidden_builtins:
                        return (
                            False,
                            f"Forbidden import of builtin: '{alias.name}'",
                        )
                    if alias.asname and alias.asname in self.forbidden_builtins:
                        return False, f"Forbidden alias name: '{alias.asname}'"

            # Check function calls (ast.Call)
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node.func)
                if call_name and call_name in self.forbidden_builtins:
                    return False, f"Forbidden function call: '{call_name}'"

                root_module = self._get_root_name(node.func)
                if root_module and root_module in self.forbidden_modules:
                    attr_name = getattr(node.func, "attr", call_name or "call")
                    return (
                        False,
                        f"Forbidden call to module '{root_module}': '{attr_name}'",
                    )

            # Check forbidden builtins used as identifiers (ast.Name)
            elif isinstance(node, ast.Name):
                if node.id in self.forbidden_builtins:
                    return False, f"Forbidden name: '{node.id}'"

            # Check forbidden builtins accessed as attributes (e.g. obj.eval)
            elif isinstance(node, ast.Attribute):
                if node.attr in self.forbidden_builtins:
                    return False, f"Forbidden attribute access: '{node.attr}'"

            # Check function or class definitions shadowing forbidden builtins
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in self.forbidden_builtins:
                    return (
                        False,
                        f"Forbidden function definition: '{node.name}'",
                    )
            elif isinstance(node, ast.ClassDef):
                if node.name in self.forbidden_builtins:
                    return False, f"Forbidden class definition: '{node.name}'"

        return True, ""

    def parse(self, llm_response: str) -> tuple[str | None, str]:
        """Extract and validate Python code from an LLM response.

        Combines extract_code and validate_safety.

        Args:
            llm_response: The raw response string from an LLM.

        Returns:
            A tuple of (code_or_None, error_message). On success, returns
            (code, ""). On failure, returns (None, error_message).
        """
        code = self.extract_code(llm_response)
        if code is None:
            return None, "No code block found in response."
        if not code.strip():
            return None, "Extracted code block is empty."

        is_safe, error_msg = self.validate_safety(code)
        if not is_safe:
            return None, error_msg

        return code, ""

    def parse_result(self, llm_response: str) -> ParseResult:
        """Extract and validate Python code, returning a ParseResult dataclass.

        Args:
            llm_response: The raw response string from an LLM.

        Returns:
            ParseResult with code, error, and is_safe fields populated.
        """
        code, error = self.parse(llm_response)
        is_safe = (code is not None) and (not error)
        return ParseResult(code=code, error=error, is_safe=is_safe)

    @staticmethod
    def _get_call_name(func_node: ast.AST) -> str | None:
        """Extract the function name from a Call node's func field.

        Args:
            func_node: The AST node representing the function being called.

        Returns:
            The identifier or attribute name as a string, or None.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    @staticmethod
    def _get_root_name(node: ast.AST) -> str | None:
        """Retrieve the leftmost root identifier from an attribute chain.

        For example, for `os.path.join`, this returns `'os'`.

        Args:
            node: An AST expression node.

        Returns:
            The root identifier string, or None if not found.
        """
        curr = node
        while isinstance(curr, ast.Attribute):
            curr = curr.value
        if isinstance(curr, ast.Name):
            return curr.id
        return None