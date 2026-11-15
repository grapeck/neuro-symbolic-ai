import ast
import pytest
from src.parser import CodeParser, ParseResult


def test_parse_result_dataclass():
    res = ParseResult(code="x = 1", error="", is_safe=True)
    assert res.code == "x = 1"
    assert res.error == ""
    assert res.is_safe is True

    res2 = ParseResult.from_tuple(("y = 2", ""))
    assert res2.code == "y = 2"
    assert res2.error == ""
    assert res2.is_safe is True

    res3 = ParseResult.from_tuple((None, "Some error"))
    assert res3.code is None
    assert res3.error == "Some error"
    assert res3.is_safe is False


def test_extract_code_python_block():
    parser = CodeParser()
    resp = "Here is the code:\n```python\nimport sympy as sp\nx = sp.Symbol('x')\n```\nHope it helps!"
    code = parser.extract_code(resp)
    assert code == "import sympy as sp\nx = sp.Symbol('x')"


def test_extract_code_py_block():
    parser = CodeParser()
    resp = "```py\nx = 1\n```"
    assert parser.extract_code(resp) == "x = 1"


def test_extract_code_generic_block():
    parser = CodeParser()
    resp = "```\nx = 2\n```"
    assert parser.extract_code(resp) == "x = 2"


def test_extract_code_none():
    parser = CodeParser()
    assert parser.extract_code("Just some text") is None
    assert parser.extract_code("") is None


def test_validate_safety_valid_sympy():
    parser = CodeParser()
    code = """
import sympy as sp
from sympy import symbols, Eq, solve
from math import sqrt, pi

x, y = symbols('x y')
eq = Eq(x**2 + y**2, 25)
sol = solve(eq, x)
"""
    is_safe, err = parser.validate_safety(code)
    assert is_safe is True
    assert err == ""


def test_validate_safety_forbidden_imports():
    parser = CodeParser()
    forbidden = [
        "import os",
        "import sys",
        "import subprocess",
        "import shutil",
        "import socket",
        "import requests",
        "import urllib",
        "import http",
        "import pathlib",
        "from os import path",
        "from sys import argv",
        "from subprocess import Popen",
    ]
    for code in forbidden:
        is_safe, err = parser.validate_safety(code)
        assert is_safe is False, f"Expected unsafe for: {code}"
        assert "Forbidden module import" in err or "not allowed" in err


def test_validate_safety_unauthorized_imports():
    parser = CodeParser()
    unauthorized = [
        "import numpy",
        "import torch",
        "import json",
        "from scipy import optimize",
    ]
    for code in unauthorized:
        is_safe, err = parser.validate_safety(code)
        assert is_safe is False
        assert "not allowed" in err


def test_validate_safety_forbidden_builtins_calls():
    parser = CodeParser()
    builtins_calls = [
        "open('test.txt')",
        "eval('1+1')",
        "exec('x = 1')",
        "compile('x = 1', '<string>', 'exec')",
        "__import__('os')",
        "globals()",
        "locals()",
        "vars()",
        "getattr(obj, 'x')",
        "setattr(obj, 'x', 1)",
        "delattr(obj, 'x')",
    ]
    for code in builtins_calls:
        is_safe, err = parser.validate_safety(code)
        assert is_safe is False, f"Expected unsafe for call: {code}"
        assert "Forbidden" in err


def test_validate_safety_forbidden_names():
    parser = CodeParser()
    names = [
        "x = eval",
        "f = open",
        "g = globals",
    ]
    for code in names:
        is_safe, err = parser.validate_safety(code)
        assert is_safe is False
        assert "Forbidden name" in err


def test_validate_safety_forbidden_nodes():
    parser = CodeParser()
    # ast.Delete
    is_safe, err = parser.validate_safety("del x")
    assert is_safe is False
    assert "Forbidden AST node" in err

    # ast.Global
    is_safe, err = parser.validate_safety("def foo():\n    global x\n    x = 1")
    assert is_safe is False
    assert "Forbidden AST node" in err

    # ast.Nonlocal
    is_safe, err = parser.validate_safety("def outer():\n    x = 1\n    def inner():\n        nonlocal x\n        x = 2")
    assert is_safe is False
    assert "Forbidden AST node" in err


def test_validate_safety_syntax_error():
    parser = CodeParser()
    is_safe, err = parser.validate_safety("def foo(:")
    assert is_safe is False
    assert "Syntax error" in err


def test_parse_success():
    parser = CodeParser()
    resp = "```python\nimport sympy as sp\nx = sp.Symbol('x')\n```"
    code, err = parser.parse(resp)
    assert code == "import sympy as sp\nx = sp.Symbol('x')"
    assert err == ""

    res = parser.parse_result(resp)
    assert res.code == code
    assert res.error == ""
    assert res.is_safe is True


def test_parse_no_code():
    parser = CodeParser()
    code, err = parser.parse("The answer is 42.")
    assert code is None
    assert "No code block found" in err

    res = parser.parse_result("The answer is 42.")
    assert res.code is None
    assert res.is_safe is False


def test_parse_unsafe_code():
    parser = CodeParser()
    resp = "```python\nimport os\nos.system('echo hi')\n```"
    code, err = parser.parse(resp)
    assert code is None
    assert "Forbidden" in err

    res = parser.parse_result(resp)
    assert res.code is None
    assert res.is_safe is False
