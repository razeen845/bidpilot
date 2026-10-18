"""AST allow-list check for AI-written analysis code. Runs before any code reaches the sandbox."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "hasattr", "globals", "locals",
    "vars", "dir", "input", "breakpoint", "help", "exit", "quit", "memoryview", "type", "super",
    "classmethod", "staticmethod", "property",
    "os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "httpx", "urllib", "importlib",
    "builtins", "io", "pickle", "ctypes", "multiprocessing", "threading", "signal",
}
# pandas / numpy attributes that touch files, evaluate arbitrary strings or reach internals.
# DataFrame.query is allowed: its expressions cannot reach dunders (strings with "__" are rejected below) and
# the code runs in the isolated sandbox process anyway.
FORBIDDEN_ATTRS = {
    "eval", "pipe", "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt",
    "savetxt", "genfromtxt", "memmap", "ctypeslib", "lib", "f2py", "testing", "distutils", "io", "api",
    "core", "compat", "util", "plotting", "options", "set_option", "system", "popen",
}
PANDAS_WRITERS = re.compile(r"^to_(csv|parquet|excel|json|pickle|sql|hdf|feather|html|latex|markdown|clipboard|"
                            r"stata|xml|orc|gbq|string)$")
MAX_CODE_CHARS = 6000


@dataclass
class PolicyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_code(code: str) -> PolicyResult:
    reasons: list[str] = []
    if len(code) > MAX_CODE_CHARS:
        return PolicyResult(False, [f"code longer than {MAX_CODE_CHARS} characters"])
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return PolicyResult(False, [f"syntax error: {e}"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS or "." in a.name:
                    reasons.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in ALLOWED_IMPORTS or "." in mod:
                reasons.append(f"import not allowed: from {mod}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                reasons.append(f"name not allowed: {node.id}")
            elif node.id.startswith("__"):
                reasons.append(f"dunder name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                reasons.append(f"private/dunder attribute not allowed: .{a}")
            elif a in FORBIDDEN_ATTRS or a in FORBIDDEN_NAMES:
                reasons.append(f"attribute not allowed: .{a}")
            elif a.startswith("read_") or PANDAS_WRITERS.match(a):
                reasons.append(f"file I/O not allowed: .{a}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
                               ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom)):
            reasons.append(f"statement not allowed: {type(node).__name__}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            reasons.append("strings containing '__' are not allowed")
    # stable, de-duplicated
    seen, out = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return PolicyResult(not out, out)
