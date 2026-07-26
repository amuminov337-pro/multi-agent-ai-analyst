"""Sandboxed execution of model-written Python (F6).

Rubric requirement: "run in a sandbox with a runtime cap — never execute
model-written code on the bare server." LangChain's PythonREPLTool calls
exec() inside this very process, which is exactly what the rubric warns
against, so execution happens here instead, behind two layers:

  1. STATIC ANALYSIS (before anything runs). The code is parsed with `ast`
     and rejected unless it only imports whitelisted stdlib modules and
     touches no dangerous builtin. AST beats regex here: `import os` and
     `__import__("os")` and `().__class__.__bases__` are all structural
     patterns, not string patterns. Rejection is preferred over rewriting —
     we never try to "fix" untrusted code.

  2. PROCESS ISOLATION (when it does run). The code executes in a separate
     interpreter (`python -I`, isolated mode) inside a throwaway temp
     directory, with a scrubbed environment that carries no API keys, a
     hard wall-clock timeout, a capped output size, and — on POSIX —
     address-space and CPU rlimits. A runaway loop is killed by the
     timeout; a crash cannot take the server down with it.

The backend is deliberately swappable: only run_python() knows how code is
executed, so moving to a container later would not change any caller.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ai.config import get_settings

# Pure-computation stdlib only: no filesystem, no network, no processes.
ALLOWED_MODULES = frozenset(
    {
        "math",
        "cmath",
        "statistics",
        "decimal",
        "fractions",
        "random",
        "itertools",
        "functools",
        "operator",
        "collections",
        "heapq",
        "bisect",
        "json",
        "re",
        "string",
        "textwrap",
        "datetime",
        "calendar",
        "time",
        "unicodedata",
        "copy",
        "numbers",
        "array",
        "enum",
        "typing",
        "dataclasses",
        "pprint",
    }
)

# Builtins that reach outside the computation: I/O, dynamic execution,
# attribute reflection (the classic sandbox-escape toolkit).
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "input",
        "exit",
        "quit",
        "help",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

MAX_OUTPUT_CHARS = 4000
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024  # POSIX only

# Environment variables the child is allowed to inherit. Everything else —
# including every API key — is dropped.
ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LANG",
    "LC_ALL",
)


class UnsafeCodeError(ValueError):
    """Raised when static analysis rejects model-written code."""


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed run."""

    ok: bool
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    exit_code: Optional[int]

    def summary(self) -> str:
        """Compact evidence string for the agent state and the critic."""
        if self.timed_out:
            return f"(timed out after {self.duration_seconds:.1f}s — no output)"
        if self.ok:
            return self.stdout.strip() or "(no output — the code printed nothing)"
        detail = self.stderr.strip() or "(no error message)"
        return f"(execution failed, exit code {self.exit_code})\n{detail}"


def _root_module(name: str) -> str:
    """`collections.abc` -> `collections`."""
    return name.split(".", 1)[0]


def _is_dunder(name: str) -> bool:
    """True for names like __class__, __globals__, __subclasses__."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def assert_safe_code(code: str) -> str:
    """Static gate (layer 1). Returns the code or raises UnsafeCodeError."""
    text = (code or "").strip()
    if not text:
        raise UnsafeCodeError("Empty program.")

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Invalid Python syntax: {exc.msg} (line {exc.lineno}).")

    for node in ast.walk(tree):
        # import x, y
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root not in ALLOWED_MODULES:
                    raise UnsafeCodeError(
                        f"Import of '{alias.name}' is not allowed. "
                        f"Permitted modules: {', '.join(sorted(ALLOWED_MODULES))}."
                    )

        # from x import y
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = _root_module(module)
            if not module or root not in ALLOWED_MODULES:
                raise UnsafeCodeError(
                    f"Import from '{module or '.'}' is not allowed. "
                    f"Permitted modules: {', '.join(sorted(ALLOWED_MODULES))}."
                )

        # bare names: open(...), eval(...), __import__(...)
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise UnsafeCodeError(
                    f"Use of '{node.id}' is not allowed in sandboxed code."
                )
            if _is_dunder(node.id):
                raise UnsafeCodeError(
                    f"Dunder access '{node.id}' is not allowed — it is a common "
                    "sandbox-escape route."
                )

        # attribute access: ().__class__, obj.__globals__
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                raise UnsafeCodeError(
                    f"Dunder attribute access '.{node.attr}' is not allowed — "
                    "it is a common sandbox-escape route."
                )

    return text


def child_env(tmpdir: str) -> Dict[str, str]:
    """Minimal environment for the child process — no secrets pass through."""
    env: Dict[str, str] = {}
    for key in ENV_ALLOWLIST:
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["TEMP"] = tmpdir
    env["TMP"] = tmpdir
    env["TMPDIR"] = tmpdir
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _posix_limits(timeout: int):
    """rlimit hook for POSIX; None on Windows, where rlimits don't exist."""
    if os.name != "posix":
        return None

    def _apply() -> None:  # pragma: no cover - platform specific
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)
        )
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return _apply


def _cap(text: str) -> str:
    """Truncate oversized output so a runaway print can't flood the context."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} chars)"


def run_python(code: str, timeout: Optional[int] = None) -> SandboxResult:
    """Validate, then execute model-written Python in an isolated process.

    Raises UnsafeCodeError if static analysis rejects the code — that is a
    refusal, not a failed run, and callers report it differently.
    """
    settings = get_settings()
    limit = timeout if timeout is not None else settings.code_timeout_seconds
    safe_code = assert_safe_code(code)

    with tempfile.TemporaryDirectory(prefix="capstone_sandbox_") as tmpdir:
        script = Path(tmpdir) / "program.py"
        script.write_text(safe_code, encoding="utf-8")

        # -I = isolated mode: ignores PYTHON* env vars and user site-packages,
        #      so the child cannot be steered by the parent's environment.
        # -B = never write .pyc files.
        command: List[str] = [sys.executable, "-I", "-B", str(script)]

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=tmpdir,
                env=child_env(tmpdir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
                check=False,
                preexec_fn=_posix_limits(limit),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                stdout="",
                stderr=f"Execution exceeded the {limit}s runtime cap and was killed.",
                timed_out=True,
                duration_seconds=time.perf_counter() - started,
                exit_code=None,
            )

        duration = time.perf_counter() - started
        return SandboxResult(
            ok=proc.returncode == 0,
            stdout=_cap(proc.stdout or ""),
            stderr=_cap(proc.stderr or ""),
            timed_out=False,
            duration_seconds=duration,
            exit_code=proc.returncode,
        )