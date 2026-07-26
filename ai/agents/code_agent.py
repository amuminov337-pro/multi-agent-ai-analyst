"""Code agent — writes and runs Python for exact computation (F6).

LLMs are unreliable at arithmetic, so numeric questions are answered by
generating a short program and running it. The generated code is untrusted
input: it goes through ai/sandbox.py, which statically rejects anything
outside a stdlib whitelist and then runs the rest in an isolated process
with a hard runtime cap.

Both the program and its output are written into state["code_result"], so
the critic (F8) and the frontend (F13) can show HOW the number was reached
rather than just asserting it.

This node never raises. A rejected program, a crash, or a timeout all come
back as recorded evidence so the supervisor can re-route.
"""

from __future__ import annotations

import re

from ai.llm import ask
from ai.sandbox import UnsafeCodeError, run_python
from ai.state import AgentState, push_step

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

CODE_PROMPT = """Write a short Python program that answers the question.

Question: {question}

Rules:
- Output ONLY the code. No explanation, no markdown fences.
- print() the final answer, and print nothing else.
- The program runs in a sandbox: only these modules may be imported —
  math, statistics, decimal, fractions, itertools, functools, collections,
  operator, json, re, datetime, string, random.
- No file access, no network, no os/sys/subprocess, no eval/exec/open.
  A program using any of those is rejected before it runs.
- It must finish in a couple of seconds; no infinite loops.
"""


def extract_code(raw: str) -> str:
    """Pull the program out of an LLM response.

    Formatting only, never security: whatever comes out still has to pass
    every check in the sandbox.
    """
    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    return text.strip()


def generate_code(question: str) -> str:
    """Ask the LLM for a program (raw text, not yet validated)."""
    return extract_code(ask(CODE_PROMPT.format(question=question)))


def answer_with_code(question: str) -> str:
    """Full generate-and-run round trip. Returns a formatted evidence string.

    Raises UnsafeCodeError when the sandbox refuses the program.
    """
    code = generate_code(question)
    result = run_python(code)
    return f"Code:\n{code}\nOutput:\n{result.summary()}"


def code_agent(state: AgentState) -> dict:
    """LangGraph node: question -> program + output in state["code_result"].

    Never raises. Failure modes recorded in steps:
      * code(rejected)  — static analysis refused the program
      * code(timeout)   — the program hit the runtime cap
      * code(error)     — the program ran but exited non-zero
      * code(failed: X) — anything else (API/network)
    """
    question = state["question"]

    try:
        code = generate_code(question)
        result = run_python(code)
        evidence = f"Code:\n{code}\nOutput:\n{result.summary()}"
        if result.timed_out:
            label = "code(timeout)"
        elif not result.ok:
            label = "code(error)"
        else:
            label = "code"
    except UnsafeCodeError as exc:
        evidence = f"Program rejected by the sandbox guard: {exc}"
        label = "code(rejected)"
    except Exception as exc:
        evidence = f"Code agent failed ({type(exc).__name__}): {exc}"
        label = f"code(failed: {type(exc).__name__})"

    return {
        "code_result": evidence,
        "steps": push_step(state, label),
    }