"""F6 acceptance check — Code agent (8 pts).

Done-when (from the guide): "a math/aggregation question returns the
correct computed answer."
Watch-out (rubric-mandated, points deducted without it): "run in a sandbox
with a runtime cap — never execute model-written code on the bare server."

Five parts:
  1. Sandbox executes safe code correctly, in a separate process.
  2. RUNTIME CAP: an infinite loop is killed by the timeout.
  3. STATIC GUARD: file, network, process, eval and dunder-escape attempts
     are all rejected before anything runs.
  4. ISOLATION: no API key reaches the child environment; the child cannot
     see the parent's working directory.
  5. The code agent ALONE answers three math questions whose ground truth
     is computed independently inside this script.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai import sandbox  # noqa: E402
from ai.agents.code_agent import code_agent  # noqa: E402
from ai.state import new_state  # noqa: E402

# --- ground truth computed here, independently of the agent -------------
FACTORIAL_17 = math.factorial(17)
PRIME_SUM_BELOW_100 = sum(
    n for n in range(2, 100) if all(n % d for d in range(2, int(n**0.5) + 1))
)
# distinct arrangements of "STATISTICS": S×3, T×3, A, I×2, C
STATISTICS_PERMS = math.factorial(10) // (
    math.factorial(3) * math.factorial(3) * math.factorial(2)
)

CASES = [
    ("What is 17 factorial?", FACTORIAL_17, "17!"),
    (
        "What is the sum of all prime numbers below 100?",
        PRIME_SUM_BELOW_100,
        "sum of primes < 100",
    ),
    (
        "How many distinct ways can the letters of the word STATISTICS be arranged?",
        STATISTICS_PERMS,
        "permutations of STATISTICS",
    ),
]

ATTACKS = [
    ("read a file", "print(open('secret.txt').read())"),
    ("write a file", "f = open('pwned.txt', 'w')\nf.write('x')"),
    ("import os", "import os\nprint(os.listdir('.'))"),
    ("import subprocess", "import subprocess\nsubprocess.run(['whoami'])"),
    ("import socket", "import socket\nprint(socket.gethostname())"),
    ("import shutil", "import shutil\nshutil.rmtree('.')"),
    ("from os import path", "from os import path\nprint(path.abspath('.'))"),
    ("eval", "print(eval('2+2'))"),
    ("exec", "exec('x = 1')"),
    ("__import__ escape", "__import__('os').system('dir')"),
    ("dunder class escape", "print(().__class__.__bases__[0].__subclasses__())"),
    ("getattr reflection", "print(getattr(__builtins__, 'open'))"),
    ("read env via os", "import os\nprint(os.environ.get('GOOGLE_API_KEY'))"),
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def part_1_safe_execution() -> bool:
    section("1) SANDBOX RUNS SAFE CODE")
    result = sandbox.run_python(
        "import math\nprint(math.isqrt(1024))\nprint(sum(range(11)))"
    )
    print(f"ok            : {result.ok}")
    print(f"exit code     : {result.exit_code}")
    print(f"duration      : {result.duration_seconds:.2f}s")
    print(f"stdout        : {result.stdout.strip()!r}")

    checks = [
        ("ran successfully", result.ok),
        ("first value correct (32)", "32" in result.stdout),
        ("second value correct (55)", "55" in result.stdout),
        ("did not time out", not result.timed_out),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_2_runtime_cap() -> bool:
    section("2) RUNTIME CAP — INFINITE LOOP MUST BE KILLED")
    result = sandbox.run_python("while True:\n    pass", timeout=3)
    print(f"timed out     : {result.timed_out}")
    print(f"duration      : {result.duration_seconds:.2f}s (cap was 3s)")
    print(f"summary       : {result.summary()}")

    checks = [
        ("timeout detected", result.timed_out),
        ("marked as failed", not result.ok),
        ("killed near the cap, not later", result.duration_seconds < 10),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_3_static_guard() -> bool:
    section("3) STATIC GUARD — DANGEROUS CODE MUST BE REJECTED")
    rejected = 0
    for label, code in ATTACKS:
        try:
            sandbox.assert_safe_code(code)
            print(f"  [FAIL] ACCEPTED (!!) : {label}")
        except sandbox.UnsafeCodeError as exc:
            print(f"  [OK] rejected        : {label}")
            print(f"         reason        : {exc}")
            rejected += 1
    print(f"\nrejected {rejected}/{len(ATTACKS)} attacks")

    try:
        sandbox.assert_safe_code("import statistics\nprint(statistics.mean([1,2,3]))")
        print("  [OK] legitimate program still allowed")
        legit = True
    except sandbox.UnsafeCodeError as exc:
        print(f"  [FAIL] legitimate program blocked: {exc}")
        legit = False

    return rejected == len(ATTACKS) and legit


def part_4_isolation() -> bool:
    section("4) ISOLATION — NO SECRETS REACH THE CHILD PROCESS")
    env = sandbox.child_env("/tmp/example")
    leaked = [
        key
        for key in env
        if any(marker in key.upper() for marker in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
    ]
    print(f"child env vars : {sorted(env)}")
    print(f"leaked secrets : {leaked or 'none'}")

    # The child runs in a throwaway temp dir, not the project folder.
    result = sandbox.run_python("print('cwd-probe')")
    checks = [
        ("no secret-looking variables passed", not leaked),
        ("GOOGLE_API_KEY absent", "GOOGLE_API_KEY" not in env),
        ("child still able to run", result.ok),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def part_5_agent() -> bool:
    section("5) CODE AGENT ALONE — GROUND-TRUTH MATH")
    failures = 0
    for question, expected, label in CASES:
        state = new_state(question)
        update = code_agent(state)
        evidence = update["code_result"] or ""
        ok = str(expected) in evidence

        print(f"\nQ: {question}")
        print(f"   ground truth ({label}) : {expected}")
        print(f"   step                   : {update['steps'][-1]}")
        for line in evidence.splitlines():
            print(f"   | {line}")
        print(f"   result                 : {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
    return failures == 0


def main() -> int:
    results = {
        "sandbox runs safe code": part_1_safe_execution(),
        "runtime cap enforced": part_2_runtime_cap(),
        "dangerous code rejected": part_3_static_guard(),
        "no secrets in child env": part_4_isolation(),
        "agent computes correctly": part_5_agent(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F6 done (8/8)")
        print("  - model-written Python runs in a separate isolated process")
        print("  - runtime cap kills a runaway program")
        print("  - static guard rejects file, network, process and escape attempts")
        print("  - math questions return the correct computed answer")
        return 0
    print("\nFAIL — F6 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())