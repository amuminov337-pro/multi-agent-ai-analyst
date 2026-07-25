"""F5 acceptance check — Data agent / text-to-SQL (10 pts).

Done-when (from the guide): "a 'how many…' question returns the correct
number from the database."
Watch-out (rubric-mandated, points deducted without it): "read-only DB
user; reject any query that isn't a SELECT. Never let it run DROP/DELETE."

Four parts:
  1. Rebuild the database and show the schema the agent will see.
  2. Run the agent ALONE on three questions with known ground-truth
     answers, verified independently by hand-written SQL.
  3. Attack the safety layer directly with malicious statements and assert
     every one of them is rejected.
  4. Prove the connection itself is read-only at the OS level: a write
     attempt must fail even outside the keyword guard.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai import db  # noqa: E402
from ai.agents.data_sql import data_agent  # noqa: E402
from ai.state import new_state  # noqa: E402

import scripts.seed_db as seed_db  # noqa: E402
from ai.config import get_settings  # noqa: E402

# (question, ground-truth SQL, human label)
CASES = [
    (
        "How many customers have churned?",
        "SELECT COUNT(*) FROM customers WHERE churned_on IS NOT NULL",
        "churned customers",
    ),
    (
        "How many employees work in the Engineering department?",
        "SELECT COUNT(*) FROM employees e JOIN departments d "
        "ON e.department_id = d.id WHERE d.name = 'Engineering'",
        "engineering headcount",
    ),
    (
        "What is the average first response time in minutes for severity 1 tickets?",
        "SELECT AVG(first_response_minutes) FROM support_tickets WHERE severity = 1",
        "avg severity-1 first response",
    ),
]

ATTACKS = [
    "DROP TABLE employees",
    "DELETE FROM customers WHERE id = 1",
    "UPDATE employees SET annual_salary_eur = 0",
    "INSERT INTO departments VALUES (9, 'Fake', 'EU')",
    "ALTER TABLE customers ADD COLUMN hacked TEXT",
    "SELECT 1; DROP TABLE expenses",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE 'evil.db' AS evil",
    "```sql\nDELETE FROM expenses\n```",
    "SELECT * FROM customers -- ; DROP TABLE customers",
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ground_truth(sql: str) -> str:
    """Compute the expected answer with hand-written, trusted SQL."""
    conn = db.get_connection()
    try:
        value = conn.execute(sql).fetchone()[0]
    finally:
        conn.close()
    return value


def acceptable_forms(value) -> list:
    """String forms an LLM-generated result might legitimately take."""
    forms = {str(value)}
    if isinstance(value, float):
        forms.add(str(int(value)) if value.is_integer() else f"{value:.1f}")
        forms.add(f"{value:.1f}")
        forms.add(f"{value:.2f}")
    if isinstance(value, int):
        forms.add(f"{value}.0")
        forms.add(f"{float(value):.1f}")
    return sorted(forms)


def part_1_database() -> bool:
    section("1) REBUILD DATABASE")
    settings = get_settings()
    counts = seed_db.build(settings.sqlite_path)
    print(f"database : {settings.sqlite_path}")
    for table, n in counts.items():
        print(f"  {table:<16} {n} rows")
    schema = db.get_schema()
    tables = schema.count("TABLE ")
    print(f"schema exposed to the LLM: {tables} tables")
    return tables == 6


def part_2_agent() -> bool:
    section("2) DATA AGENT ALONE — GROUND-TRUTH QUESTIONS")
    failures = 0
    for question, truth_sql, label in CASES:
        expected = ground_truth(truth_sql)
        forms = acceptable_forms(expected)

        state = new_state(question)
        update = data_agent(state)
        result = update["sql_result"] or ""

        ok = any(form in result for form in forms)
        print(f"\nQ: {question}")
        print(f"   ground truth ({label}) : {expected}")
        print(f"   accepted forms         : {forms}")
        print(f"   step                   : {update['steps'][-1]}")
        for line in result.splitlines():
            print(f"   | {line}")
        print(f"   result                 : {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
    return failures == 0


def part_3_guard() -> bool:
    section("3) SAFETY GUARD — MALICIOUS SQL MUST BE REJECTED")
    passed = 0
    for attack in ATTACKS:
        label = attack.replace("\n", " ")[:52]
        try:
            db.assert_read_only(attack)
            print(f"  [FAIL] ACCEPTED (!!) : {label}")
        except db.UnsafeQueryError as exc:
            print(f"  [OK] rejected        : {label}")
            print(f"         reason        : {exc}")
            passed += 1
    print(f"\nrejected {passed}/{len(ATTACKS)} attacks")

    # A legitimate read must still pass, or the guard is useless.
    try:
        db.assert_read_only("SELECT COUNT(*) FROM customers")
        print("  [OK] legitimate SELECT still allowed")
        legit = True
    except db.UnsafeQueryError as exc:
        print(f"  [FAIL] legitimate SELECT blocked: {exc}")
        legit = False

    return passed == len(ATTACKS) and legit


def part_4_readonly_connection() -> bool:
    section("4) OS-LEVEL READ-ONLY CONNECTION")
    conn = db.get_connection()
    try:
        conn.execute("CREATE TABLE should_not_exist (id INTEGER)")
        print("  [FAIL] write succeeded — connection is NOT read-only")
        return False
    except sqlite3.OperationalError as exc:
        print(f"  [OK] write blocked by SQLite: {exc}")
        return True
    finally:
        conn.close()


def main() -> int:
    results = {
        "database rebuilt": part_1_database(),
        "agent answers correctly": part_2_agent(),
        "malicious SQL rejected": part_3_guard(),
        "connection is read-only": part_4_readonly_connection(),
    }

    section("RESULT")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nPASS — F5 done (10/10)")
        print("  - text-to-SQL returns the correct number from the database")
        print("  - SELECT-only guard rejects every write/DDL attempt")
        print("  - database is opened read-only at the OS level")
        return 0
    print("\nFAIL — F5 not complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())