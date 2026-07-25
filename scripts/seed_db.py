"""Seed the SQLite analytics database used by the data agent (F5).

data/ is git-ignored, so the database file itself is never committed —
this script is the committed source of truth and rebuilds it byte-for-byte
identically on any machine:

    python scripts/seed_db.py

The schema deliberately mirrors the company described in
documents/company_handbook.md (Nordvik Analytics), so that F9 can answer
mixed questions: the NUMBER comes from this database, the REASON or POLICY
comes from the document corpus.

This script is the only place that opens the database for writing. The
data agent always connects read-only (see ai/db.py).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.config import get_settings  # noqa: E402

SCHEMA = """
CREATE TABLE departments (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    region  TEXT NOT NULL
);

CREATE TABLE employees (
    id             INTEGER PRIMARY KEY,
    full_name      TEXT NOT NULL,
    department_id  INTEGER NOT NULL REFERENCES departments(id),
    level          TEXT NOT NULL,
    hire_date      TEXT NOT NULL,
    annual_salary_eur INTEGER NOT NULL,
    is_active      INTEGER NOT NULL
);

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    tier        TEXT NOT NULL,
    country     TEXT NOT NULL,
    mrr_eur     INTEGER NOT NULL,
    signed_on   TEXT NOT NULL,
    churned_on  TEXT
);

CREATE TABLE support_tickets (
    id                     INTEGER PRIMARY KEY,
    customer_id            INTEGER NOT NULL REFERENCES customers(id),
    severity               INTEGER NOT NULL,
    opened_at              TEXT NOT NULL,
    first_response_minutes INTEGER NOT NULL,
    resolved_at            TEXT
);

CREATE TABLE expenses (
    id           INTEGER PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(id),
    category     TEXT NOT NULL,
    amount_eur   REAL NOT NULL,
    incurred_on  TEXT NOT NULL,
    status       TEXT NOT NULL
);

CREATE TABLE leave_requests (
    id           INTEGER PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(id),
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    days         INTEGER NOT NULL,
    status       TEXT NOT NULL
);
"""

DEPARTMENTS = [
    (1, "Engineering", "EU-North"),
    (2, "Data Science", "EU-West"),
    (3, "Customer Success", "EU-West"),
    (4, "Sales", "EU-North"),
]

EMPLOYEES = [
    (1, "Anna Lind", 1, "L4", "2021-03-15", 78000, 1),
    (2, "Bjorn Haugen", 1, "L5", "2019-09-01", 95000, 1),
    (3, "Carla Mendes", 2, "L3", "2022-06-20", 66000, 1),
    (4, "Dmitri Sokolov", 1, "L2", "2023-01-10", 54000, 1),
    (5, "Elena Rossi", 2, "L4", "2020-11-05", 81000, 1),
    (6, "Farid Karim", 3, "L3", "2022-02-14", 58000, 1),
    (7, "Greta Nowak", 3, "L2", "2023-08-01", 47000, 1),
    (8, "Hugo Bauer", 4, "L4", "2021-07-19", 72000, 1),
    (9, "Ines Duarte", 4, "L3", "2022-10-03", 61000, 1),
    (10, "Jonas Vik", 1, "L6", "2018-04-02", 118000, 1),
    (11, "Karin Osmond", 2, "L2", "2024-02-26", 52000, 1),
    (12, "Lars Pedersen", 3, "L1", "2024-09-16", 41000, 0),
]

CUSTOMERS = [
    (1, "Aurora Retail", "Enterprise", "Sweden", 4200, "2022-01-15", None),
    (2, "Baltic Freight", "Growth", "Estonia", 1800, "2022-05-02", None),
    (3, "Cedar Foods", "Growth", "Denmark", 1500, "2021-11-20", "2026-02-10"),
    (4, "Delta Insurance", "Enterprise", "Germany", 5600, "2020-08-11", None),
    (5, "Eiger Bank", "Enterprise", "Switzerland", 6100, "2023-03-07", None),
    (6, "Fjord Energy", "Growth", "Norway", 2100, "2022-09-30", "2026-01-22"),
    (7, "Granite Media", "Starter", "Ireland", 600, "2024-04-18", None),
    (8, "Helio Travel", "Starter", "Portugal", 750, "2023-12-05", "2026-03-14"),
    (9, "Iris Health", "Enterprise", "Netherlands", 4900, "2021-06-25", None),
    (10, "Juno Logistics", "Growth", "Poland", 1950, "2024-07-09", None),
]

SUPPORT_TICKETS = [
    (1, 4, 1, "2026-01-08", 32, "2026-01-08"),
    (2, 5, 1, "2026-02-11", 47, "2026-02-12"),
    (3, 1, 1, "2026-03-02", 25, "2026-03-02"),
    (4, 9, 1, "2026-04-19", 56, "2026-04-20"),
    (5, 2, 2, "2026-01-14", 130, "2026-01-16"),
    (6, 6, 2, "2026-01-20", 210, "2026-01-23"),
    (7, 10, 2, "2026-02-27", 95, "2026-02-28"),
    (8, 1, 2, "2026-03-30", 150, "2026-04-01"),
    (9, 4, 2, "2026-04-05", 175, "2026-04-07"),
    (10, 7, 3, "2026-02-03", 640, "2026-02-09"),
    (11, 3, 3, "2026-01-09", 720, "2026-01-15"),
    (12, 8, 3, "2026-02-25", 880, None),
    (13, 2, 3, "2026-03-18", 505, "2026-03-24"),
    (14, 9, 3, "2026-04-11", 690, "2026-04-18"),
    (15, 10, 3, "2026-04-22", 560, None),
]

EXPENSES = [
    (1, 1, "travel", 420.50, "2026-01-12", "approved"),
    (2, 2, "hotel", 180.00, "2026-01-13", "approved"),
    (3, 3, "training", 1200.00, "2026-02-01", "approved"),
    (4, 5, "hotel", 260.00, "2026-02-14", "approved"),
    (5, 8, "client_entertainment", 540.00, "2026-02-20", "rejected"),
    (6, 4, "equipment", 1150.00, "2026-03-03", "approved"),
    (7, 6, "meals", 55.00, "2026-03-11", "approved"),
    (8, 10, "travel", 890.25, "2026-03-19", "approved"),
    (9, 9, "training", 2000.00, "2026-04-02", "approved"),
    (10, 2, "meals", 48.75, "2026-04-08", "approved"),
    (11, 11, "equipment", 1200.00, "2026-04-15", "pending"),
    (12, 8, "travel", 610.00, "2026-04-21", "approved"),
]

LEAVE_REQUESTS = [
    (1, 1, "2026-02-16", "2026-02-20", 5, "approved"),
    (2, 3, "2026-03-09", "2026-03-13", 5, "approved"),
    (3, 2, "2026-04-06", "2026-04-17", 10, "approved"),
    (4, 5, "2026-01-19", "2026-01-23", 5, "approved"),
    (5, 6, "2026-05-04", "2026-05-08", 5, "pending"),
    (6, 8, "2026-02-02", "2026-02-06", 5, "approved"),
    (7, 10, "2026-03-23", "2026-04-03", 10, "approved"),
    (8, 4, "2026-04-27", "2026-04-30", 4, "rejected"),
]


def build(db_path: Path) -> dict:
    """Create the database from scratch and return row counts."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO departments VALUES (?,?,?)", DEPARTMENTS)
        conn.executemany(
            "INSERT INTO employees VALUES (?,?,?,?,?,?,?)", EMPLOYEES
        )
        conn.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)", CUSTOMERS
        )
        conn.executemany(
            "INSERT INTO support_tickets VALUES (?,?,?,?,?,?)", SUPPORT_TICKETS
        )
        conn.executemany("INSERT INTO expenses VALUES (?,?,?,?,?,?)", EXPENSES)
        conn.executemany(
            "INSERT INTO leave_requests VALUES (?,?,?,?,?,?)", LEAVE_REQUESTS
        )
        conn.commit()

        counts = {}
        for table in (
            "departments",
            "employees",
            "customers",
            "support_tickets",
            "expenses",
            "leave_requests",
        ):
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        return counts
    finally:
        conn.close()


def main() -> int:
    settings = get_settings()
    db_path = settings.sqlite_path
    counts = build(db_path)
    print(f"Database rebuilt at: {db_path}")
    for table, n in counts.items():
        print(f"  {table:<16} {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())