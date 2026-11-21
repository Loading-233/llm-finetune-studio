"""SQLite 存储层。"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 0,
    stats_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT,
    base_model TEXT,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    output_dir TEXT DEFAULT '',
    adapter_id TEXT,
    config_json TEXT DEFAULT '{}',
    metrics_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adapters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    base_model TEXT,
    dataset_id TEXT,
    job_id TEXT,
    meta_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evals (
    id TEXT PRIMARY KEY,
    adapter_id TEXT,
    dataset_id TEXT,
    status TEXT NOT NULL,
    report_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def execute(sql: str, params: tuple = ()) -> None:
    with connect() as conn:
        conn.execute(sql, params)


def insert(table: str, payload: dict) -> None:
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(payload.values()))


def update(table: str, row_id: str, payload: dict) -> None:
    assignments = ", ".join(f"{key} = ?" for key in payload)
    execute(f"UPDATE {table} SET {assignments} WHERE id = ?", (*payload.values(), row_id))


def one(table: str, row_id: str) -> dict | None:
    rows = query(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
    return rows[0] if rows else None


# ---------------------------------------------------------------- 常用查询

def list_datasets() -> list[dict]:
    return query("SELECT * FROM datasets ORDER BY created_at DESC")


def list_jobs(limit: int = 30) -> list[dict]:
    return query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))


def list_adapters() -> list[dict]:
    return query("SELECT * FROM adapters ORDER BY created_at DESC")


def hydrate(row: dict | None, json_fields: tuple[str, ...] = ()) -> dict | None:
    if not row:
        return None
    for field in json_fields:
        key = field.replace("_json", "")
        try:
            row[key] = json.loads(row.get(field) or "{}")
        except json.JSONDecodeError:
            row[key] = {}
    return row
