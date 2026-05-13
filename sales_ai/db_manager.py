"""SQLite + FTS5 storage layer for the prototype.

All DBs live under DATA_DIR. Each schema in `schemas.py` gets its own
file ({schema}.db) created lazily on first write. We use FTS5 with
external content tables and triggers so the FTS index stays in sync.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import schemas

DATA_DIR = Path(os.environ.get("SALES_AI_DATA_DIR", "data/sales_ai"))


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def db_path(schema_name: str) -> Path:
    _ensure_dir()
    return DATA_DIR / f"{schema_name}.db"


def connect(schema_name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(schema_name))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    init_schema(conn, schemas.get(schema_name))
    return conn


def init_schema(conn: sqlite3.Connection, schema: schemas.DBSchema) -> None:
    cols_sql = ",\n    ".join(f"{c} {t}" for c, t in schema.columns)
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS {schema.table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            {cols_sql},
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS {schema.table}_fts
        USING fts5(
            {", ".join(schema.fts_columns)},
            content='{schema.table}', content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS {schema.table}_ai
        AFTER INSERT ON {schema.table} BEGIN
            INSERT INTO {schema.table}_fts(rowid, {", ".join(schema.fts_columns)})
            VALUES (new.id, {", ".join("new." + c for c in schema.fts_columns)});
        END;

        CREATE TRIGGER IF NOT EXISTS {schema.table}_ad
        AFTER DELETE ON {schema.table} BEGIN
            INSERT INTO {schema.table}_fts({schema.table}_fts, rowid, {", ".join(schema.fts_columns)})
            VALUES ('delete', old.id, {", ".join("old." + c for c in schema.fts_columns)});
        END;
    """)
    conn.commit()


def insert_record(schema_name: str, record: dict[str, Any], source_file: str = "") -> int:
    """Insert a structured record. Unknown keys are ignored; missing keys → NULL."""
    schema = schemas.get(schema_name)
    valid_cols = {c for c, _ in schema.columns}
    cols = ["source_file"] + [c for c in record.keys() if c in valid_cols]
    placeholders = ", ".join("?" for _ in cols)
    values = [source_file] + [record.get(c) for c in cols[1:]]

    with connect(schema_name) as conn:
        cur = conn.execute(
            f"INSERT INTO {schema.table} ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return cur.lastrowid


def list_records(schema_name: str, limit: int = 100) -> list[sqlite3.Row]:
    schema = schemas.get(schema_name)
    with connect(schema_name) as conn:
        cur = conn.execute(
            f"SELECT * FROM {schema.table} ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def count_records(schema_name: str) -> int:
    schema = schemas.get(schema_name)
    with connect(schema_name) as conn:
        cur = conn.execute(f"SELECT COUNT(*) AS n FROM {schema.table}")
        return cur.fetchone()["n"]


def delete_record(schema_name: str, record_id: int) -> None:
    schema = schemas.get(schema_name)
    with connect(schema_name) as conn:
        conn.execute(f"DELETE FROM {schema.table} WHERE id = ?", (record_id,))
        conn.commit()


def search_fts(schema_name: str, query: str, limit: int = 3) -> list[sqlite3.Row]:
    """BM25-ranked FTS5 lookup. Returns full content rows joined with score."""
    schema = schemas.get(schema_name)
    if not query.strip():
        return []
    # FTS5 query syntax is picky and the trigram tokenizer needs >=3 char
    # spans. Split on whitespace, drop too-short tokens, and quote each so
    # punctuation can't break the parser. OR-join for recall.
    raw_tokens = [t for t in query.replace('"', " ").split() if t]
    tokens = [t for t in raw_tokens if len(t) >= 3] or raw_tokens
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)

    with connect(schema_name) as conn:
        try:
            cur = conn.execute(
                f"""
                SELECT c.*, bm25({schema.table}_fts) AS score
                FROM {schema.table}_fts f
                JOIN {schema.table} c ON c.id = f.rowid
                WHERE {schema.table}_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (match, limit),
            )
            return cur.fetchall()
        except sqlite3.OperationalError:
            return []


def export_db_bytes(schema_name: str) -> bytes:
    p = db_path(schema_name)
    if not p.exists():
        return b""
    return p.read_bytes()


def import_db_bytes(schema_name: str, data: bytes) -> None:
    _ensure_dir()
    db_path(schema_name).write_bytes(data)
