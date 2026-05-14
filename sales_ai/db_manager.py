"""Single-SQLite storage layer for the prototype.

All tables (`deals` master + 5 child tables) live in one DB file so that
deal-scoped joins and full-bundle retrieval stay simple. FTS5 mirrors
use the trigram tokenizer for Japanese substring matching.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from . import schemas

DATA_DIR = Path(os.environ.get("SALES_AI_DATA_DIR", "data/sales_ai"))
DB_FILE = DATA_DIR / "sales_ai.db"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_all(conn)
    return conn


def _init_all(conn: sqlite3.Connection) -> None:
    # deals master
    deal_cols_sql = ",\n        ".join(f"{c} {t}" for c, t in schemas.DEAL_COLUMNS)
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {deal_cols_sql},
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS deals_fts
        USING fts5(
            {", ".join(schemas.DEAL_FTS_COLUMNS)},
            content='deals', content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS deals_ai AFTER INSERT ON deals BEGIN
            INSERT INTO deals_fts(rowid, {", ".join(schemas.DEAL_FTS_COLUMNS)})
            VALUES (new.id, {", ".join("new." + c for c in schemas.DEAL_FTS_COLUMNS)});
        END;
        CREATE TRIGGER IF NOT EXISTS deals_ad AFTER DELETE ON deals BEGIN
            INSERT INTO deals_fts(deals_fts, rowid, {", ".join(schemas.DEAL_FTS_COLUMNS)})
            VALUES ('delete', old.id, {", ".join("old." + c for c in schemas.DEAL_FTS_COLUMNS)});
        END;
        CREATE TRIGGER IF NOT EXISTS deals_au AFTER UPDATE ON deals BEGIN
            INSERT INTO deals_fts(deals_fts, rowid, {", ".join(schemas.DEAL_FTS_COLUMNS)})
            VALUES ('delete', old.id, {", ".join("old." + c for c in schemas.DEAL_FTS_COLUMNS)});
            INSERT INTO deals_fts(rowid, {", ".join(schemas.DEAL_FTS_COLUMNS)})
            VALUES (new.id, {", ".join("new." + c for c in schemas.DEAL_FTS_COLUMNS)});
        END;
    """)

    # child tables
    for sc in schemas.all_children():
        cols_sql = ",\n            ".join(f"{c} {t}" for c, t in sc.columns)
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {sc.table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id INTEGER NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
                source_file TEXT,
                {cols_sql},
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS {sc.table}_deal_idx ON {sc.table}(deal_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS {sc.table}_fts
            USING fts5(
                {", ".join(sc.fts_columns)},
                content='{sc.table}', content_rowid='id',
                tokenize='trigram'
            );

            CREATE TRIGGER IF NOT EXISTS {sc.table}_ai AFTER INSERT ON {sc.table} BEGIN
                INSERT INTO {sc.table}_fts(rowid, {", ".join(sc.fts_columns)})
                VALUES (new.id, {", ".join("new." + c for c in sc.fts_columns)});
            END;
            CREATE TRIGGER IF NOT EXISTS {sc.table}_ad AFTER DELETE ON {sc.table} BEGIN
                INSERT INTO {sc.table}_fts({sc.table}_fts, rowid, {", ".join(sc.fts_columns)})
                VALUES ('delete', old.id, {", ".join("old." + c for c in sc.fts_columns)});
            END;
        """)
    conn.commit()


# --- deal CRUD ---------------------------------------------------------------

def create_deal(record: dict[str, Any]) -> int:
    cols = [c for c, _ in schemas.DEAL_COLUMNS if c in record]
    vals = [record.get(c) for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO deals ({', '.join(cols)}) VALUES ({placeholders})", vals
        )
        conn.commit()
        return cur.lastrowid


def update_deal(deal_id: int, record: dict[str, Any]) -> None:
    cols = [c for c, _ in schemas.DEAL_COLUMNS if c in record]
    if not cols:
        return
    sets = ", ".join(f"{c}=?" for c in cols)
    vals = [record.get(c) for c in cols] + [deal_id]
    with connect() as conn:
        conn.execute(f"UPDATE deals SET {sets} WHERE id=?", vals)
        conn.commit()


def delete_deal(deal_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM deals WHERE id=?", (deal_id,))
        conn.commit()


def list_deals(only_champions: bool = False) -> list[sqlite3.Row]:
    with connect() as conn:
        q = "SELECT * FROM deals"
        if only_champions:
            q += " WHERE outcome='成約'"
        q += " ORDER BY id DESC"
        return conn.execute(q).fetchall()


def get_deal(deal_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM deals WHERE id=?", (deal_id,)).fetchone()


def count_deals(only_champions: bool = False) -> int:
    with connect() as conn:
        q = "SELECT COUNT(*) AS n FROM deals"
        if only_champions:
            q += " WHERE outcome='成約'"
        return conn.execute(q).fetchone()["n"]


# --- child CRUD --------------------------------------------------------------

def insert_child(schema_name: str, deal_id: int, record: dict[str, Any], source_file: str = "") -> int:
    sc = schemas.get_child(schema_name)
    valid = {c for c, _ in sc.columns}
    keep_cols = [c for c in record.keys() if c in valid]
    cols = ["deal_id", "source_file"] + keep_cols
    vals = [deal_id, source_file] + [record.get(c) for c in keep_cols]
    placeholders = ", ".join("?" for _ in cols)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO {sc.table} ({', '.join(cols)}) VALUES ({placeholders})", vals
        )
        conn.commit()
        return cur.lastrowid


def list_children(schema_name: str, deal_id: int | None = None, limit: int = 200) -> list[sqlite3.Row]:
    sc = schemas.get_child(schema_name)
    with connect() as conn:
        if deal_id is None:
            return conn.execute(
                f"SELECT * FROM {sc.table} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return conn.execute(
            f"SELECT * FROM {sc.table} WHERE deal_id=? ORDER BY id DESC LIMIT ?",
            (deal_id, limit),
        ).fetchall()


def delete_child(schema_name: str, record_id: int) -> None:
    sc = schemas.get_child(schema_name)
    with connect() as conn:
        conn.execute(f"DELETE FROM {sc.table} WHERE id=?", (record_id,))
        conn.commit()


def count_children(schema_name: str, deal_id: int | None = None) -> int:
    sc = schemas.get_child(schema_name)
    with connect() as conn:
        if deal_id is None:
            return conn.execute(f"SELECT COUNT(*) AS n FROM {sc.table}").fetchone()["n"]
        return conn.execute(
            f"SELECT COUNT(*) AS n FROM {sc.table} WHERE deal_id=?", (deal_id,)
        ).fetchone()["n"]


# --- bundle fetch (champion case = deal + all 5 children) --------------------

def fetch_bundle(deal_id: int) -> dict[str, Any]:
    deal = get_deal(deal_id)
    if not deal:
        return {}
    children = {}
    for sc in schemas.all_children():
        children[sc.name] = list_children(sc.name, deal_id=deal_id)
    return {"deal": deal, "children": children}


# --- FTS search --------------------------------------------------------------

def _build_match(query: str) -> str | None:
    raw = [t for t in query.replace('"', " ").split() if t]
    tokens = [t for t in raw if len(t) >= 3] or raw
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def search_deals(query: str, limit: int = 3, only_champions: bool = True) -> list[sqlite3.Row]:
    match = _build_match(query)
    if not match:
        return []
    with connect() as conn:
        try:
            sql = f"""
                SELECT d.*, bm25(deals_fts) AS score
                FROM deals_fts f
                JOIN deals d ON d.id = f.rowid
                WHERE deals_fts MATCH ?
                {"AND d.outcome='成約'" if only_champions else ""}
                ORDER BY score ASC
                LIMIT ?
            """
            return conn.execute(sql, (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []


def search_children(schema_name: str, query: str, limit: int = 5) -> list[sqlite3.Row]:
    sc = schemas.get_child(schema_name)
    match = _build_match(query)
    if not match:
        return []
    with connect() as conn:
        try:
            sql = f"""
                SELECT c.*, bm25({sc.table}_fts) AS score
                FROM {sc.table}_fts f
                JOIN {sc.table} c ON c.id = f.rowid
                WHERE {sc.table}_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
            """
            return conn.execute(sql, (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []


# --- export / import ---------------------------------------------------------

def export_db_bytes() -> bytes:
    if not DB_FILE.exists():
        return b""
    return DB_FILE.read_bytes()


def import_db_bytes(data: bytes) -> None:
    _ensure_dir()
    DB_FILE.write_bytes(data)
