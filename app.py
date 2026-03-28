"""
SoyLM - Local-first RAG tool
FastAPI + Jinja2 / SSE streaming / Nemotron (vLLM)
"""

import asyncio
import json
import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from search import fetch_url_text

# ─── Config ───────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SOURCES_DIR = DATA_DIR / "sources"
SOURCES_DIR.mkdir(exist_ok=True)

# Gateway on 8000 manages vLLM lifecycle (auto-start, idle stop)
NEMOTRON_BASE = os.getenv("NEMOTRON_BASE", "http://localhost:8000/v1")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")
STREAM_MAX_TOKENS = int(os.getenv("STREAM_MAX_TOKENS", "8192"))

# System prompt — accuracy-focused, grounding rules
DEFAULT_SYSTEM_PROMPT = """You are a research assistant grounded in the user's sources.

Rules:
- Cite sources using [1], [2] etc. matching the order they appear in 【ソースデータ】.
- When combining multiple sources, cite all: [1, 3].
- If the answer requires information NOT in the sources, explicitly state: "This is not in the provided sources" before answering from general knowledge.
- If the sources lack relevant information, say so honestly.
- If the query is ambiguous, ask the user to clarify before answering.
- Respond in the same language the user wrote their question in."""


app = FastAPI(title="SoyLM")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

templates = Jinja2Templates(directory="templates")


# ─── Database ─────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DATA_DIR / "soylm.db")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'text',
            raw_text TEXT,
            summary TEXT,
            key_points TEXT,
            flash_analysis TEXT,
            content_hash TEXT,
            token_estimate INTEGER DEFAULT 0,
            loaded INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
            id, filename, raw_text, summary, key_points, flash_analysis,
            content='sources',
            content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS sources_ai AFTER INSERT ON sources BEGIN
            INSERT INTO sources_fts(rowid, id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES (new.rowid, new.id, new.filename, new.raw_text, new.summary, new.key_points, new.flash_analysis);
        END;
        CREATE TRIGGER IF NOT EXISTS sources_ad AFTER DELETE ON sources BEGIN
            INSERT INTO sources_fts(sources_fts, rowid, id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES ('delete', old.rowid, old.id, old.filename, old.raw_text, old.summary, old.key_points, old.flash_analysis);
        END;
        CREATE TRIGGER IF NOT EXISTS sources_au AFTER UPDATE ON sources BEGIN
            INSERT INTO sources_fts(sources_fts, rowid, id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES ('delete', old.rowid, old.id, old.filename, old.raw_text, old.summary, old.key_points, old.flash_analysis);
            INSERT INTO sources_fts(rowid, id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES (new.rowid, new.id, new.filename, new.raw_text, new.summary, new.key_points, new.flash_analysis);
        END;
        CREATE TABLE IF NOT EXISTS chatlogs (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            chatlog_id TEXT NOT NULL REFERENCES chatlogs(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    db.close()


init_db()


# ─── Helpers ──────────────────────────────────────────────────────
def uid() -> str:
    return uuid.uuid4().hex[:12]


def estimate_tokens(text: str) -> int:
    return len(text) // 3  # rough estimate for mixed jp/en


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─── Nemotron (vLLM OpenAI compat) ───────────────────────────────
async def nemotron_generate(prompt: str, system: str = "",
                            max_tokens: int = 8192,
                            enable_thinking: bool = True,
                            messages_override: list[dict] | None = None) -> dict:
    """Non-streaming Nemotron call. Returns full message dict."""
    url = f"{NEMOTRON_BASE}/chat/completions"
    if messages_override:
        messages = messages_override
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    body = {
        "model": NEMOTRON_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]


async def nemotron_generate_text(prompt: str, system: str = "",
                                  max_tokens: int = 8192) -> str:
    """Non-streaming Nemotron call (text only). Thinking disabled for utility calls."""
    msg = await nemotron_generate(prompt, system=system, max_tokens=max_tokens,
                                   enable_thinking=False)
    return msg.get("content", "")


async def nemotron_stream(prompt: str, system: str = "",
                          max_tokens: int = 16384,
                          enable_thinking: bool = True,
                          messages_override: list[dict] | None = None,
                          temperature: float = 0.1) -> AsyncGenerator[dict, None]:
    """Streaming via vLLM OpenAI-compatible endpoint with thinking support.
    Yields dicts: {"type": "thinking"|"text", "content": str}"""
    url = f"{NEMOTRON_BASE}/chat/completions"
    if messages_override:
        messages = messages_override
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    body = {
        "model": NEMOTRON_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    thinking_started = False
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"]
                        reasoning = delta.get("reasoning_content", "")
                        content = delta.get("content", "")
                        if reasoning:
                            if not thinking_started:
                                thinking_started = True
                            yield {"type": "thinking", "content": reasoning}
                        if content:
                            yield {"type": "text", "content": content}
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# ─── PDF Extraction ───────────────────────────────────────────
def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pymupdf."""
    import pymupdf
    text_parts = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n\n".join(text_parts)


# ─── Flash Source Loader ──────────────────────────────────────────
async def flash_load_source(source_id: str, raw_text: str, filename: str):
    """Analyze and structure source data using Nemotron."""
    prompt = f"""以下の文書を分析してください。

【ファイル名】{filename}

【内容】
{raw_text[:100000]}

以下のJSON形式で出力してください:
{{
  "summary": "200字以内の要約",
  "key_points": ["重要ポイント1", "重要ポイント2", ...],
  "topics": ["トピック1", "トピック2", ...],
  "entities": ["固有名詞1", "固有名詞2", ...],
  "language": "ja/en/other"
}}
JSONのみ出力。"""

    try:
        raw = await nemotron_generate_text(prompt, max_tokens=2048)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        analysis = json.loads(raw.strip())
    except Exception:
        analysis = {
            "summary": raw_text[:200],
            "key_points": [],
            "topics": [],
            "entities": [],
            "language": "unknown",
        }

    analysis["full_text"] = raw_text[:100000]
    db = get_db()
    db.execute("""
        UPDATE sources SET
            summary = ?, key_points = ?, flash_analysis = ?,
            loaded = 1, token_estimate = ?
        WHERE id = ?
    """, (
        analysis.get("summary", ""),
        json.dumps(analysis.get("key_points", []), ensure_ascii=False),
        json.dumps(analysis, ensure_ascii=False),
        estimate_tokens(raw_text),
        source_id,
    ))
    db.commit()
    db.close()
    return analysis


# ─── RAG Search ───────────────────────────────────────────────────
async def _extract_search_keywords(query: str) -> str:
    """Translate query to English, extract nouns, return EN+JA keywords."""
    prompt = f"""Translate to English, then list only the nouns. Output original Japanese nouns and English nouns, comma-separated. Max 8 words. Nothing else.

Q: Chromebookのセットアップ方法
A: Chromebook, setup, セットアップ

Q: {query}
A:"""
    try:
        raw = await nemotron_generate_text(prompt, max_tokens=64)
        return raw.strip()
    except Exception:
        return query


def _keywords_to_fts5(keywords: str) -> str:
    """Convert comma-separated keywords to FTS5 MATCH query."""
    import re
    tokens = re.split(r'[,\s]+', keywords)
    terms = []
    for t in tokens:
        cleaned = re.sub(r'[^\w]', '', t)
        if cleaned and len(cleaned) > 1:
            terms.append(f'"{cleaned}"')
    if not terms:
        return '""'
    return ' OR '.join(terms)


async def search_sources(notebook_id: str, query: str, limit: int = 10) -> list[dict]:
    """FTS5 search: translate to EN, extract nouns, search with BM25."""
    keywords = await _extract_search_keywords(query)
    db = get_db()
    fts_query = _keywords_to_fts5(keywords)
    rows = db.execute("""
        SELECT s.id, s.filename, s.summary, s.key_points, s.flash_analysis,
               s.raw_text, s.token_estimate,
               rank
        FROM sources_fts fts
        JOIN sources s ON s.id = fts.id
        WHERE sources_fts MATCH ? AND s.notebook_id = ? AND s.loaded = 1
        ORDER BY rank
        LIMIT ?
    """, (fts_query, notebook_id, limit)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_all_loaded_sources(notebook_id: str) -> list[dict]:
    """Get all loaded sources for full-context mode."""
    db = get_db()
    rows = db.execute("""
        SELECT id, filename, summary, key_points, flash_analysis, raw_text, token_estimate
        FROM sources WHERE notebook_id = ? AND loaded = 1
        ORDER BY created_at
    """, (notebook_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]


# ─── Chat context builder ────────────────────────────────────────
def _format_source(filename: str, flash_analysis: str, summary: str = "") -> str:
    """Format a source's flash_analysis into readable text for LLM."""
    try:
        data = json.loads(flash_analysis) if flash_analysis else {}
    except json.JSONDecodeError:
        return f"[{filename}]\n{flash_analysis or summary}"
    parts = [f"[{filename}]"]
    if data.get("summary"):
        parts.append(f"Summary: {data['summary']}")
    if data.get("key_points"):
        kp = data["key_points"]
        if isinstance(kp, list):
            parts.append("Key points:\n" + "\n".join(f"- {p}" for p in kp))
        else:
            parts.append(f"Key points: {kp}")
    if data.get("full_text"):
        parts.append(f"\nFull text:\n{data['full_text']}")
    return "\n".join(parts)


def _build_context_from_ids(source_ids: list[str]) -> str:
    """Build RAG context from explicitly selected source IDs."""
    if not source_ids:
        return ""
    db = get_db()
    placeholders = ",".join("?" for _ in source_ids)
    rows = db.execute(f"""
        SELECT filename, flash_analysis, summary
        FROM sources WHERE id IN ({placeholders}) AND loaded = 1
    """, source_ids).fetchall()
    db.close()
    parts = []
    for r in rows:
        parts.append(_format_source(r["filename"], r["flash_analysis"] or "", r["summary"] or ""))
    return "\n\n---\n\n".join(parts)


def _build_full_context(notebook_id: str) -> str:
    """Build full source context string for a notebook (all loaded sources)."""
    sources = get_all_loaded_sources(notebook_id)
    parts = []
    for s in sources:
        parts.append(_format_source(s['filename'], s.get('flash_analysis', ''), s.get('summary', '')))
    return "\n\n---\n\n".join(parts)


# ─── Routes: Pages ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    db = get_db()
    notebooks = db.execute("SELECT * FROM notebooks ORDER BY updated_at DESC").fetchall()
    db.close()
    return templates.TemplateResponse("index.html", {
        "request": request, "notebooks": notebooks
    })


@app.get("/notebook/{notebook_id}", response_class=HTMLResponse)
async def notebook_page(request: Request, notebook_id: str):
    db = get_db()
    notebook = db.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
    if not notebook:
        return HTMLResponse("Not found", status_code=404)
    sources = db.execute(
        "SELECT * FROM sources WHERE notebook_id = ? ORDER BY created_at", (notebook_id,)
    ).fetchall()
    chatlogs = db.execute(
        "SELECT * FROM chatlogs WHERE notebook_id = ? ORDER BY updated_at DESC", (notebook_id,)
    ).fetchall()
    db.close()
    return templates.TemplateResponse("notebook.html", {
        "request": request,
        "notebook": notebook,
        "sources": sources,
        "chatlogs": chatlogs,
    })


# ─── Routes: Notebooks CRUD ──────────────────────────────────────
@app.post("/api/notebooks")
async def create_notebook(name: str = Form("New Notebook")):
    nid = uid()
    db = get_db()
    db.execute("INSERT INTO notebooks (id, name) VALUES (?, ?)", (nid, name))
    db.commit()
    db.close()
    return JSONResponse({"id": nid, "name": name})


@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    db = get_db()
    db.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    db.commit()
    db.close()
    return JSONResponse({"ok": True})


# ─── Routes: Sources ─────────────────────────────────────────────
@app.post("/api/sources/upload")
async def upload_source(
    notebook_id: str = Form(...),
    files: list[UploadFile] = File(None),
    urls: str = Form(""),
    paste_text: str = Form(""),
):
    """Upload files, URLs, or pasted text as sources."""
    db = get_db()
    added = []

    if files:
        for f in files:
            if not f.filename:
                continue
            file_bytes = await f.read()
            if f.filename.lower().endswith(".pdf"):
                raw = _extract_pdf_text(file_bytes)
            else:
                raw = file_bytes.decode("utf-8", errors="replace")
            chash = content_hash(raw)
            exists = db.execute(
                "SELECT id FROM sources WHERE notebook_id = ? AND content_hash = ?",
                (notebook_id, chash)
            ).fetchone()
            if exists:
                continue
            sid = uid()
            db.execute("""
                INSERT INTO sources (id, notebook_id, filename, source_type, raw_text, content_hash, token_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sid, notebook_id, f.filename, "file", raw, chash, estimate_tokens(raw)))
            added.append({"id": sid, "filename": f.filename, "type": "file"})

    if urls.strip():
        for url in urls.strip().split("\n"):
            url = url.strip()
            if not url:
                continue
            try:
                raw = await fetch_url_text(url)
                chash = content_hash(raw)
                exists = db.execute(
                    "SELECT id FROM sources WHERE notebook_id = ? AND content_hash = ?",
                    (notebook_id, chash)
                ).fetchone()
                if exists:
                    continue
                sid = uid()
                stype = "youtube" if ("youtube.com" in url or "youtu.be" in url) else "url"
                db.execute("""
                    INSERT INTO sources (id, notebook_id, filename, source_type, raw_text, content_hash, token_estimate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sid, notebook_id, url[:100], stype, raw, chash, estimate_tokens(raw)))
                added.append({"id": sid, "filename": url[:100], "type": stype})
            except Exception as e:
                added.append({"id": None, "filename": url[:100], "type": "error", "error": str(e)})

    if paste_text.strip():
        raw = paste_text.strip()
        chash = content_hash(raw)
        exists = db.execute(
            "SELECT id FROM sources WHERE notebook_id = ? AND content_hash = ?",
            (notebook_id, chash)
        ).fetchone()
        if not exists:
            sid = uid()
            db.execute("""
                INSERT INTO sources (id, notebook_id, filename, source_type, raw_text, content_hash, token_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sid, notebook_id, f"paste_{sid}", "paste", raw, chash, estimate_tokens(raw)))
            added.append({"id": sid, "filename": f"paste_{sid}", "type": "paste"})

    db.commit()
    db.close()
    return JSONResponse({"added": added})


@app.post("/api/sources/load/{notebook_id}")
async def load_all_sources(notebook_id: str):
    """Process sources using Nemotron. Unloaded sources get full analysis.
    Already-loaded sources without full_text get backfilled."""
    db = get_db()
    unloaded = db.execute(
        "SELECT id, filename, raw_text FROM sources WHERE notebook_id = ? AND loaded = 0",
        (notebook_id,)
    ).fetchall()
    loaded = db.execute(
        "SELECT id, filename, raw_text, flash_analysis FROM sources WHERE notebook_id = ? AND loaded = 1",
        (notebook_id,)
    ).fetchall()
    db.close()

    backfilled = 0
    for src in loaded:
        fa = src["flash_analysis"] or ""
        if '"full_text"' not in fa and src["raw_text"]:
            try:
                analysis = json.loads(fa) if fa else {}
            except json.JSONDecodeError:
                analysis = {}
            analysis["full_text"] = src["raw_text"][:100000]
            db = get_db()
            db.execute(
                "UPDATE sources SET flash_analysis = ? WHERE id = ?",
                (json.dumps(analysis, ensure_ascii=False), src["id"])
            )
            db.commit()
            db.close()
            backfilled += 1

    tasks = [flash_load_source(src["id"], src["raw_text"], src["filename"]) for src in unloaded]
    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

    loaded_count = sum(1 for r in results if not isinstance(r, Exception))
    errors = [str(r) for r in results if isinstance(r, Exception)]

    return JSONResponse({
        "loaded": loaded_count,
        "backfilled": backfilled,
        "errors": errors,
        "total": len(unloaded) + len(loaded),
    })


@app.get("/api/sources/{source_id}")
async def get_source(source_id: str):
    """Get source content for preview."""
    db = get_db()
    row = db.execute(
        "SELECT id, filename, source_type, raw_text, summary, key_points, flash_analysis, loaded, token_estimate FROM sources WHERE id = ?",
        (source_id,)
    ).fetchone()
    db.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(dict(row))


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str):
    db = get_db()
    db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    db.commit()
    db.close()
    return JSONResponse({"ok": True})


# ─── Source extraction (Extract step) ─────────────────────────────
@app.post("/api/sources/extract")
async def extract_sources(request: Request):
    """Extract matching sources for a query. Returns results for user review before execution."""
    body = await request.json()
    notebook_id = body.get("notebook_id", "")
    message = body.get("message", "")
    if not notebook_id or not message:
        return JSONResponse({"sources": [], "keywords": ""})

    keywords = await _extract_search_keywords(message)
    fts_results = await search_sources(notebook_id, message, limit=5)

    sources_out = []
    for r in fts_results:
        sources_out.append({
            "id": r["id"],
            "filename": r["filename"],
            "summary": r.get("summary", ""),
            "token_estimate": r.get("token_estimate", 0),
            "rank": r.get("rank", 0),
        })

    return JSONResponse({
        "sources": sources_out,
        "keywords": keywords,
    })


# ─── Routes: Chat SSE ────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """SSE streaming chat. Thinking is streamed in real-time, content is sent as a complete block."""
    body = await request.json()
    notebook_id = body.get("notebook_id", "")
    message = body.get("message", "")
    chatlog_id = body.get("chatlog_id", "")
    custom_system = body.get("system_prompt", "").strip()
    temperature = body.get("temperature", 0.1)
    temperature = max(0.0, min(2.0, float(temperature)))

    # Build RAG context
    source_ids = body.get("source_ids", [])
    if source_ids:
        rag_context = _build_context_from_ids(source_ids)
    elif notebook_id:
        rag_context = _build_full_context(notebook_id)
    else:
        rag_context = ""

    system = custom_system if custom_system else DEFAULT_SYSTEM_PROMPT

    # Build user prompt
    prompt_parts = []
    if rag_context:
        prompt_parts.append(f"【ソースデータ】\n{rag_context}")
    prompt_parts.append(f"\n【質問】\n{message}")
    full_prompt = "\n\n".join(prompt_parts)

    # Save user message
    if chatlog_id:
        db = get_db()
        db.execute(
            "INSERT INTO messages (id, chatlog_id, role, content) VALUES (?, ?, ?, ?)",
            (uid(), chatlog_id, "user", message)
        )
        db.commit()
        db.close()

    # Build messages for LLM
    llm_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": full_prompt},
    ]

    # Stream thinking in real-time, collect content, send as single block at end
    async def event_stream() -> AsyncGenerator[str, None]:
        full_response = []

        try:
            async for chunk in nemotron_stream(
                prompt="", system="",
                messages_override=llm_messages,
                temperature=temperature,
                max_tokens=STREAM_MAX_TOKENS,
            ):
                if chunk["type"] == "thinking":
                    yield f"data: {json.dumps({'type': 'thinking', 'content': chunk['content']}, ensure_ascii=False)}\n\n"
                else:
                    full_response.append(chunk["content"])

            # Send complete content as single block
            response_text = "".join(full_response)
            if response_text:
                yield f"data: {json.dumps({'type': 'content', 'content': response_text}, ensure_ascii=False)}\n\n"

            # Save assistant message
            if chatlog_id:
                db = get_db()
                db.execute(
                    "INSERT INTO messages (id, chatlog_id, role, content, model) VALUES (?, ?, ?, ?, ?)",
                    (uid(), chatlog_id, "assistant", response_text, "nemotron")
                )
                db.execute("UPDATE chatlogs SET updated_at = datetime('now') WHERE id = ?", (chatlog_id,))
                db.commit()
                db.close()

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Routes: Chatlog CRUD ────────────────────────────────────────
@app.post("/api/chatlogs")
async def create_chatlog(notebook_id: str = Form(...), title: str = Form("")):
    cid = uid()
    if not title:
        title = f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
    db = get_db()
    db.execute("INSERT INTO chatlogs (id, notebook_id, title) VALUES (?, ?, ?)",
               (cid, notebook_id, title))
    db.commit()
    db.close()
    return JSONResponse({"id": cid, "title": title})


@app.get("/api/chatlogs/{chatlog_id}/messages")
async def get_messages(chatlog_id: str):
    db = get_db()
    msgs = db.execute(
        "SELECT * FROM messages WHERE chatlog_id = ? ORDER BY created_at", (chatlog_id,)
    ).fetchall()
    db.close()
    return JSONResponse([dict(m) for m in msgs])


@app.get("/api/chatlogs/{chatlog_id}/download")
async def download_chatlog(chatlog_id: str):
    """Download chatlog as JSON."""
    db = get_db()
    chatlog = db.execute("SELECT * FROM chatlogs WHERE id = ?", (chatlog_id,)).fetchone()
    msgs = db.execute(
        "SELECT role, content, model, created_at FROM messages WHERE chatlog_id = ? ORDER BY created_at",
        (chatlog_id,)
    ).fetchall()
    db.close()
    if not chatlog:
        return JSONResponse({"error": "not found"}, status_code=404)

    export = {
        "title": chatlog["title"],
        "created_at": chatlog["created_at"],
        "messages": [dict(m) for m in msgs],
    }
    return JSONResponse(
        export,
        headers={"Content-Disposition": f'attachment; filename="chatlog_{chatlog_id}.json"'}
    )


@app.delete("/api/chatlogs/{chatlog_id}")
async def delete_chatlog(chatlog_id: str):
    db = get_db()
    db.execute("DELETE FROM chatlogs WHERE id = ?", (chatlog_id,))
    db.commit()
    db.close()
    return JSONResponse({"ok": True})


# ─── vLLM status ─────────────────────────────────────────────────
def _get_vllm_pid() -> int | None:
    """Find PID of the vLLM process listening on port 8100."""
    import subprocess as sp
    try:
        out = sp.check_output(["ss", "-tlnp"], text=True, timeout=3)
        for line in out.splitlines():
            if ":8100 " in line and "vllm" in line:
                # Extract pid from users:(("vllm",pid=XXXXX,...))
                import re
                m = re.search(r"pid=(\d+)", line)
                return int(m.group(1)) if m else None
    except Exception:
        pass
    return None


@app.get("/api/vllm/status")
async def vllm_status():
    """Check vLLM health and report PID."""
    pid = _get_vllm_pid()
    if pid is None:
        return JSONResponse({"ready": False, "pid": None})
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://localhost:8100/health")
            return JSONResponse({"ready": r.status_code == 200, "pid": pid})
    except Exception:
        return JSONResponse({"ready": False, "pid": pid})


# ─── Health ───────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
