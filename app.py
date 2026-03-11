"""
SoyLM - Local-first RAG tool
FastAPI + Jinja2 / SSE streaming / Gemini + Nemotron
"""

import asyncio
import json
import hashlib
import os
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

# ─── Config ───────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SOURCES_DIR = DATA_DIR / "sources"
SOURCES_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

NEMOTRON_BASE = os.getenv("NEMOTRON_BASE", "http://localhost:8000/v1")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")

DDG_BASE = "https://api.duckduckgo.com/"

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
            source_id, filename, raw_text, summary, key_points, flash_analysis,
            content='sources',
            content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS sources_ai AFTER INSERT ON sources BEGIN
            INSERT INTO sources_fts(rowid, source_id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES (new.rowid, new.id, new.filename, new.raw_text, new.summary, new.key_points, new.flash_analysis);
        END;
        CREATE TRIGGER IF NOT EXISTS sources_ad AFTER DELETE ON sources BEGIN
            INSERT INTO sources_fts(sources_fts, rowid, source_id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES ('delete', old.rowid, old.id, old.filename, old.raw_text, old.summary, old.key_points, old.flash_analysis);
        END;
        CREATE TRIGGER IF NOT EXISTS sources_au AFTER UPDATE ON sources BEGIN
            INSERT INTO sources_fts(sources_fts, rowid, source_id, filename, raw_text, summary, key_points, flash_analysis)
            VALUES ('delete', old.rowid, old.id, old.filename, old.raw_text, old.summary, old.key_points, old.flash_analysis);
            INSERT INTO sources_fts(rowid, source_id, filename, raw_text, summary, key_points, flash_analysis)
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
            search_results TEXT,
            fact_check TEXT,
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


# ─── Gemini API ───────────────────────────────────────────────────
async def gemini_generate(prompt: str, model: str = GEMINI_FLASH_MODEL,
                          system: str = "", max_tokens: int = 8192) -> str:
    """Non-streaming Gemini call (for Flash preprocessing)."""
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_stream(prompt: str, model: str = GEMINI_PRO_MODEL,
                        system: str = "", max_tokens: int = 16384) -> AsyncGenerator[str, None]:
    """Streaming Gemini call via SSE."""
    url = f"{GEMINI_BASE}/{model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                        yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# ─── Nemotron (vLLM OpenAI compat) ───────────────────────────────
async def nemotron_generate(prompt: str, system: str = "",
                            max_tokens: int = 8192) -> str:
    """Non-streaming Nemotron call (for loading, no thinking)."""
    url = f"{NEMOTRON_BASE}/chat/completions"
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def nemotron_stream(prompt: str, system: str = "",
                          max_tokens: int = 16384,
                          enable_thinking: bool = True) -> AsyncGenerator[str, None]:
    """Streaming via vLLM OpenAI-compatible endpoint with thinking support."""
    url = f"{NEMOTRON_BASE}/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": NEMOTRON_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    in_thinking = False
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            # Handle thinking tags
                            if "<think>" in delta:
                                in_thinking = True
                                delta = delta.replace("<think>", "")
                                if delta:
                                    yield f"💭 {delta}"
                                continue
                            if "</think>" in delta:
                                in_thinking = False
                                delta = delta.replace("</think>", "")
                                if delta:
                                    yield f"\n\n{delta}"
                                continue
                            if in_thinking:
                                yield delta
                            else:
                                yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# ─── DDG Search ───────────────────────────────────────────────────
async def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo search via ddgs library (proven in StockAnalyzer)."""
    from ddgs import DDGS
    results = []
    try:
        loop = asyncio.get_event_loop()
        def _search():
            ddgs = DDGS()
            return ddgs.text(query, region="jp-jp", safesearch="off", max_results=max_results)
        raw = await loop.run_in_executor(None, _search)
        for r in raw:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")[:300],
            })
    except Exception as e:
        print(f"DDG search error: {e}")
    return results[:max_results]


# ─── Fact Checker (Gemini Flash) ──────────────────────────────────
async def fact_check(claim: str, context: str = "") -> dict:
    """Use Gemini Flash to fact-check a claim against context + DDG."""
    search_results = await ddg_search(claim, max_results=3)
    search_context = "\n".join(
        f"- {r['title']}: {r['snippet']}" for r in search_results
    )
    prompt = f"""あなたはファクトチェッカーです。以下の主張を検証してください。

【主張】
{claim}

【提供されたコンテキスト】
{context[:3000] if context else 'なし'}

【Web検索結果】
{search_context if search_context else 'なし'}

以下のJSON形式で回答してください:
{{"verdict": "正確/不正確/要検証/情報不足", "confidence": 0.0-1.0, "explanation": "理由", "sources": ["参照元"]}}
JSONのみ出力。"""

    try:
        raw = await gemini_generate(prompt, model=GEMINI_FLASH_MODEL, max_tokens=1024)
        # strip markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        return {"verdict": "エラー", "confidence": 0, "explanation": str(e), "sources": []}


# ─── URL / YouTube fetch ─────────────────────────────────────────
def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


async def _fetch_youtube_transcript(video_id: str) -> str:
    """Fetch YouTube transcript via youtube-transcript-api v1.2+."""
    from youtube_transcript_api import YouTubeTranscriptApi

    loop = asyncio.get_event_loop()
    def _get():
        api = YouTubeTranscriptApi()
        try:
            result = api.fetch(video_id, languages=['ja', 'en'])
            return [s.text for s in result.snippets]
        except Exception:
            # Fallback: grab whatever language is available
            try:
                tlist = api.list(video_id)
                for t in tlist:
                    result = api.fetch(video_id, languages=[t.language_code])
                    return [s.text for s in result.snippets]
            except Exception:
                return []
    segments = await loop.run_in_executor(None, _get)
    if not segments:
        return ""
    return "\n".join(segments)


async def fetch_url_text(url: str) -> str:
    """Fetch text content from URL. Handles YouTube with transcript extraction."""
    if "youtube.com" in url or "youtu.be" in url:
        # Get video metadata via oEmbed
        meta_text = ""
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"}
            )
            if r.status_code == 200:
                data = r.json()
                meta_text = f"[YouTube] {data.get('title', 'Unknown')}\nAuthor: {data.get('author_name', '')}\nURL: {url}"
            else:
                meta_text = f"[YouTube] {url}"

        # Try to get transcript
        video_id = _extract_video_id(url)
        if video_id:
            try:
                transcript = await _fetch_youtube_transcript(video_id)
                if transcript:
                    meta_text += f"\n\n--- Transcript ---\n{transcript[:100000]}"
            except Exception as e:
                meta_text += f"\n\n(字幕取得失敗: {e})"

        return meta_text
    else:
        return await _fetch_url_with_depth(url, max_depth=1)


async def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
    import re
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def _fetch_with_playwright(url: str) -> str:
    """Fetch page content using headless Chromium (for JS-rendered sites)."""
    from playwright.async_api import async_playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # Wait a bit for dynamic content
            await page.wait_for_timeout(2000)
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        print(f"Playwright fetch error: {e}")
        return ""


def _extract_same_domain_links(html: str, base_url: str, max_links: int = 15) -> list[str]:
    """Extract same-domain links from HTML, prioritizing content pages."""
    import re
    from urllib.parse import urljoin, urlparse
    base_domain = urlparse(base_url).netloc
    base_path = urlparse(base_url).path.rstrip('/')
    raw_links = re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\']', html)
    seen = set()
    content_links = []  # likely article/content pages
    nav_links = []      # navigation/utility pages

    skip_ext = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js',
                '.pdf', '.zip', '.mp4', '.mp3', '.woff', '.woff2', '.ico')
    # Short paths are usually nav (e.g. /en/, /catalog), longer are content
    skip_paths = ('/feed', '/rss', '/sitemap', '/tag/', '/category/', '/author/')

    for href in raw_links:
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc != base_domain:
            continue
        if any(parsed.path.lower().endswith(e) for e in skip_ext):
            continue
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        if clean in seen or clean.rstrip('/') == base_url.rstrip('/'):
            continue
        if any(s in parsed.path.lower() for s in skip_paths):
            continue
        seen.add(clean)
        # Heuristic: paths with hyphens and length > 2 segments are likely articles
        path_parts = [p for p in parsed.path.split('/') if p]
        has_slug = any('-' in p and len(p) > 5 for p in path_parts)
        if has_slug or len(path_parts) >= 2:
            content_links.append(clean)
        else:
            nav_links.append(clean)

    # Prioritize content pages over navigation
    result = content_links + nav_links
    return result[:max_links]


async def _fetch_url_with_depth(url: str, max_depth: int = 2) -> str:
    """Fetch URL text with crawl depth (same-domain links only)."""
    visited = set()
    all_texts = []
    total_chars = 0
    max_total = 100000  # cap total at 100k chars

    async def _crawl(target_url: str, depth: int):
        nonlocal total_chars
        if target_url in visited or depth > max_depth or total_chars >= max_total:
            return
        visited.add(target_url)

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                r = await client.get(target_url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                html = r.text
        except Exception:
            return

        text = await _html_to_text(html)

        # Fallback to Playwright if static fetch got too little content (JS-rendered site)
        if len(text) < 500:
            pw_html = await _fetch_with_playwright(target_url)
            if pw_html:
                pw_text = await _html_to_text(pw_html)
                if len(pw_text) > len(text):
                    text = pw_text
                    html = pw_html  # use JS-rendered HTML for link extraction too

        if not text:
            return

        # Cap per page
        text = text[:30000]
        all_texts.append(f"[URL: {target_url}]\n{text}")
        total_chars += len(text)

        # Crawl deeper if allowed
        if depth < max_depth and total_chars < max_total:
            child_links = _extract_same_domain_links(html, target_url, max_links=5)
            tasks = [_crawl(link, depth + 1) for link in child_links]
            await asyncio.gather(*tasks)

    await _crawl(url, depth=1)
    return "\n\n---\n\n".join(all_texts)[:max_total]


# ─── Flash Source Loader ──────────────────────────────────────────
async def flash_load_source(source_id: str, raw_text: str, filename: str,
                            loader_model: str = "gemini-flash"):
    """Analyze and structure source data using specified model."""
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
        if loader_model == "nemotron":
            raw = await nemotron_generate(prompt, max_tokens=2048)
        else:
            raw = await gemini_generate(prompt, model=GEMINI_FLASH_MODEL, max_tokens=2048)
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
def search_sources(notebook_id: str, query: str, limit: int = 10) -> list[dict]:
    """FTS5 search across loaded sources."""
    db = get_db()
    rows = db.execute("""
        SELECT s.id, s.filename, s.summary, s.key_points, s.flash_analysis,
               s.raw_text, s.token_estimate,
               rank
        FROM sources_fts fts
        JOIN sources s ON s.id = fts.source_id
        WHERE sources_fts MATCH ? AND s.notebook_id = ? AND s.loaded = 1
        ORDER BY rank
        LIMIT ?
    """, (query, notebook_id, limit)).fetchall()
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
def build_rag_context(notebook_id: str, query: str, full_context: bool = False) -> str:
    """Build context string for LLM prompt."""
    if full_context:
        sources = get_all_loaded_sources(notebook_id)
        # Use flash_analysis (compressed) instead of raw_text
        parts = []
        for s in sources:
            analysis = s.get("flash_analysis", "")
            parts.append(f"[{s['filename']}]\n{analysis}")
        return "\n\n---\n\n".join(parts)
    else:
        results = search_sources(notebook_id, query, limit=5)
        if not results:
            # Fallback: return all summaries
            sources = get_all_loaded_sources(notebook_id)
            parts = [f"[{s['filename']}] {s.get('summary', '')}" for s in sources]
            return "\n".join(parts)
        parts = []
        for r in results:
            parts.append(f"[{r['filename']}]\n{r.get('flash_analysis', r.get('summary', ''))}")
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
async def create_notebook(name: str = Form("新規ノートブック")):
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

    # File uploads
    if files:
        for f in files:
            if not f.filename:
                continue
            raw = (await f.read()).decode("utf-8", errors="replace")
            chash = content_hash(raw)
            # Dedup
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

    # URLs
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

    # Pasted text
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
async def load_all_sources(notebook_id: str, request: Request):
    """Process all unloaded sources using specified model."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    loader_model = body.get("loader_model", "gemini-flash")

    db = get_db()
    unloaded = db.execute(
        "SELECT id, filename, raw_text FROM sources WHERE notebook_id = ? AND loaded = 0",
        (notebook_id,)
    ).fetchall()
    db.close()

    results = []
    tasks = []
    for src in unloaded:
        tasks.append(flash_load_source(src["id"], src["raw_text"], src["filename"], loader_model))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

    loaded_count = sum(1 for r in results if not isinstance(r, Exception))
    errors = [str(r) for r in results if isinstance(r, Exception)]

    return JSONResponse({
        "loaded": loaded_count,
        "errors": errors,
        "total": len(unloaded),
    })


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str):
    db = get_db()
    db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    db.commit()
    db.close()
    return JSONResponse({"ok": True})


# ─── Routes: Chat SSE ────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """SSE streaming chat endpoint."""
    body = await request.json()
    notebook_id = body.get("notebook_id", "")
    message = body.get("message", "")
    model = body.get("model", "gemini-flash")  # gemini-flash / gemini-pro / nemotron
    full_context = body.get("full_context", False)
    chatlog_id = body.get("chatlog_id", "")
    enable_search = body.get("enable_search", False)
    enable_factcheck = body.get("enable_factcheck", False)

    # Build RAG context
    rag_context = build_rag_context(notebook_id, message, full_context) if notebook_id else ""

    # Optional DDG search
    search_data = []
    if enable_search:
        search_data = await ddg_search(message, max_results=5)

    search_text = ""
    if search_data:
        search_text = "\n【Web検索結果】\n" + "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in search_data
        )

    # System prompt
    system = """You are an expert research assistant.
Answer accurately based on the provided source data.
If information is not in the sources, clearly state it is speculation.
Respond in the same language as the user's question."""

    # Build user prompt
    prompt_parts = []
    if rag_context:
        prompt_parts.append(f"【ソースデータ】\n{rag_context}")
    if search_text:
        prompt_parts.append(search_text)
    prompt_parts.append(f"\n【質問】\n{message}")
    full_prompt = "\n\n".join(prompt_parts)

    # Save user message
    if chatlog_id:
        db = get_db()
        db.execute(
            "INSERT INTO messages (id, chatlog_id, role, content, search_results) VALUES (?, ?, ?, ?, ?)",
            (uid(), chatlog_id, "user", message,
             json.dumps(search_data, ensure_ascii=False) if search_data else None)
        )
        db.commit()
        db.close()

    # Stream response
    async def event_stream() -> AsyncGenerator[str, None]:
        full_response = []
        try:
            if model == "gemini-pro":
                gen = gemini_stream(full_prompt, model=GEMINI_PRO_MODEL, system=system)
            elif model == "gemini-flash":
                gen = gemini_stream(full_prompt, model=GEMINI_FLASH_MODEL, system=system)
            else:
                gen = nemotron_stream(full_prompt, system=system)

            async for chunk in gen:
                full_response.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            # Fact check if enabled
            fact_result = None
            if enable_factcheck and full_response:
                response_text = "".join(full_response)
                fact_result = await fact_check(response_text[:1000], rag_context[:2000])
                yield f"data: {json.dumps({'type': 'factcheck', 'result': fact_result}, ensure_ascii=False)}\n\n"

            # Save assistant message
            if chatlog_id:
                response_text = "".join(full_response)
                db = get_db()
                db.execute(
                    "INSERT INTO messages (id, chatlog_id, role, content, model, fact_check) VALUES (?, ?, ?, ?, ?, ?)",
                    (uid(), chatlog_id, "assistant", response_text, model,
                     json.dumps(fact_result, ensure_ascii=False) if fact_result else None)
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
        "SELECT role, content, model, search_results, fact_check, created_at FROM messages WHERE chatlog_id = ? ORDER BY created_at",
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


# ─── DDG search API endpoint ─────────────────────────────────────
@app.get("/api/search")
async def search_api(q: str = ""):
    if not q:
        return JSONResponse([])
    results = await ddg_search(q)
    return JSONResponse(results)


# ─── Fact check API endpoint ─────────────────────────────────────
@app.post("/api/factcheck")
async def factcheck_api(request: Request):
    body = await request.json()
    result = await fact_check(body.get("claim", ""), body.get("context", ""))
    return JSONResponse(result)


# ─── Health ───────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
