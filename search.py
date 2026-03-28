"""
SoyLM - Search & URL fetch module
DDG search, URL/YouTube fetching, Playwright fallback
"""

import asyncio
import re
from urllib.parse import urljoin, urlparse

import httpx


# ─── DDG Search ───────────────────────────────────────────────────
async def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo search via ddgs library."""
    from ddgs import DDGS
    results = []
    try:
        loop = asyncio.get_running_loop()
        def _search():
            ddgs = DDGS()
            return ddgs.text(query, region="wt-wt", safesearch="off", max_results=max_results)
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


# ─── YouTube ──────────────────────────────────────────────────────
def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
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

    loop = asyncio.get_running_loop()
    def _get():
        api = YouTubeTranscriptApi()
        try:
            result = api.fetch(video_id, languages=['ja', 'en'])
            return [s.text for s in result.snippets]
        except Exception:
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


# ─── HTML / URL Fetch ─────────────────────────────────────────────
async def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
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
            await page.wait_for_timeout(2000)
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        print(f"Playwright fetch error: {e}")
        return ""


def _extract_same_domain_links(html: str, base_url: str, max_links: int = 15) -> list[str]:
    """Extract same-domain links from HTML, prioritizing content pages."""
    base_domain = urlparse(base_url).netloc
    raw_links = re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\']', html)
    seen = set()
    content_links = []
    nav_links = []

    skip_ext = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js',
                '.pdf', '.zip', '.mp4', '.mp3', '.woff', '.woff2', '.ico')
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
        path_parts = [p for p in parsed.path.split('/') if p]
        has_slug = any('-' in p and len(p) > 5 for p in path_parts)
        if has_slug or len(path_parts) >= 2:
            content_links.append(clean)
        else:
            nav_links.append(clean)

    return (content_links + nav_links)[:max_links]


async def _fetch_url_with_depth(url: str, max_depth: int = 2) -> str:
    """Fetch URL text with crawl depth (same-domain links only)."""
    visited = set()
    all_texts = []
    total_chars = 0
    max_total = 100000

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

        if len(text) < 500:
            pw_html = await _fetch_with_playwright(target_url)
            if pw_html:
                pw_text = await _html_to_text(pw_html)
                if len(pw_text) > len(text):
                    text = pw_text
                    html = pw_html

        if not text:
            return

        text = text[:30000]
        all_texts.append(f"[URL: {target_url}]\n{text}")
        total_chars += len(text)

        if depth < max_depth and total_chars < max_total:
            child_links = _extract_same_domain_links(html, target_url, max_links=5)
            tasks = [_crawl(link, depth + 1) for link in child_links]
            await asyncio.gather(*tasks)

    await _crawl(url, depth=1)
    return "\n\n---\n\n".join(all_texts)[:max_total]


async def fetch_url_text(url: str) -> str:
    """Fetch text content from URL. Handles YouTube with transcript extraction."""
    if "youtube.com" in url or "youtu.be" in url:
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

        video_id = _extract_video_id(url)
        if video_id:
            try:
                transcript = await _fetch_youtube_transcript(video_id)
                if transcript:
                    meta_text += f"\n\n--- Transcript ---\n{transcript[:100000]}"
            except Exception as e:
                meta_text += f"\n\n(Transcript unavailable: {e})"

        return meta_text
    else:
        return await _fetch_url_with_depth(url, max_depth=1)


async def fetch_web_content(url: str) -> dict:
    """Fetch web page text content for RAG context."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            text = await _html_to_text(r.text)
            return {"url": url, "content": text[:3000]}
    except Exception:
        return {"url": url, "content": ""}
