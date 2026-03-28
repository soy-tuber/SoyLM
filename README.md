# SoyLM

Local-first RAG tool powered by [Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) via vLLM.

## What it does

Upload documents, URLs, or YouTube videos as sources. SoyLM analyzes them with a local LLM, stores structured summaries in SQLite, and lets you chat with your sources using RAG (FTS5 + BM25).

## Features

- **Source ingestion** — Files, web URLs (with Playwright JS rendering fallback), YouTube transcripts, paste text
- **Local LLM** — Nemotron-Nano-9B via vLLM (OpenAI-compatible API), thinking mode for reasoning
- **RAG search** — LLM keyword extraction (JA→EN) + SQLite FTS5 full-text search with BM25 ranking
- **SSE streaming** — Thinking streamed in real-time, content sent as complete block
- **Chat history** — Persistent chat logs with JSON export
- **Deduplication** — SHA-256 hash prevents duplicate sources
- **Gateway integration** — vLLM lifecycle managed by Nemotron Gateway (auto-start, idle stop)

## Architecture

```
Browser (Jinja2 SSR + vanilla JS)
  ├── Sources panel     ← upload / manage
  ├── Chat (SSE)        ← streaming Q&A
  └── Chat history      ← logs + export
        │
        ▼
FastAPI backend (app.py + search.py + tools.py)
  ├── Nemotron Gateway (localhost:8000) → vLLM (localhost:8100)
  ├── SQLite (soylm.db) + FTS5
  ├── Playwright (JS-rendered pages)
  └── youtube-transcript-api
```

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with vLLM serving Nemotron (or any OpenAI-compatible endpoint)

### Install

```bash
git clone https://github.com/soy-tuber/SoyLM.git
cd SoyLM
uv venv && uv pip install -r requirements.txt
playwright install chromium
```

### Run

```bash
# Option 1: Use start.sh (starts vLLM + app)
./start.sh

# Option 2: If Nemotron Gateway is running on 8000
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_BASE` | `http://localhost:8000/v1` | LLM endpoint (Gateway) |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` | Model name |
| `STREAM_MAX_TOKENS` | `8192` | Max tokens per streaming response |

## Usage

1. Create a notebook
2. Add sources (files, URLs, YouTube links, or paste text)
3. Click **Load Sources** to analyze with LLM
4. Ask a question — matching sources are extracted automatically via FTS5
5. Click **Generate** to get a grounded answer with source citations

## File structure

```
SoyLM/
├── app.py              # FastAPI backend + RAG logic
├── search.py           # Web fetching (URL, Playwright, YouTube)
├── tools.py            # Tool definitions (preserved for future use)
├── start.sh            # Convenience launcher (vLLM + app)
├── templates/
│   ├── index.html      # Home (notebook list)
│   └── notebook.html   # Main 3-column UI
├── data/               # Auto-generated, gitignored
│   └── soylm.db        # SQLite database
└── requirements.txt
```

## License

MIT
