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

## Architecture

```
Browser (Jinja2 SSR + vanilla JS)
  ├── Sources panel     ← upload / manage
  ├── Chat (SSE)        ← streaming Q&A
  └── Chat history      ← logs + export
        │
        ▼
FastAPI backend (app.py + search.py)
  ├── Nemotron (vLLM, OpenAI-compatible API)
  ├── SQLite (soylm.db) + FTS5
  └── search.py (URL fetch, YouTube transcripts, Playwright fallback)
```

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with vLLM serving Nemotron-Nano-9B

### Install

```bash
git clone https://github.com/soy-tuber/SoyLM.git
cd SoyLM
uv venv && uv pip install -r requirements.txt
playwright install chromium
```

### Run

```bash
# vLLM must be running (directly or via Gateway)
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_BASE` | `http://localhost:8000/v1` | vLLM endpoint (OpenAI-compatible) |
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

## Appendix: Prefix caching bug with Mamba2 hybrid models

vLLM's `--enable-prefix-caching` causes issues with Nemotron-Nano-9B (Mamba2+Attention hybrid architecture). Specifically:

- **[vllm#27264](https://github.com/vllm-project/vllm/issues/27264)**: Prefix caching + `--mamba_ssm_cache_dtype float32` produces NaN outputs due to block table wraparound. Fixed in [PR #27753](https://github.com/vllm-project/vllm/pull/27753) (vLLM v0.15.1+).
- **`enable_thinking` mismatch**: Nemotron's chat template appends different tokens depending on the `enable_thinking` flag (`<think></think>` vs `<think>\n`). If a prefix cache warmup uses a different `enable_thinking` value than the actual streaming request, the final token block differs and the cache misses entirely.

SoyLM v3 disables prefix cache warmup to avoid these issues.

## License

MIT
