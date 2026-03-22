# SoyLM

Local-first RAG tool powered by [Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) via vLLM.

## What it does

Upload documents, URLs, or YouTube videos as sources. SoyLM analyzes them with a local LLM, stores structured summaries in SQLite, and lets you chat with your sources using RAG (FTS5 + BM25) and optional web search (DuckDuckGo).

## Features

- **Source ingestion** — Files, web URLs (with Playwright JS rendering fallback), YouTube transcripts
- **Local LLM** — Nemotron-Nano-9B via vLLM (OpenAI-compatible API), thinking mode for inference
- **Tool calling** — Nemotron decides when to search via `<TOOLCALL>` (agent loop, max 3 rounds)
- **RAG search** — SQLite FTS5 full-text search with BM25 ranking
- **Web search** — DuckDuckGo integration, invoked autonomously by LLM when search is enabled
- **SSE streaming** — Real-time streamed responses
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
FastAPI backend (single file: app.py)
  ├── Nemotron (vLLM localhost:8000)
  ├── SQLite (soylm.db) + FTS5
  ├── Playwright (JS-rendered pages)
  ├── youtube-transcript-api
  └── DuckDuckGo search (ddgs)
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
# Start vLLM with Nemotron parser plugins (tool calling + reasoning)
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  --trust-remote-code \
  --tool-call-parser nemotron_json \
  --tool-parser-plugin nemotron_toolcall_parser_streaming.py \
  --reasoning-parser nemotron_nano_v2 \
  --reasoning-parser-plugin nemotron_nano_v2_reasoning_parser.py \
  --enable-auto-tool-choice

# Then start SoyLM
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

> **Parser plugins**: Nemotron v2 uses `<TOOLCALL>` as regular text tokens, not dedicated tokenizer tokens. vLLM's built-in parsers (qwen3_coder, nemotron_v3) don't work for this model. The plugin-based parsers from [NeMo](https://github.com/NVIDIA/NeMo) are the correct approach. See [nemoclaw-local-inference-guide](https://github.com/soy-tuber/nemoclaw-local-inference-guide) for details.

Open `http://localhost:8080`

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_BASE` | `http://localhost:8000/v1` | vLLM endpoint |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` | Model name |

## Usage

1. Create a notebook
2. Add sources (files, URLs, YouTube links)
3. Click **Load Sources**
4. Ask questions

Toggle **Full Context** to feed all source analyses into the prompt.
Toggle **Web Search** to enable tool calling — Nemotron autonomously decides when to search DuckDuckGo.

## File structure

```
SoyLM/
├── app.py              # All backend logic
├── templates/
│   ├── index.html      # Home (notebook list)
│   └── notebook.html   # Main 3-column UI
├── data/               # Auto-generated, gitignored
│   └── soylm.db        # SQLite database
└── requirements.txt
```

## License

MIT
