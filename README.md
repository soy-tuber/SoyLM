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

vLLM's `--enable-prefix-caching` is **not recommended** with Nemotron-Nano-9B (Mamba2+Attention hybrid architecture). Even on v0.15.1 where the NaN crash is fixed, prefix caching corrupts the SSM state and destroys thinking quality.

### Symptoms

- **NaN outputs** ([vllm#27264](https://github.com/vllm-project/vllm/issues/27264)): Prefix caching + `--mamba_ssm_cache_dtype float32` produces NaN outputs. Fixed numerically in [PR #27753](https://github.com/vllm-project/vllm/pull/27753) (v0.12.0+), but this only eliminates the NaN crash — it does not fix SSM state integrity.
- **Thinking corruption** (confirmed on v0.15.1): With prefix caching enabled, thinking tokens degrade into garbage (`is do f s aa`). The SSM state is initialized with incorrect values from the cache, and all subsequent decoding is destroyed. NaN gone ≠ correct behavior.
- **`enable_thinking` mismatch**: Nemotron's chat template appends different final tokens depending on `enable_thinking` (`<think></think>` vs `<think>\n`). Warmup and streaming with different values cause cache miss on the last block.

### Resolution

Disable prefix caching entirely and increase `max_tokens` to compensate. This is the only reliable configuration for Mamba2 hybrid models in production as of v0.15.1.

## Appendix 2: vLLM v0.15.1 notes for Nemotron hybrid models

Relevant fixes and features in vLLM v0.15.0–v0.15.1 for Mamba2+Attention hybrid architectures.

### Fixes in v0.15.1

| PR | Description |
|---|---|
| [#27753](https://github.com/vllm-project/vllm/pull/27753) | NaN fix: passes kernel block size to builders, preventing FlashAttention from reading NaN-filled partial blocks in fp32 Mamba SSM cache. Tested with Nemotron-Nano-9B-v2 + `--mamba_ssm_cache_dtype float32`. |
| [#33524](https://github.com/vllm-project/vllm/pull/33524) | Fixes prefix cache hit rate = 0% for hybrid attention models (1 Full Attn group + 1 SWA group). |
| [#33417](https://github.com/vllm-project/vllm/pull/33417) | SM120 (RTX Blackwell) support for NVFP4 MoE kernels. |
| [#33189](https://github.com/vllm-project/vllm/pull/33189) | Lazy load cv2 in `nemotron_parse.py` (import error fix). |

### New in v0.15.0: Mamba prefix caching align mode

[PR #30877](https://github.com/vllm-project/vllm/pull/30877) adds `--mamba-cache-mode align` for block-aligned prefix caching of Mamba/hybrid models. Caches Mamba states directly for ~2x speedup on repeated context (e.g., RAG system prompts). **Do not use** — prefix caching corrupts SSM state on Mamba2 hybrid models even after NaN fix. See Appendix 1.

```
--enable-prefix-caching --mamba-cache-mode align
```

### Future versions

| Version | Feature |
|---|---|
| v0.14.0+ (available) | `--default-chat-template-kwargs` — set `enable_thinking` server-wide instead of per-request |
| v0.18.0+ | `max_thinking_tokens` ([PR #20859](https://github.com/vllm-project/vllm/pull/20859)) — hard limit on thinking tokens via logit processor. **Do not upgrade**: v0.18.0 introduces regressions with Nemotron-Nano-9B-v2-Japanese — the model outputs English instead of Japanese, and the corruption propagates to other services sharing the same vLLM instance. Stay on v0.15.1. |

### Open issues

- [#26936](https://github.com/vllm-project/vllm/issues/26936): Hybrid Attention models broken after flashinfer 0.4 — partially addressed by #27753, still open.

## License

MIT
