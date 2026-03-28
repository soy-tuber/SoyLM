<img width="1675" height="899" alt="image" src="https://github.com/user-attachments/assets/f0c7ff89-cb5e-4a7b-93b1-385ae151ea7c" />

# SoyLM

Local-first RAG system powered by a single 9B-parameter LLM. No vector database, no embedding models, no cloud APIs — just SQLite FTS5, BM25 ranking, and [Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) served locally via vLLM.

## Overview

SoyLM is a self-contained Retrieval-Augmented Generation application that runs entirely on local hardware. Upload documents, URLs, or YouTube videos as sources. The LLM analyzes each source into structured summaries stored in SQLite, then enables grounded Q&A with source citations — all through a single 9B model handling every stage of the pipeline.

### What makes it different

- **No vector database, no embeddings.** Retrieval uses SQLite FTS5 full-text search with BM25 ranking. The LLM extracts bilingual keywords (JA↔EN) from the user's query, which are used as FTS5 MATCH terms. This eliminates the need for separate embedding models, vector stores, and the associated infrastructure.
- **Single model for the entire pipeline.** One Nemotron-Nano-9B instance handles source analysis, keyword extraction, and answer generation. No multi-model orchestration.
- **Minimal footprint.** ~1,900 lines total (Python + HTML/JS). No React, no Node.js build step, no external search infrastructure. Two Python files, two HTML templates, one SQLite database.
- **Thinking transparency.** Nemotron's chain-of-thought reasoning tokens are streamed to the user in real-time via SSE, making the model's thought process visible before the final answer arrives.

## Features

| Feature | Details |
|---|---|
| **Source ingestion** | Files (.txt, .md, .py, .pdf, etc.), web URLs, YouTube transcripts, paste text, DuckDuckGo web search for URL discovery |
| **Web fetching** | httpx with automatic Playwright (headless Chromium) fallback for JS-rendered pages; same-domain link crawling |
| **YouTube** | Automatic transcript extraction via `youtube-transcript-api`, with oEmbed metadata |
| **Source analysis** | LLM-generated structured JSON (summary, key points, topics, entities, language) with FTS5 trigger auto-indexing |
| **RAG search** | Bilingual LLM keyword extraction (JA↔EN) → SQLite FTS5 MATCH with BM25 ranking |
| **Streaming** | SSE with separated thinking (real-time) and content (complete block) channels |
| **Deduplication** | SHA-256 content hashing prevents duplicate sources within a notebook |
| **Chat history** | Persistent chat logs per notebook with JSON export |

## Architecture

```
Browser (Jinja2 SSR + vanilla JS)
  ├── Sources panel     ← upload / manage / DDG search
  ├── Chat (SSE)        ← streaming Q&A with thinking
  └── Chat history      ← logs + JSON export
        │
        ▼
FastAPI backend
  ├── app.py    (~810 LOC)  ← routes, RAG logic, LLM calls
  ├── search.py (~220 LOC)  ← URL fetch, Playwright, YouTube, DDG
  ├── Nemotron-Nano-9B (vLLM, OpenAI-compatible API)
  └── SQLite (soylm.db, WAL mode, FTS5 virtual table)
```

### RAG pipeline

```
User query
  │
  ├─ 1. Keyword extraction (LLM, thinking disabled)
  │     "Chromebookのセットアップ方法"
  │       → "Chromebook, setup, セットアップ"
  │
  ├─ 2. FTS5 search (SQLite, BM25 ranking)
  │     SELECT ... FROM sources_fts
  │     WHERE sources_fts MATCH '"Chromebook" OR "setup" OR "セットアップ"'
  │     ORDER BY rank
  │
  ├─ 3. Context assembly
  │     Top-N sources → full text + structured metadata
  │
  └─ 4. Generation (LLM, streaming, thinking enabled)
        System prompt + 【ソースデータ】[1]..[N] + 【質問】
          → Thinking tokens (streamed real-time)
          → Answer with citations [1], [2] (sent as complete block)
```

The keyword extraction step is what makes cross-lingual retrieval work without embeddings: a Japanese query is decomposed into both Japanese and English noun terms, and the combined set is used as FTS5 search terms. Sources in either language can match queries in either language.

### Source loading pipeline

```
Input (file / URL / YouTube / paste)
  │
  ├─ Deduplication (SHA-256 hash check)
  │
  ├─ Text extraction
  │   ├── Files: UTF-8 decode / PyMuPDF for PDFs
  │   ├── URLs: httpx → Playwright fallback (if < 500 chars)
  │   ├── YouTube: youtube-transcript-api → oEmbed metadata
  │   └── Paste: direct text
  │
  └─ LLM analysis → structured JSON
      { summary, key_points, topics, entities, language, full_text }
      → SQLite INSERT triggers automatic FTS5 indexing
```

### Streaming architecture

Thinking tokens and content tokens are separated at the SSE level:

- **Thinking** — streamed chunk-by-chunk in real-time as the model reasons (`reasoning_content` field from vLLM)
- **Content** — collected server-side and sent as a single complete block after thinking finishes

This design ensures the final answer is coherent and not interleaved with partial reasoning, while maintaining full transparency into the model's chain-of-thought process.

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with vLLM serving Nemotron-Nano-9B-v2-Japanese

### vLLM configuration

Critical flags for Nemotron-Nano-9B (Mamba2+Attention hybrid architecture):

```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  --mamba_ssm_cache_dtype float32 \
  --max-model-len 16384 \
  --dtype auto
```

| Flag | Requirement | Reason |
|---|---|---|
| `--mamba_ssm_cache_dtype float32` | **Mandatory** | Without this, the Mamba2 SSM cache uses reduced precision and produces degraded outputs |
| `--enable-prefix-caching` | **Do NOT enable** | Corrupts SSM state on Mamba2 hybrid models — see [Appendix 1](#appendix-1-prefix-caching-bug-with-mamba2-hybrid-models) |

**Recommended version:** vLLM **v0.15.1**. Do not upgrade to v0.18.0+ — see [Appendix 2](#appendix-2-vllm-v0151-notes-for-nemotron-hybrid-models).

### Install

```bash
git clone https://github.com/soy-tuber/SoyLM.git
cd SoyLM
uv venv && uv pip install -r requirements.txt
playwright install chromium
```

### Run

```bash
# vLLM must be running on an OpenAI-compatible endpoint
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_BASE` | `http://localhost:8000/v1` | vLLM endpoint (any OpenAI-compatible API) |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` | Model name |
| `STREAM_MAX_TOKENS` | `8192` | Max tokens per streaming response |

SoyLM connects to any OpenAI-compatible endpoint — it does not manage the vLLM process. Use systemd, a process manager, or a gateway service for vLLM lifecycle management.

## Usage

1. Create a notebook
2. Add sources — upload files, enter URLs, paste YouTube links, paste text, or search DuckDuckGo for URLs
3. Click **Load Sources** — the LLM analyzes each source and generates structured summaries
4. Ask a question — matching sources are extracted automatically via FTS5 + BM25
5. Click **Generate** — the model thinks (visible in real-time), then delivers a grounded answer with `[1]`, `[2]` source citations

## Design rationale

### Why FTS5 + BM25 instead of vector search

Most RAG systems use vector search (FAISS, Chroma, Qdrant, pgvector, etc.) with a separate embedding model. SoyLM deliberately avoids this:

1. **Infrastructure cost.** Vector databases require a separate embedding model (often another GPU or API call per document), a vector store process, and index management. FTS5 runs inside SQLite — zero additional infrastructure.
2. **Predictability.** BM25 ranks by exact term frequency. For a system grounded in specific source documents (not open-domain semantic search), exact matching with known keywords is more predictable than cosine similarity in embedding space.
3. **Cross-lingual retrieval via LLM.** Instead of multilingual embeddings, SoyLM uses the LLM itself to extract bilingual keywords from the query. This is a single lightweight LLM call (~64 tokens) that produces both Japanese and English search terms, enabling cross-lingual retrieval through the same FTS5 index.
4. **No chunking required.** Vector search typically requires splitting documents into fixed-size chunks and embedding each. SoyLM stores full documents with LLM-generated metadata, and FTS5 searches across the complete text. The LLM's context window (up to 16K tokens) handles the full source content.

The trade-off: FTS5 cannot match semantically similar terms that don't share surface forms. In practice, the LLM keyword extraction compensates for this by generating synonyms and translations.

### Why a single 9B model for everything

Nemotron-Nano-9B-v2-Japanese is a Mamba2+Attention hybrid that handles Japanese and English natively. Using one model for all pipeline stages eliminates:

- Model coordination and routing logic
- Multiple GPU memory allocations
- Latency from cross-model API calls

The model's built-in thinking mode (`enable_thinking` via chat template) provides chain-of-thought reasoning without requiring a larger model or separate reasoning step. With `--mamba_ssm_cache_dtype float32` and prefix caching disabled, output quality is production-grade.

### Inference parameters

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | `0.1` | Low temperature for factual grounding — reduces hallucination while allowing slight variation |
| `max_tokens` (streaming) | `8192` | Sufficient for detailed answers with citations |
| `max_tokens` (utility calls) | `64–2048` | Minimal allocation for keyword extraction and source analysis |
| `enable_thinking` | `true` (chat) / `false` (utility) | Thinking enabled only for user-facing generation; disabled for keyword extraction and analysis to save tokens |

## File structure

```
SoyLM/
├── app.py              # FastAPI backend, RAG logic, LLM interface
├── search.py           # URL fetch, Playwright fallback, YouTube, DDG
├── tools.py            # Tool definitions (reserved for future use)
├── start.sh            # Convenience launcher
├── prompt_nvidia.yaml  # NVIDIA RAG Blueprint prompt templates (reference)
├── templates/
│   ├── index.html      # Home page — notebook list
│   └── notebook.html   # Main UI — 3-column layout (sources, chat, history)
├── data/               # Auto-created, gitignored
│   └── soylm.db        # SQLite database (WAL mode, FTS5)
└── requirements.txt    # 9 dependencies
```

### Dependencies

| Package | Purpose |
|---|---|
| FastAPI + Uvicorn | Web framework + ASGI server |
| httpx | Async HTTP client for vLLM API and URL fetching |
| Jinja2 | Server-side HTML rendering |
| python-multipart | File upload handling |
| Playwright | Headless Chromium for JS-rendered pages |
| PyMuPDF | PDF text extraction |
| youtube-transcript-api | YouTube transcript fetching |
| ddgs | DuckDuckGo search |

## Appendix 1: Prefix caching bug with Mamba2 hybrid models

vLLM's `--enable-prefix-caching` is **not compatible** with Nemotron-Nano-9B (Mamba2+Attention hybrid architecture). Even on v0.15.1 where the NaN crash is numerically fixed, prefix caching corrupts the SSM state and destroys output quality.

### Symptoms

- **NaN outputs** ([vllm#27264](https://github.com/vllm-project/vllm/issues/27264)): Prefix caching + `--mamba_ssm_cache_dtype float32` produces NaN outputs. Fixed numerically in [PR #27753](https://github.com/vllm-project/vllm/pull/27753) (v0.12.0+), but this only eliminates the NaN crash — it does not fix SSM state integrity.
- **Thinking corruption** (confirmed on v0.15.1): With prefix caching enabled, thinking tokens degrade into incoherent fragments (e.g., `is do f s aa`). The SSM state is initialized with incorrect values from the prefix cache, and all subsequent decoding is destroyed. **NaN gone ≠ correct behavior.**
- **`enable_thinking` mismatch**: Nemotron's chat template appends different final tokens depending on `enable_thinking` (`<think></think>` vs `<think>\n`). Warmup and streaming requests with different `enable_thinking` values cause cache misses on the last block, compounding the corruption.

### Root cause

Mamba2 (selective state space model) maintains a recurrent hidden state that is fundamentally sequential — each token's state depends on the full preceding sequence. Block-aligned prefix caching assumes that restoring a cached state at a block boundary produces identical results to computing from scratch. This assumption holds for pure Transformer attention but fails for SSM layers, where the cached state may not accurately represent the true recurrence.

### Resolution

Disable prefix caching entirely. This is the only reliable configuration for Mamba2 hybrid models in production as of vLLM v0.15.1.

```bash
# Do NOT use these flags:
# --enable-prefix-caching
# --enable-prefix-caching --mamba-cache-mode align
```

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

[PR #30877](https://github.com/vllm-project/vllm/pull/30877) adds `--mamba-cache-mode align` for block-aligned prefix caching of Mamba/hybrid models. Caches Mamba states directly for ~2x speedup on repeated context (e.g., RAG system prompts). **Do not use** — prefix caching corrupts SSM state on Mamba2 hybrid models even after the NaN fix. See [Appendix 1](#appendix-1-prefix-caching-bug-with-mamba2-hybrid-models).

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
