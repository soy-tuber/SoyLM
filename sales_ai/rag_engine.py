"""Multi-DB RAG: query each selected DB via FTS5, build a single context block."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from . import db_manager, llm_router, schemas


SYSTEM_PROMPT = """あなたは営業インストラクター AI です。
社内の優秀な営業事例(チャンピオンデータ)を参照して、営業担当者の質問に
具体的・実践的なアドバイスを返してください。

回答ルール:
- 参照データに含まれる具体的な事例を必ず引用する (例: 「成果データ #3 では...」)。
- 「なぜそれが効果的だったか」の分析を添える。
- 失注事例があれば成約事例と対比して教訓を示す。
- 参照データに無いことを聞かれた場合は、その旨を明示してから一般論を述べる。
- 簡潔な見出しと箇条書きを使い、3〜6 段落程度にまとめる。
"""


@dataclass
class Hit:
    schema_name: str
    record_id: int
    score: float
    snippet: str


def _row_snippet(row, rag_columns: list[str]) -> str:
    parts = []
    for col in rag_columns:
        val = row[col] if col in row.keys() else None
        if val is None or val == "":
            continue
        text = str(val).strip()
        if len(text) > 400:
            text = text[:400] + "..."
        parts.append(f"  - {col}: {text}")
    return "\n".join(parts)


def retrieve(
    query: str, selected_schemas: Iterable[str], per_db_limit: int = 3
) -> list[Hit]:
    hits: list[Hit] = []
    for name in selected_schemas:
        sc = schemas.get(name)
        rows = db_manager.search_fts(name, query, limit=per_db_limit)
        for r in rows:
            hits.append(
                Hit(
                    schema_name=name,
                    record_id=r["id"],
                    score=float(r["score"]),
                    snippet=_row_snippet(r, sc.rag_columns),
                )
            )
    return hits


def build_context(hits: list[Hit]) -> str:
    if not hits:
        return "(参照データは見つかりませんでした)"
    by_db: dict[str, list[Hit]] = {}
    for h in hits:
        by_db.setdefault(h.schema_name, []).append(h)

    blocks = []
    for name, group in by_db.items():
        sc = schemas.get(name)
        block = [f"### {sc.display} ({len(group)} 件ヒット)"]
        for h in group:
            block.append(f"[{sc.display} #{h.record_id}] (score={h.score:.2f})")
            block.append(h.snippet)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def stream_answer(
    question: str,
    selected_schemas: Iterable[str],
    provider: str,
    model: str,
    api_key: str,
    chat_history: list[dict] | None = None,
) -> Iterator[str]:
    hits = retrieve(question, selected_schemas)
    context = build_context(hits)
    system = f"{SYSTEM_PROMPT}\n\n## 参照データ\n{context}"
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})
    yield from llm_router.stream_chat(
        provider=provider, model=model, api_key=api_key,
        system=system, messages=messages, max_tokens=2048,
    )


def preview_hits(query: str, selected_schemas: Iterable[str]) -> list[Hit]:
    return retrieve(query, selected_schemas)
