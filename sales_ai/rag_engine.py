"""Champion-case retrieval and context building.

Two-step retrieval:
  1. `retrieve_similar_deals(query)` — FTS5 over `deals_fts` (deal master)
     scoped to champion deals (outcome=成約).
  2. For each matched deal, fetch its full A-E bundle and render as
     compact Japanese context for the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from . import db_manager, schemas


@dataclass
class DealMatch:
    deal: Any                       # sqlite3.Row
    score: float
    bundle: dict[str, Any]          # {"deal": Row, "children": {schema: [Row...]}}


def retrieve_similar_deals(query: str, top_k: int = 3) -> list[DealMatch]:
    rows = db_manager.search_deals(query, limit=top_k, only_champions=True)
    matches: list[DealMatch] = []
    for r in rows:
        bundle = db_manager.fetch_bundle(r["id"])
        matches.append(DealMatch(deal=r, score=float(r["score"]), bundle=bundle))
    return matches


def _truncate(text: str, n: int = 350) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return s if len(s) <= n else s[:n] + "…"


def render_deal_card(deal_row, brief: bool = False) -> str:
    """Render a deal master row as a short markdown card."""
    parts = [f"### 案件 #{deal_row['id']}: {deal_row['deal_name'] or '(無題)'}"]
    pairs = [
        ("顧客", deal_row["customer_name"]),
        ("業種", deal_row["customer_industry"]),
        ("規模", deal_row["customer_size"]),
        ("決裁者", deal_row["decision_maker_role"]),
        ("商品", deal_row["product_sold"]),
        ("金額(円)", deal_row["deal_size_yen"]),
        ("営業サイクル(日)", deal_row["sales_cycle_days"]),
        ("結果", deal_row["outcome"]),
    ]
    parts.append("- " + " / ".join(f"{k}: {v}" for k, v in pairs if v not in (None, "", 0)))
    if deal_row["problem_statement"]:
        parts.append(f"- 課題: {_truncate(deal_row['problem_statement'])}")
    if deal_row["win_reason"]:
        parts.append(f"- 成約要因: {_truncate(deal_row['win_reason'])}")
    if not brief and deal_row["summary"]:
        parts.append(f"- 概要: {_truncate(deal_row['summary'], 600)}")
    return "\n".join(parts)


def render_bundle(bundle: dict[str, Any], include: Iterable[str] | None = None) -> str:
    """Render a deal + its children as compact context blocks."""
    if not bundle:
        return ""
    deal = bundle["deal"]
    children = bundle["children"]
    out = [render_deal_card(deal, brief=False)]
    want = set(include) if include else set(s.name for s in schemas.all_children())
    for sc in schemas.all_children():
        if sc.name not in want:
            continue
        rows = children.get(sc.name, [])
        if not rows:
            continue
        out.append(f"\n**{sc.display}** ({len(rows)} 件)")
        for r in rows[:3]:
            lines = []
            for col in sc.rag_columns:
                if col in r.keys() and r[col]:
                    lines.append(f"  - {col}: {_truncate(r[col])}")
            if lines:
                out.append("\n".join(lines))
    return "\n".join(out)


def build_champion_context(matches: list[DealMatch], include: Iterable[str] | None = None) -> str:
    if not matches:
        return "(類似する成約事例は見つかりませんでした。一般論で回答してください。)"
    blocks = []
    for i, m in enumerate(matches, 1):
        blocks.append(f"## マッチ #{i} (BM25 score={m.score:.2f})\n" + render_bundle(m.bundle, include))
    return "\n\n---\n\n".join(blocks)
