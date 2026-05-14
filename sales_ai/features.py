"""Three feature orchestrators: pre-visit briefing, proposal draft, post-visit follow-up.

Each function builds a query from a structured form input, retrieves
similar champion cases, composes a system prompt that fixes both the
persona and the expected output shape, and yields the LLM's streaming
response. The Streamlit layer is responsible for displaying the
matched cases alongside the streamed answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from . import llm_router, rag_engine


# ---------------------------------------------------------------------------
# 1. 訪問前ブリーフィング
# ---------------------------------------------------------------------------

BRIEFING_SYSTEM = """あなたは経験豊富な営業インストラクターです。
営業担当者が訪問前に行うブリーフィングをサポートします。
過去の成約済みチャンピオン事例を参考に、訪問先に最適化した戦略を提示します。

## 出力フォーマット (必ずこの見出し構成で Markdown 出力)
### 1. 類似チャンピオン事例の要約
- 提示された事例のうちどれが最も類似するか、理由とともに 2〜3 行で説明
### 2. キーメッセージ案 (3 つ)
- 訪問先に響く可能性が高い切り口を箇条書きで
### 3. 投げかけるべき質問 (5 つ)
- SPIN / BANT を意識したオープン質問
### 4. 想定される反論と切り返し
- 表形式 (| 想定反論 | 切り返し |) で 3 行以上
### 5. 差別化ポイント
- チャンピオン事例の win_reason / differentiator を引用しながら具体的に
### 6. 訪問前チェックリスト
- 当日までに準備すべき事項を 5〜7 個

## ルール
- チャンピオン事例を引用する際は `[案件 #ID]` の形式で明示する
- 「不明」と書かず、事例から推測した仮説に「(仮説)」のラベルを付ける
- 全体で 800〜1500 字程度に収める
"""


@dataclass
class BriefingInput:
    product: str
    customer_name: str
    industry: str
    size: str
    decision_maker: str
    current_situation: str
    suspected_pain: str
    competitor: str = ""


def briefing_query(b: BriefingInput) -> str:
    return " ".join(
        x for x in [b.product, b.industry, b.size, b.suspected_pain, b.competitor, b.current_situation]
        if x
    )


def briefing_user_message(b: BriefingInput) -> str:
    return f"""## 訪問対象
- 売りたい商品/サービス: {b.product}
- 顧客名: {b.customer_name or "(未確定)"}
- 業種: {b.industry}
- 規模: {b.size}
- 想定決裁者: {b.decision_maker}
- 競合: {b.competitor or "(不明)"}
- 現在の状況: {b.current_situation}
- 想定される課題: {b.suspected_pain}

上記に向けた訪問前ブリーフィングをお願いします。"""


def run_briefing(
    inp: BriefingInput, provider: str, model: str, api_key: str, top_k: int = 3
) -> tuple[list, Iterator[str]]:
    matches = rag_engine.retrieve_similar_deals(briefing_query(inp), top_k=top_k)
    context = rag_engine.build_champion_context(
        matches, include=("proposals", "talks", "results")
    )
    system = f"{BRIEFING_SYSTEM}\n\n## 参照チャンピオン事例\n{context}"
    stream = llm_router.stream_chat(
        provider, model, api_key,
        system=system,
        messages=[{"role": "user", "content": briefing_user_message(inp)}],
        max_tokens=2048,
    )
    return matches, stream


# ---------------------------------------------------------------------------
# 2. 提案書作成
# ---------------------------------------------------------------------------

PROPOSAL_SYSTEM = """あなたは経験豊富な営業コンサルタントです。
過去の成約済みチャンピオン事例を参考に、顧客向けの提案書ドラフトを Markdown で作成します。

## 出力フォーマット
完成度の高い提案書として、以下のセクションを **必ず** 含めること:

# 提案書: {顧客名} 様 — {商品名}

## エグゼクティブサマリ
- 3 行で本提案の要点

## 1. 顧客の現状と課題
- ヒアリング情報をもとに、顧客の状況・課題・影響を整理

## 2. 提案ソリューション
- 提案内容を機能・サービス単位で

## 3. 期待される効果 (ROI)
- 定量効果 (試算) と定性効果。チャンピオン事例の数値があれば引用

## 4. 他社事例
- チャンピオン事例から 1〜2 件、`[案件 #ID]` を引用しつつ要約

## 5. なぜ当社か (差別化)
- 競合との違いを 3 点

## 6. 価格・スケジュール
- 価格レンジ (案) と導入ステップ

## 7. 次のステップ
- 意思決定までのアクション

## ルール
- チャンピオン事例の数値・フレーズを積極的に引用し、その都度 `[案件 #ID]` を付す
- 不確実な数値には `(目安)` ラベルを付ける
- 装飾的な絵文字は使わない
- 全体で 1500〜3000 字を目安
"""


@dataclass
class ProposalInput:
    customer_name: str
    industry: str
    size: str
    product: str
    problem: str
    key_requirements: str
    budget_hint: str = ""
    competitor: str = ""
    timeline: str = ""


def proposal_query(p: ProposalInput) -> str:
    return " ".join(x for x in [p.product, p.industry, p.problem, p.size, p.competitor] if x)


def proposal_user_message(p: ProposalInput) -> str:
    return f"""以下の情報をもとに提案書ドラフトを作成してください。

- 顧客名: {p.customer_name}
- 業種: {p.industry}
- 規模: {p.size}
- 商品: {p.product}
- 課題: {p.problem}
- 重要要件: {p.key_requirements}
- 予算感: {p.budget_hint or "(未確定)"}
- 競合: {p.competitor or "(不明)"}
- 想定タイムライン: {p.timeline or "(未確定)"}
"""


def run_proposal(
    inp: ProposalInput, provider: str, model: str, api_key: str, top_k: int = 3
) -> tuple[list, Iterator[str]]:
    matches = rag_engine.retrieve_similar_deals(proposal_query(inp), top_k=top_k)
    context = rag_engine.build_champion_context(matches)  # 全コンポーネント
    system = f"{PROPOSAL_SYSTEM}\n\n## 参照チャンピオン事例 (引用元)\n{context}"
    stream = llm_router.stream_chat(
        provider, model, api_key,
        system=system,
        messages=[{"role": "user", "content": proposal_user_message(inp)}],
        max_tokens=4096,
    )
    return matches, stream


# ---------------------------------------------------------------------------
# 3. 訪問後フォローアップ
# ---------------------------------------------------------------------------

FOLLOWUP_SYSTEM = """あなたは営業マネージャーとして、訪問直後の振り返りと
次アクションの優先順位付けを行います。過去のチャンピオン事例のフォロー
パターンを踏まえてアドバイスします。

## 出力フォーマット (必ずこの見出し構成で Markdown 出力)
### 商談評価
- A / B / C / D の 4 段階評価とその理由 (3 行)

### 強み (Good)
- 今回の訪問で良かった点 3 つ

### 弱み・リスク
- 改善すべき点と、放置した場合のリスク 3 つ

### ToDo (期限付き)
以下の形式の表で必ず 5〜7 行出力:
| 優先度 | ToDo | 期限 | 想定所要時間 |
|---|---|---|---|
| 高 | … | 訪問翌日 | 30 分 |

優先度は 高 / 中 / 低、期限は具体的な日数表現 (例: 翌日、3 日以内、1 週間以内、2 週間以内) を使う。

### 推奨フォローメール文案
- 件名: 〜
- 本文 (3〜5 段落、敬体、合計 300〜500 字)
- チャンピオン事例のフォロー文があれば引用して `[案件 #ID]` を付す

### 次回打合せに向けた仮説
- 想定される顧客反応とその対応方針を 3 点

## ルール
- ToDo は具体的な行動 (例: 「価格 2 パターンの見積を作成」) で書く
- 想像で書かず、訪問レポートに書かれている事実をベースに分析する
"""


@dataclass
class FollowupInput:
    visit_date: str
    customer_name: str
    visit_report: str
    customer_reaction: str = ""
    next_step_idea: str = ""


def followup_query(f: FollowupInput) -> str:
    return " ".join(x for x in [f.customer_name, f.visit_report[:200], f.customer_reaction] if x)


def followup_user_message(f: FollowupInput) -> str:
    return f"""以下の訪問レポートをもとに、振り返りと次アクションを整理してください。

- 訪問日: {f.visit_date}
- 顧客名: {f.customer_name}
- 顧客の反応 (営業担当の主観): {f.customer_reaction or "(記入なし)"}
- 営業担当が考える次アクション: {f.next_step_idea or "(記入なし)"}

## 訪問レポート本文
{f.visit_report}
"""


def run_followup(
    inp: FollowupInput, provider: str, model: str, api_key: str, top_k: int = 3
) -> tuple[list, Iterator[str]]:
    matches = rag_engine.retrieve_similar_deals(followup_query(inp), top_k=top_k)
    context = rag_engine.build_champion_context(
        matches, include=("reports", "followups", "results")
    )
    system = f"{FOLLOWUP_SYSTEM}\n\n## 参照チャンピオン事例\n{context}"
    stream = llm_router.stream_chat(
        provider, model, api_key,
        system=system,
        messages=[{"role": "user", "content": followup_user_message(inp)}],
        max_tokens=2500,
    )
    return matches, stream
