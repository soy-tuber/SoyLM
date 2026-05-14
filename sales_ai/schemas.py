"""Schema definitions for the sales-instructor prototype.

Data model:
  - `deals` is the master ("案件マスタ"). Champion case = a deal with
    outcome == "成約" and at least some of the 5 components attached.
  - Five child tables (proposals / talks / reports / results / followups)
    each carry a `deal_id` foreign key pointing to `deals`.
  - Full-text search uses the FTS5 trigram tokenizer so Japanese queries
    do substring matching against indexed columns.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChildSchema:
    name: str
    display: str
    table: str
    columns: list[tuple[str, str]]   # NOTE: deal_id is implicitly added
    fts_columns: list[str]
    rag_columns: list[str]
    prompt_hint: str


CHILD_SCHEMAS: dict[str, ChildSchema] = {
    "proposals": ChildSchema(
        name="proposals",
        display="A. 提案資料",
        table="proposals",
        columns=[
            ("problem_statement", "TEXT"),
            ("proposed_solution", "TEXT"),
            ("differentiator", "TEXT"),
            ("pricing_approach", "TEXT"),
            ("key_phrases", "TEXT"),
            ("full_text", "TEXT"),
        ],
        fts_columns=["problem_statement", "proposed_solution", "differentiator", "key_phrases", "full_text"],
        rag_columns=["problem_statement", "proposed_solution", "differentiator", "pricing_approach", "key_phrases"],
        prompt_hint=(
            "提案資料・営業提案書からの抽出。顧客の課題、提案内容、差別化点、"
            "価格提示方法、印象的なフレーズを構造化する。"
        ),
    ),
    "talks": ChildSchema(
        name="talks",
        display="B. 営業トーク",
        table="talks",
        columns=[
            ("talk_type", "TEXT"),
            ("situation", "TEXT"),
            ("script", "TEXT"),
            ("techniques_used", "TEXT"),
            ("customer_reaction", "TEXT"),
        ],
        fts_columns=["situation", "script", "techniques_used", "customer_reaction"],
        rag_columns=["talk_type", "situation", "script", "techniques_used", "customer_reaction"],
        prompt_hint=(
            "営業トーク・商談録の文字起こしからの抽出。"
            "トーク種別(初回/深掘り/クロージング/反論対応)、状況、トーク全文、"
            "使用テクニック(SPIN, BANT等)、顧客反応を構造化する。"
        ),
    ),
    "reports": ChildSchema(
        name="reports",
        display="C. レポート",
        table="reports",
        columns=[
            ("report_type", "TEXT"),
            ("period", "TEXT"),
            ("activities_summary", "TEXT"),
            ("pipeline_status", "TEXT"),
            ("challenges", "TEXT"),
            ("next_actions", "TEXT"),
            ("full_text", "TEXT"),
        ],
        fts_columns=["activities_summary", "pipeline_status", "challenges", "next_actions", "full_text"],
        rag_columns=["report_type", "period", "activities_summary", "pipeline_status", "challenges", "next_actions"],
        prompt_hint=(
            "週報・月報・案件報告などの営業レポートからの抽出。"
            "種別、対象期間、活動サマリ、パイプライン状況、課題、次のアクションを構造化する。"
        ),
    ),
    "results": ChildSchema(
        name="results",
        display="D. 成果",
        table="results",
        columns=[
            ("win_reason", "TEXT"),
            ("loss_reason", "TEXT"),
            ("competitor", "TEXT"),
            ("key_decision_factor", "TEXT"),
            ("closed_date", "TEXT"),
        ],
        fts_columns=["win_reason", "loss_reason", "key_decision_factor"],
        rag_columns=["win_reason", "loss_reason", "competitor", "key_decision_factor", "closed_date"],
        prompt_hint=(
            "成約・失注事例の成果データからの抽出。成約/失注要因、競合、"
            "決定要因、クローズ日を構造化する。"
        ),
    ),
    "followups": ChildSchema(
        name="followups",
        display="E. フォローメール/電話",
        table="followups",
        columns=[
            ("followup_type", "TEXT"),
            ("timing", "TEXT"),
            ("context", "TEXT"),
            ("content", "TEXT"),
            ("customer_response", "TEXT"),
            ("days_after_last_contact", "INTEGER"),
        ],
        fts_columns=["context", "content", "customer_response"],
        rag_columns=["followup_type", "timing", "context", "content", "customer_response"],
        prompt_hint=(
            "フォローアップのメール・電話・訪問記録からの抽出。"
            "種別、タイミング、商談状況、フォロー内容、顧客の反応、前回接触からの日数を構造化する。"
        ),
    ),
}


# --- deals master ------------------------------------------------------------

DEAL_COLUMNS: list[tuple[str, str]] = [
    ("deal_name", "TEXT"),
    ("customer_name", "TEXT"),
    ("customer_industry", "TEXT"),
    ("customer_size", "TEXT"),
    ("decision_maker_role", "TEXT"),
    ("product_sold", "TEXT"),
    ("problem_statement", "TEXT"),
    ("win_reason", "TEXT"),
    ("key_decision_factor", "TEXT"),
    ("deal_size_yen", "INTEGER"),
    ("sales_cycle_days", "INTEGER"),
    ("closed_date", "TEXT"),
    ("outcome", "TEXT"),         # 成約 / 継続中 / 失注 — champion = 成約
    ("summary", "TEXT"),         # 1〜3 段落の概要。FTS の主要ターゲット
]

DEAL_FTS_COLUMNS = [
    "deal_name", "customer_industry", "customer_size",
    "product_sold", "problem_statement", "win_reason",
    "key_decision_factor", "summary",
]


# --- helpers -----------------------------------------------------------------

def all_children() -> list[ChildSchema]:
    return list(CHILD_SCHEMAS.values())


def get_child(name: str) -> ChildSchema:
    return CHILD_SCHEMAS[name]
