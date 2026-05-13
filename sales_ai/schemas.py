"""Predefined DB schemas for the sales-instructor prototype.

Each entry defines:
  - the SQLite tables (content + FTS5 mirror with triggers)
  - the JSON shape the LLM must produce when structuring a document
  - the columns displayed in the management UI
  - which columns are surfaced as context during RAG search
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DBSchema:
    name: str            # filename stem -> {name}.db
    display: str         # Japanese label for UI
    table: str           # primary table name
    columns: list[tuple[str, str]]   # (col, sql_type) excluding id/created_at/source_file
    fts_columns: list[str]            # which columns are indexed in FTS5
    rag_columns: list[str]            # which columns are quoted into the prompt
    prompt_hint: str                  # description shown to the LLM during structuring


SCHEMAS: dict[str, DBSchema] = {
    "proposals": DBSchema(
        name="proposals",
        display="提案資料",
        table="proposals",
        columns=[
            ("customer_industry", "TEXT"),
            ("customer_size", "TEXT"),
            ("problem_statement", "TEXT"),
            ("proposed_solution", "TEXT"),
            ("differentiator", "TEXT"),
            ("pricing_approach", "TEXT"),
            ("outcome", "TEXT"),
            ("key_phrases", "TEXT"),
            ("full_text", "TEXT"),
        ],
        fts_columns=[
            "problem_statement",
            "proposed_solution",
            "differentiator",
            "key_phrases",
            "full_text",
        ],
        rag_columns=[
            "customer_industry",
            "customer_size",
            "problem_statement",
            "proposed_solution",
            "differentiator",
            "pricing_approach",
            "outcome",
            "key_phrases",
        ],
        prompt_hint=(
            "提案資料・営業提案書からの抽出。顧客の課題、提案内容、差別化点、"
            "価格提示方法、クロージング結果、印象的なフレーズを構造化する。"
        ),
    ),
    "talks": DBSchema(
        name="talks",
        display="営業トーク",
        table="talks",
        columns=[
            ("talk_type", "TEXT"),
            ("customer_persona", "TEXT"),
            ("situation", "TEXT"),
            ("script", "TEXT"),
            ("techniques_used", "TEXT"),
            ("customer_reaction", "TEXT"),
            ("outcome", "TEXT"),
        ],
        fts_columns=["situation", "script", "techniques_used", "customer_reaction"],
        rag_columns=[
            "talk_type",
            "customer_persona",
            "situation",
            "script",
            "techniques_used",
            "customer_reaction",
            "outcome",
        ],
        prompt_hint=(
            "営業トーク・商談録の文字起こしからの抽出。"
            "トーク種別(初回/深掘り/クロージング/反論対応)、顧客ペルソナ、"
            "状況、トーク全文、使用テクニック(SPIN, BANT等)、顧客反応、結果を構造化する。"
        ),
    ),
    "reports": DBSchema(
        name="reports",
        display="レポート",
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
        fts_columns=[
            "activities_summary",
            "pipeline_status",
            "challenges",
            "next_actions",
            "full_text",
        ],
        rag_columns=[
            "report_type",
            "period",
            "activities_summary",
            "pipeline_status",
            "challenges",
            "next_actions",
        ],
        prompt_hint=(
            "週報・月報・案件報告などの営業レポートからの抽出。"
            "種別、対象期間、活動サマリ、パイプライン状況、課題、次のアクションを構造化する。"
        ),
    ),
    "results": DBSchema(
        name="results",
        display="成果データ",
        table="results",
        columns=[
            ("deal_name", "TEXT"),
            ("customer_name", "TEXT"),
            ("customer_industry", "TEXT"),
            ("deal_size_yen", "INTEGER"),
            ("sales_cycle_days", "INTEGER"),
            ("win_reason", "TEXT"),
            ("loss_reason", "TEXT"),
            ("competitor", "TEXT"),
            ("key_decision_factor", "TEXT"),
            ("outcome", "TEXT"),
            ("closed_date", "TEXT"),
        ],
        fts_columns=[
            "deal_name",
            "win_reason",
            "loss_reason",
            "key_decision_factor",
        ],
        rag_columns=[
            "deal_name",
            "customer_name",
            "customer_industry",
            "deal_size_yen",
            "win_reason",
            "loss_reason",
            "competitor",
            "key_decision_factor",
            "outcome",
        ],
        prompt_hint=(
            "成約・失注事例の成果データからの抽出。案件名、顧客、規模(円)、"
            "営業サイクル日数、成約/失注要因、競合、決定要因、結果、クローズ日を構造化する。"
            "成約=outcome は '成約'、失注=outcome は '失注' と日本語で記載。"
        ),
    ),
    "followups": DBSchema(
        name="followups",
        display="フォロー事例",
        table="followups",
        columns=[
            ("followup_type", "TEXT"),
            ("timing", "TEXT"),
            ("context", "TEXT"),
            ("content", "TEXT"),
            ("customer_response", "TEXT"),
            ("outcome", "TEXT"),
            ("days_after_last_contact", "INTEGER"),
        ],
        fts_columns=["context", "content", "customer_response"],
        rag_columns=[
            "followup_type",
            "timing",
            "context",
            "content",
            "customer_response",
            "outcome",
        ],
        prompt_hint=(
            "フォローアップのメール・電話・訪問記録からの抽出。"
            "種別、タイミング、商談状況、フォロー内容、顧客の反応、結果、前回接触からの日数を構造化する。"
        ),
    ),
}


def all_schemas() -> list[DBSchema]:
    return list(SCHEMAS.values())


def get(name: str) -> DBSchema:
    return SCHEMAS[name]
