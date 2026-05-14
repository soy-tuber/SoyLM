"""Seed 10 illustrative champion deals for the prototype.

Idempotent: only inserts if the deals table is empty. Each deal comes
with at least one record in each of the five child tables so the RAG
side has full bundles to retrieve from on day one.
"""
from __future__ import annotations

from . import db_manager


SEED_DEALS: list[dict] = [
    {
        "deal_name": "製造業 IoT 予知保全 — 大手自動車部品メーカー",
        "customer_name": "Anonymized Auto Parts Co.",
        "customer_industry": "製造業 (自動車部品)",
        "customer_size": "大企業 (従業員 5000+)",
        "decision_maker_role": "工場長 / 生産技術本部長",
        "product_sold": "IoT 予知保全プラットフォーム + AI 分析サービス",
        "problem_statement": "主要ラインの計画外停止が月 4 件発生し、年間損失 2 億円超。現場依存の保全で属人化が深刻。",
        "win_reason": "現場 1 ライン PoC で MTBF 30% 改善を実証し、CFO に ROI 8 ヶ月を提示できた点。",
        "key_decision_factor": "ROI 1 年以内 + 既存 PLC とのプロトコル互換性",
        "deal_size_yen": 48000000,
        "sales_cycle_days": 180,
        "closed_date": "2025-09-30",
        "outcome": "成約",
        "summary": "計画外停止の損失を定量化し、PoC で実証してから全社展開を提案。CFO 向けには ROI を、工場長向けには現場負荷削減を強調する二段構えのストーリーが奏功。",
    },
    {
        "deal_name": "SaaS 営業 DX — 中堅 SI 企業",
        "customer_name": "Mid-tier SI Co.",
        "customer_industry": "情報サービス (SI)",
        "customer_size": "中堅 (従業員 500〜1000)",
        "decision_maker_role": "営業本部長",
        "product_sold": "営業活動分析 SaaS",
        "problem_statement": "営業案件の半分が見える化されておらず、失注理由がレポート不在で改善できない。",
        "win_reason": "競合 SFA から 1 週間でデータ移行可能な点と、既存運用フローを変えないシンプル UI。",
        "key_decision_factor": "導入の容易さ (運用負担の少なさ)",
        "deal_size_yen": 8400000,
        "sales_cycle_days": 75,
        "closed_date": "2025-08-15",
        "outcome": "成約",
        "summary": "営業現場の運用負担を下げることを最優先に位置付け、既存 SFA との切替コストの低さで競合に勝利。",
    },
    {
        "deal_name": "金融 AI 与信モデル — 地方銀行",
        "customer_name": "Regional Bank A",
        "customer_industry": "金融 (地銀)",
        "customer_size": "大企業 (従業員 2000+)",
        "decision_maker_role": "リテール本部長 / リスク統括部長",
        "product_sold": "中小企業向け AI 与信スコアリングエンジン",
        "problem_statement": "中小向け融資の審査コストが高く、若手担当の判断ばらつきが大きい。",
        "win_reason": "既存与信モデルとの並走 PoC で AUC +0.07 を実証し、リスク統括部の懸念を払拭。",
        "key_decision_factor": "監査対応可能な説明性 (SHAP) と既存ルールとの併用設計",
        "deal_size_yen": 36000000,
        "sales_cycle_days": 240,
        "closed_date": "2025-07-20",
        "outcome": "成約",
        "summary": "規制業界では精度より説明性。SHAP で根拠を可視化し、リスク部門を味方に付けた。",
    },
    {
        "deal_name": "小売 RFID 在庫 — アパレル大手",
        "customer_name": "Apparel Chain B",
        "customer_industry": "小売 (アパレル)",
        "customer_size": "大企業 (店舗 400+)",
        "decision_maker_role": "サプライチェーン本部長",
        "product_sold": "RFID タグ在庫管理ソリューション",
        "problem_statement": "店頭欠品率 8% / 在庫差異 3%。在庫精度が EC との連携を阻害。",
        "win_reason": "5 店舗 POC で在庫差異 0.4% を達成し、EC 連携での売上機会損失削減を試算可能にした点。",
        "key_decision_factor": "棚卸し工数 80% 削減の現場メリット",
        "deal_size_yen": 120000000,
        "sales_cycle_days": 300,
        "closed_date": "2025-06-10",
        "outcome": "成約",
        "summary": "本部だけでなく店長会議に出席し、現場メリットを直接訴求。本部と現場の両方を巻き込む二正面攻撃。",
    },
    {
        "deal_name": "医療 電子カルテ AI 要約 — 中規模病院",
        "customer_name": "Hospital C",
        "customer_industry": "医療",
        "customer_size": "中規模 (病床 400)",
        "decision_maker_role": "病院長 / 医療情報部長",
        "product_sold": "電子カルテ AI サマリ生成プラグイン",
        "problem_statement": "医師のカルテ記載時間が 1 日 90 分。働き方改革 (時間外 960h 上限) 対応で逼迫。",
        "win_reason": "既存ベンダー API の制約下で 30 分短縮を実証。診療科長会議で支持を獲得。",
        "key_decision_factor": "医療情報安全管理ガイドライン準拠 + 既存ベンダー協調",
        "deal_size_yen": 18000000,
        "sales_cycle_days": 210,
        "closed_date": "2025-09-05",
        "outcome": "成約",
        "summary": "規制と既存ベンダー関係が鍵の業界では、対立より協調姿勢を示した提案が刺さる。",
    },
    {
        "deal_name": "建設 BIM 連携 — ゼネコン",
        "customer_name": "General Contractor D",
        "customer_industry": "建設",
        "customer_size": "大企業 (従業員 8000)",
        "decision_maker_role": "技術研究所長 / DX 推進部長",
        "product_sold": "BIM データ連携 + 進捗管理 SaaS",
        "problem_statement": "現場の進捗が紙とエクセル管理で、本社が週次でしか把握できない。",
        "win_reason": "現場所長との 3 ヶ月伴走 PoC で、定着率 80% を実証。",
        "key_decision_factor": "現場 IT リテラシーに合わせた段階導入計画",
        "deal_size_yen": 60000000,
        "sales_cycle_days": 270,
        "closed_date": "2025-05-25",
        "outcome": "成約",
        "summary": "現場 IT リテラシーを甘く見ない。段階導入と所長の巻き込みを明示することで信頼を獲得。",
    },
    {
        "deal_name": "物流 配車最適化 — 中堅運送",
        "customer_name": "Logistics Co. E",
        "customer_industry": "運輸 (一般貨物)",
        "customer_size": "中堅 (車両 800 台)",
        "decision_maker_role": "運行管理部長 / 経営企画室長",
        "product_sold": "AI 配車最適化エンジン",
        "problem_statement": "2024 年問題でドライバー不足。配車計画の属人化が深刻。",
        "win_reason": "PoC で実車率 +12pt、空車回送 -20% を実証。経営企画室が ROI を社内合意。",
        "key_decision_factor": "ベテラン配車係の暗黙知を AI が再現できるか (説明性)",
        "deal_size_yen": 24000000,
        "sales_cycle_days": 150,
        "closed_date": "2025-08-30",
        "outcome": "成約",
        "summary": "2024 年問題というマクロトレンドを背景に、ベテラン依存リスクを定量化して経営層を動かした。",
    },
    {
        "deal_name": "教育 学習分析 — 私立大学",
        "customer_name": "Private University F",
        "customer_industry": "教育 (高等教育)",
        "customer_size": "中規模 (学生 12000 人)",
        "decision_maker_role": "学長補佐 (IR 担当) / 情報センター長",
        "product_sold": "学習行動分析 / 中退予測ダッシュボード",
        "problem_statement": "中退率 8% が経営課題。早期介入の仕組みがない。",
        "win_reason": "他大学導入実績 (3 校) と中退予測 AUC 0.82 のベンチマーク提示。",
        "key_decision_factor": "他大学導入実績による安心感",
        "deal_size_yen": 9600000,
        "sales_cycle_days": 180,
        "closed_date": "2025-04-15",
        "outcome": "成約",
        "summary": "保守的な業界では数値より「他校でも導入されているか」の安心材料が決め手になる。",
    },
    {
        "deal_name": "不動産 賃料査定 AI — 地場仲介チェーン",
        "customer_name": "Real Estate Chain G",
        "customer_industry": "不動産 (賃貸仲介)",
        "customer_size": "中堅 (店舗 60)",
        "decision_maker_role": "営業企画部長 / 社長",
        "product_sold": "賃料査定 / 募集条件最適化 AI",
        "problem_statement": "募集賃料の設定が担当者勘で、空室期間が業界平均より 1.5 ヶ月長い。",
        "win_reason": "1 店舗で 3 ヶ月試験運用し、空室期間 -45 日を実証。社長が全店展開を即決。",
        "key_decision_factor": "オーナー説明資料として使える PDF レポート出力機能",
        "deal_size_yen": 7200000,
        "sales_cycle_days": 90,
        "closed_date": "2025-08-01",
        "outcome": "成約",
        "summary": "中小企業オーナー社長案件は、社長が PoC 数値に納得すれば一気に決まる。",
    },
    {
        "deal_name": "化学 工程パラメータ最適化 — 大手化学",
        "customer_name": "Chemical Co. H",
        "customer_industry": "製造業 (化学)",
        "customer_size": "大企業 (従業員 10000+)",
        "decision_maker_role": "プラント長 / R&D 本部長",
        "product_sold": "工程パラメータ最適化 ML サービス",
        "problem_statement": "歩留まり改善が頭打ち。熟練オペレータの暗黙知を引き継げない。",
        "win_reason": "歴史データ 3 年分の解析で 1.8% の改善余地を発見し、PoC で 1.2% を実現。",
        "key_decision_factor": "歩留まり 1% = 年 6 億円という具体額提示",
        "deal_size_yen": 84000000,
        "sales_cycle_days": 360,
        "closed_date": "2025-03-20",
        "outcome": "成約",
        "summary": "大手製造業は意思決定が遅いが、具体的な金額換算 (1% = X 億円) を最初に握ると稟議が早い。",
    },
]


def _children_for(deal_idx: int, summary: str, deal: dict) -> dict[str, dict]:
    """Generate stand-in records for A–E components from the deal summary."""
    return {
        "proposals": {
            "problem_statement": deal["problem_statement"],
            "proposed_solution": f"{deal['product_sold']} を導入し、{deal['problem_statement']} を解決。",
            "differentiator": deal["win_reason"],
            "pricing_approach": "年間サブスクリプション + PoC 期間のディスカウント",
            "key_phrases": deal["win_reason"],
            "full_text": summary,
        },
        "talks": {
            "talk_type": "クロージング",
            "situation": f"{deal['customer_industry']} の {deal['decision_maker_role']} 向け最終提案",
            "script": (
                f"「御社の課題である『{deal['problem_statement']}』について、"
                f"私たちは {deal['win_reason']} を実証できる唯一のパートナーです。」"
            ),
            "techniques_used": "SPIN (Implication, Need-payoff), 数値提示",
            "customer_reaction": "ROI 試算に納得感を示し、稟議に進めると即答。",
        },
        "reports": {
            "report_type": "案件報告",
            "period": deal["closed_date"],
            "activities_summary": f"{deal['sales_cycle_days']} 日のサイクルで {deal['deal_size_yen']:,} 円受注。",
            "pipeline_status": "受注済み",
            "challenges": deal["problem_statement"],
            "next_actions": "導入キックオフ → 月次レビュー定着化",
            "full_text": summary,
        },
        "results": {
            "win_reason": deal["win_reason"],
            "loss_reason": "",
            "competitor": "(主要競合 1〜2 社)",
            "key_decision_factor": deal["key_decision_factor"],
            "closed_date": deal["closed_date"],
        },
        "followups": {
            "followup_type": "メール",
            "timing": "訪問翌日",
            "context": f"{deal['decision_maker_role']} との最終商談直後",
            "content": (
                f"本日はお時間いただきありがとうございました。"
                f"ご懸念いただいた点に関して、{deal['win_reason']} を改めて整理しました。"
                f"添付の ROI 試算では、{deal['key_decision_factor']} の観点で再計算しております。"
            ),
            "customer_response": "翌々日に「社内で前向きに検討」との返信。",
            "days_after_last_contact": 1,
        },
    }


def is_empty() -> bool:
    return db_manager.count_deals() == 0


def seed(force: bool = False) -> int:
    """Insert the 10 sample deals if the master table is empty. Returns count."""
    if not force and not is_empty():
        return 0
    inserted = 0
    for i, d in enumerate(SEED_DEALS):
        deal_id = db_manager.create_deal(d)
        for schema_name, rec in _children_for(i, d["summary"], d).items():
            db_manager.insert_child(schema_name, deal_id, rec, source_file="seed.py")
        inserted += 1
    return inserted
