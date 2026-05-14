"""営業インストラクター AI — Streamlit prototype.

3 main features:
  1. 訪問前ブリーフィング (pre-visit briefing)
  2. 提案書作成 (proposal draft)
  3. 訪問後フォローアップ (post-visit follow-up)

Backed by a `deals` master + 5 child tables (proposals / talks /
reports / results / followups) holding champion cases. The retrieval
side scopes to deals with outcome=成約 and bundles the full A-E set
into the LLM context.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sales_ai import db_manager, features, llm_router, pipeline, rag_engine, schemas, seed

st.set_page_config(
    page_title="営業インストラクター AI",
    page_icon="🧑‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- bootstrap ---------------------------------------------------------------

def _secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or ""
    except Exception:
        return ""


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault(
        "api_keys",
        {"claude": _secret("anthropic_api_key"), "gemini": _secret("google_api_key")},
    )
    # default models
    ss.setdefault("model_briefing_provider", "claude")
    ss.setdefault("model_briefing_model", "claude-sonnet-4-5")
    ss.setdefault("model_proposal_provider", "claude")
    ss.setdefault("model_proposal_model", "claude-opus-4-5")
    ss.setdefault("model_followup_provider", "claude")
    ss.setdefault("model_followup_model", "claude-sonnet-4-5")
    ss.setdefault("model_ingest_provider", "gemini")
    ss.setdefault("model_ingest_model", "gemini-2.5-flash")
    # auto-seed on first run
    if db_manager.count_deals() == 0 and not ss.get("seed_attempted"):
        try:
            n = seed.seed()
            if n:
                ss["seed_notice"] = f"サンプルチャンピオン事例 {n} 件を読み込みました。"
        except Exception as e:
            ss["seed_notice"] = f"シード失敗: {e}"
        ss["seed_attempted"] = True


_init_state()


# --- helpers -----------------------------------------------------------------

def _api_key(provider: str) -> str:
    return st.session_state["api_keys"].get(provider, "")


def _model_picker(prefix: str, label_provider: str = "プロバイダー", label_model: str = "モデル"):
    col1, col2 = st.columns(2)
    with col1:
        providers = llm_router.PROVIDERS
        cur = st.session_state.get(f"model_{prefix}_provider", providers[0])
        provider = st.selectbox(label_provider, providers, index=providers.index(cur), key=f"sel_{prefix}_provider")
    with col2:
        models = llm_router.list_models(provider)
        cur_m = st.session_state.get(f"model_{prefix}_model", models[0])
        if cur_m not in models:
            cur_m = models[0]
        model = st.selectbox(label_model, models, index=models.index(cur_m), key=f"sel_{prefix}_model")
    st.session_state[f"model_{prefix}_provider"] = provider
    st.session_state[f"model_{prefix}_model"] = model
    if not _api_key(provider):
        st.warning(f"{provider} の API キーが未設定です。⚙️ 設定ページで入力してください。")
    return provider, model


def _render_matches(matches: list[rag_engine.DealMatch], title: str = "参照したチャンピオン事例"):
    if not matches:
        st.info("類似する成約事例は見つかりませんでした。LLM は一般論で回答します。")
        return
    with st.expander(f"{title} ({len(matches)} 件)", expanded=True):
        for m in matches:
            st.markdown(rag_engine.render_deal_card(m.deal, brief=True))
            st.caption(f"BM25 score = {m.score:.2f}")
            st.divider()


# --- sidebar nav -------------------------------------------------------------

PAGES = [
    "🏠 ホーム",
    "🎯 訪問前ブリーフィング",
    "📝 提案書作成",
    "🔄 訪問後フォローアップ",
    "📚 案件マスタ",
    "📥 データ投入",
    "⚙️ 設定",
]

st.sidebar.title("🧑‍🏫 営業インストラクター AI")
st.sidebar.caption("チャンピオン事例から学ぶ営業 AI")
page = st.sidebar.radio("ナビゲーション", PAGES, label_visibility="collapsed")

# sidebar status
champ = db_manager.count_deals(only_champions=True)
total = db_manager.count_deals()
st.sidebar.metric("チャンピオン事例", f"{champ} / {total}", help="成約 / 全案件")

with st.sidebar.expander("コンポーネント件数", expanded=False):
    for sc in schemas.all_children():
        st.caption(f"{sc.display}: {db_manager.count_children(sc.name)} 件")

if msg := st.session_state.pop("seed_notice", None):
    st.sidebar.success(msg)


# === 🏠 Home =================================================================

def render_home():
    st.title("🧑‍🏫 営業インストラクター AI")
    st.write(
        "過去の成約済み「チャンピオン事例」を参照しながら、"
        "**訪問前 / 提案書作成 / 訪問後フォローアップ** の 3 シーンで"
        "営業担当者に伴走する AI コーチです。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("🎯 訪問前ブリーフィング")
            st.write(
                "売りたい商品と訪問先情報を入力すると、類似する成約事例を抽出し、"
                "キーメッセージ・想定質問・反論対応・差別化ポイントを提案します。"
            )
            if st.button("ブリーフィングを開く", use_container_width=True, key="home_to_briefing"):
                st.session_state["_next_page"] = "🎯 訪問前ブリーフィング"
                st.rerun()
    with c2:
        with st.container(border=True):
            st.subheader("📝 提案書作成")
            st.write(
                "顧客情報・商品・課題・要件を入力すると、Opus が "
                "チャンピオン事例を引用しながら **Markdown 形式の提案書** を生成します。"
            )
            if st.button("提案書ドラフトを始める", use_container_width=True, key="home_to_proposal"):
                st.session_state["_next_page"] = "📝 提案書作成"
                st.rerun()
    with c3:
        with st.container(border=True):
            st.subheader("🔄 訪問後フォローアップ")
            st.write(
                "訪問レポートを貼り付けると、商談評価 (A〜D)・ToDo (期限付き)・"
                "推奨フォローメール文案・リスク分析を出力します。"
            )
            if st.button("フォローアップを始める", use_container_width=True, key="home_to_followup"):
                st.session_state["_next_page"] = "🔄 訪問後フォローアップ"
                st.rerun()

    st.divider()
    st.subheader("📚 案件マスタ (チャンピオン事例)")
    st.caption(
        f"現在 {champ} 件 / 全 {total} 件。"
        " クロージングまで到達した案件のみが「チャンピオン」として参照対象になります。"
        " 各案件は A. 提案資料 / B. 営業トーク / C. レポート / D. 成果 / E. フォローメール の 5 点セットで登録します。"
    )
    deals = db_manager.list_deals()
    if deals:
        df = pd.DataFrame([{
            "id": d["id"],
            "案件名": d["deal_name"],
            "業種": d["customer_industry"],
            "規模": d["customer_size"],
            "金額(円)": d["deal_size_yen"],
            "結果": d["outcome"],
        } for d in deals])
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("案件がまだありません。📚 案件マスタから新規登録できます。")


# handle inter-page navigation request from home buttons
if "_next_page" in st.session_state:
    page = st.session_state.pop("_next_page")


# === 🎯 Briefing =============================================================

def render_briefing():
    st.title("🎯 訪問前ブリーフィング")
    st.caption(
        "売りたい商品と顧客情報を入れてください。"
        " 類似する成約事例 (チャンピオン) を抽出し、訪問前に押さえるべき論点を整理します。"
    )

    with st.form("briefing_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            product = st.text_input("売りたい商品 / サービス *", placeholder="例: AI 配車最適化エンジン")
            customer_name = st.text_input("顧客名 (任意)", placeholder="例: 〇〇運送株式会社")
            industry = st.text_input("業種 *", placeholder="例: 運輸 (一般貨物)")
            size = st.text_input("規模 *", placeholder="例: 中堅 (車両 500 台)")
        with c2:
            decision_maker = st.text_input("想定決裁者 *", placeholder="例: 運行管理部長")
            competitor = st.text_input("競合 (分かれば)", placeholder="例: 某 SaaS A 社")
            current_situation = st.text_area(
                "現在の状況 *", height=100,
                placeholder="例: 既存配車は紙とエクセル。2024 年問題でドライバー不足が深刻化。",
            )
            suspected_pain = st.text_area(
                "想定される課題 *", height=100,
                placeholder="例: 配車計画の属人化、空車回送の多さ、夜間ドライバー残業の偏り",
            )
        st.markdown("**回答 LLM** (推奨: Sonnet)")
        provider, model = _model_picker("briefing")
        submit = st.form_submit_button("ブリーフィング生成", type="primary", use_container_width=True)

    if not submit:
        return
    if not all([product, industry, size, decision_maker, current_situation, suspected_pain]):
        st.error("必須項目 (*) を埋めてください")
        return
    if not _api_key(provider):
        st.error(f"{provider} の API キーが必要です")
        return

    inp = features.BriefingInput(
        product=product, customer_name=customer_name, industry=industry, size=size,
        decision_maker=decision_maker, current_situation=current_situation,
        suspected_pain=suspected_pain, competitor=competitor,
    )
    matches, stream = features.run_briefing(inp, provider, model, _api_key(provider))
    _render_matches(matches)
    st.subheader("AI ブリーフィング")
    try:
        st.write_stream(stream)
    except Exception as e:
        st.error(f"生成エラー: {e}")


# === 📝 Proposal =============================================================

def render_proposal():
    st.title("📝 提案書作成")
    st.caption(
        "Opus がチャンピオン事例を引用しながら Markdown 形式の提案書ドラフトを生成します。"
        " 出来上がった提案書はそのままコピーして使えます。"
    )

    with st.form("proposal_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input("顧客名 *", placeholder="例: 株式会社〇〇")
            industry = st.text_input("業種 *", placeholder="例: 製造業 (化学)")
            size = st.text_input("規模 *", placeholder="例: 大企業 (従業員 10000+)")
            product = st.text_input("提案商品 / サービス *", placeholder="例: 工程パラメータ最適化 ML")
        with c2:
            problem = st.text_area("顧客の課題 *", height=100, placeholder="例: 歩留まり改善が頭打ち、暗黙知の継承課題")
            key_requirements = st.text_area("重要要件 *", height=100, placeholder="例: 既存 MES と API 連携、説明性、PoC ありき")
            c2a, c2b = st.columns(2)
            with c2a:
                budget_hint = st.text_input("予算感", placeholder="例: 5000 万円規模")
                competitor = st.text_input("競合", placeholder="例: 海外ベンダ X 社")
            with c2b:
                timeline = st.text_input("想定タイムライン", placeholder="例: PoC 3ヶ月 → 本番 1年")
        st.markdown("**生成 LLM** (推奨: Opus)")
        provider, model = _model_picker("proposal")
        submit = st.form_submit_button("提案書ドラフトを生成", type="primary", use_container_width=True)

    if not submit:
        return
    if not all([customer_name, industry, size, product, problem, key_requirements]):
        st.error("必須項目 (*) を埋めてください")
        return
    if not _api_key(provider):
        st.error(f"{provider} の API キーが必要です")
        return

    inp = features.ProposalInput(
        customer_name=customer_name, industry=industry, size=size, product=product,
        problem=problem, key_requirements=key_requirements,
        budget_hint=budget_hint, competitor=competitor, timeline=timeline,
    )
    matches, stream = features.run_proposal(inp, provider, model, _api_key(provider))
    _render_matches(matches, title="引用元チャンピオン事例")
    st.subheader("提案書ドラフト (Markdown)")
    try:
        st.write_stream(stream)
    except Exception as e:
        st.error(f"生成エラー: {e}")


# === 🔄 Followup =============================================================

def render_followup():
    st.title("🔄 訪問後フォローアップ")
    st.caption("訪問レポートを貼り付けると、商談評価・ToDo・フォローメール文案を生成します。")

    with st.form("followup_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            visit_date = st.text_input("訪問日 *", placeholder="例: 2026-05-13")
            customer_name = st.text_input("顧客名 *", placeholder="例: 株式会社〇〇")
            customer_reaction = st.text_area(
                "顧客の反応 (主観)", height=100,
                placeholder="例: 価格には難色、ROI 算定には強い関心",
            )
        with c2:
            next_step_idea = st.text_area(
                "営業担当が考える次アクション (任意)", height=100,
                placeholder="例: 価格 2 パターンの再提案 + 同業他社事例の追加",
            )
        visit_report = st.text_area(
            "訪問レポート本文 *", height=240,
            placeholder=(
                "誰と / 何を話したか / 顧客の反応 / 競合状況 / 未確定論点 / 持ち帰り事項を、"
                "できるだけ事実ベースで詳しく記載してください。"
            ),
        )
        st.markdown("**回答 LLM** (推奨: Sonnet)")
        provider, model = _model_picker("followup")
        submit = st.form_submit_button("フォローアップを生成", type="primary", use_container_width=True)

    if not submit:
        return
    if not all([visit_date, customer_name, visit_report]):
        st.error("必須項目 (*) を埋めてください")
        return
    if not _api_key(provider):
        st.error(f"{provider} の API キーが必要です")
        return

    inp = features.FollowupInput(
        visit_date=visit_date, customer_name=customer_name,
        visit_report=visit_report, customer_reaction=customer_reaction,
        next_step_idea=next_step_idea,
    )
    matches, stream = features.run_followup(inp, provider, model, _api_key(provider))
    _render_matches(matches)
    st.subheader("フォローアップ分析")
    try:
        st.write_stream(stream)
    except Exception as e:
        st.error(f"生成エラー: {e}")


# === 📚 Deal master ==========================================================

def render_deal_master():
    st.title("📚 案件マスタ")
    st.caption("チャンピオン事例 = クロージングまで到達した案件 + 5 コンポーネント (A〜E) のセット。")

    tabs = st.tabs(["一覧", "新規登録", "詳細 / 編集", "DB エクスポート"])

    # --- list
    with tabs[0]:
        only_champ = st.checkbox("成約のみ表示", value=False)
        deals = db_manager.list_deals(only_champions=only_champ)
        if deals:
            df = pd.DataFrame([dict(d) for d in deals])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("案件がありません")

    # --- create
    with tabs[1]:
        with st.form("new_deal_form", border=True):
            c1, c2 = st.columns(2)
            with c1:
                deal_name = st.text_input("案件名 *")
                customer_name = st.text_input("顧客名")
                customer_industry = st.text_input("業種 *")
                customer_size = st.text_input("規模 *")
                decision_maker_role = st.text_input("決裁者役職")
                product_sold = st.text_input("商品 / サービス *")
            with c2:
                problem_statement = st.text_area("課題", height=80)
                win_reason = st.text_area("成約要因", height=80)
                key_decision_factor = st.text_input("決定要因")
                deal_size_yen = st.number_input("金額(円)", min_value=0, step=100000, value=0)
                sales_cycle_days = st.number_input("営業サイクル(日)", min_value=0, step=1, value=0)
            outcome = st.selectbox("結果 *", ["成約", "継続中", "失注"])
            closed_date = st.text_input("クローズ日", placeholder="YYYY-MM-DD")
            summary = st.text_area("概要 (FTS 主要対象)", height=120)
            if st.form_submit_button("登録", type="primary"):
                if not all([deal_name, customer_industry, customer_size, product_sold]):
                    st.error("必須項目 (*) を埋めてください")
                else:
                    new_id = db_manager.create_deal({
                        "deal_name": deal_name, "customer_name": customer_name,
                        "customer_industry": customer_industry, "customer_size": customer_size,
                        "decision_maker_role": decision_maker_role, "product_sold": product_sold,
                        "problem_statement": problem_statement, "win_reason": win_reason,
                        "key_decision_factor": key_decision_factor,
                        "deal_size_yen": int(deal_size_yen),
                        "sales_cycle_days": int(sales_cycle_days),
                        "closed_date": closed_date, "outcome": outcome, "summary": summary,
                    })
                    st.success(f"案件 #{new_id} を登録しました。「📥 データ投入」から A〜E のコンポーネントを追加してください。")

    # --- detail
    with tabs[2]:
        deals = db_manager.list_deals()
        if not deals:
            st.info("案件がありません")
        else:
            opt = {f"#{d['id']} {d['deal_name']}": d["id"] for d in deals}
            choice = st.selectbox("案件を選択", list(opt.keys()))
            deal_id = opt[choice]
            bundle = db_manager.fetch_bundle(deal_id)
            if bundle:
                st.markdown(rag_engine.render_deal_card(bundle["deal"]))
                st.divider()
                for sc in schemas.all_children():
                    rows = bundle["children"].get(sc.name, [])
                    with st.expander(f"{sc.display}  ({len(rows)} 件)", expanded=False):
                        if rows:
                            st.dataframe(pd.DataFrame([dict(r) for r in rows]),
                                         hide_index=True, use_container_width=True)
                            rid = st.number_input(
                                "削除する子レコード id", min_value=0, step=1,
                                value=0, key=f"del_{sc.name}_{deal_id}",
                            )
                            if st.button("削除", key=f"btn_del_{sc.name}_{deal_id}", disabled=rid <= 0):
                                db_manager.delete_child(sc.name, int(rid))
                                st.rerun()
                        else:
                            st.caption("未登録")
                st.divider()
                if st.button("この案件を削除", type="secondary"):
                    db_manager.delete_deal(deal_id)
                    st.success("削除しました")
                    st.rerun()

    # --- export
    with tabs[3]:
        data = db_manager.export_db_bytes()
        st.download_button(
            "sales_ai.db をダウンロード",
            data=data if data else b"",
            file_name="sales_ai.db",
            mime="application/octet-stream",
            disabled=not data,
            use_container_width=True,
        )
        st.caption("Streamlit Cloud のディスクは再起動で消えるため、ダウンロードで退避してください。")
        up = st.file_uploader("sales_ai.db をアップロード (上書き)", type=["db"])
        if up and st.button("上書き実行"):
            db_manager.import_db_bytes(up.read())
            st.success("上書きしました。ページを再読み込みしてください。")


# === 📥 Pipeline =============================================================

def render_pipeline():
    st.title("📥 データ投入")
    st.caption("既存の案件に A〜E のコンポーネントを LLM 構造化で追加します。")

    deals = db_manager.list_deals()
    if not deals:
        st.warning("先に「📚 案件マスタ」で案件を作成してください。")
        return
    opt = {f"#{d['id']} {d['deal_name']}": d["id"] for d in deals}
    choice = st.selectbox("投入先の案件", list(opt.keys()))
    deal_id = opt[choice]

    schema_opts = {sc.display: sc.name for sc in schemas.all_children()}
    chosen_display = st.selectbox("コンポーネント種別", list(schema_opts.keys()))
    schema_name = schema_opts[chosen_display]
    sc = schemas.get_child(schema_name)
    with st.expander("このスキーマのフィールド"):
        st.dataframe(pd.DataFrame(sc.columns, columns=["column", "sql_type"]), hide_index=True)

    st.markdown("**構造化 LLM** (推奨: Flash / Haiku)")
    provider, model = _model_picker("ingest")

    uploads = st.file_uploader(
        "ファイル (PDF / DOCX / PPTX / XLSX / CSV / TXT / MD) — 複数可",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "json", "vtt", "srt"],
    )
    pasted = st.text_area("または、テキストを直接貼り付け", height=160)

    if st.button("投入実行", type="primary", disabled=not _api_key(provider)):
        items: list[tuple[str, bytes]] = []
        if uploads:
            for f in uploads:
                items.append((f.name, f.read()))
        if pasted.strip():
            items.append(("pasted_text.txt", pasted.encode("utf-8")))
        if not items:
            st.error("ファイルまたはテキストを指定してください")
            return
        progress = st.progress(0.0)
        for i, (name, data) in enumerate(items, 1):
            with st.status(f"処理中: {name}", expanded=False) as status:
                try:
                    result = pipeline.ingest(
                        schema_name=schema_name, deal_id=deal_id,
                        filename=name, data=data,
                        provider=provider, model=model, api_key=_api_key(provider),
                    )
                    status.update(label=f"✅ {name} (id={result['id']})", state="complete")
                    st.json(result["record"])
                except Exception as e:
                    status.update(label=f"❌ {name}: {e}", state="error")
            progress.progress(i / len(items))


# === ⚙️ Settings =============================================================

def render_settings():
    st.title("⚙️ 設定")
    st.write(
        "API キーはこのセッションにのみ保持され、リポジトリには保存されません。"
        " Streamlit Cloud に常時設定する場合は Settings → Secrets に登録してください:"
    )
    st.code(
        "anthropic_api_key = \"sk-ant-...\"\ngoogle_api_key    = \"AIza...\"",
        language="toml",
    )

    keys = st.session_state["api_keys"]
    with st.form("api_keys_form"):
        claude = st.text_input("Anthropic API key", value=keys.get("claude", ""), type="password")
        gemini = st.text_input("Google AI Studio API key", value=keys.get("gemini", ""), type="password")
        if st.form_submit_button("保存"):
            st.session_state["api_keys"] = {"claude": claude.strip(), "gemini": gemini.strip()}
            st.success("保存しました")

    st.subheader("接続テスト")
    c1, c2 = st.columns(2)
    if c1.button("Claude にハロー", disabled=not _api_key("claude")):
        try:
            out = llm_router.chat(
                "claude", llm_router.CLAUDE_MODELS[-1], _api_key("claude"),
                system="Respond with exactly: OK",
                messages=[{"role": "user", "content": "ping"}], max_tokens=16,
            )
            st.success(f"Claude: {out}")
        except Exception as e:
            st.error(f"Claude エラー: {e}")
    if c2.button("Gemini にハロー", disabled=not _api_key("gemini")):
        try:
            out = llm_router.chat(
                "gemini", llm_router.GEMINI_MODELS[-1], _api_key("gemini"),
                system="Respond with exactly: OK",
                messages=[{"role": "user", "content": "ping"}], max_tokens=16,
            )
            st.success(f"Gemini: {out}")
        except Exception as e:
            st.error(f"Gemini エラー: {e}")

    st.divider()
    st.subheader("サンプル事例の再シード")
    if st.button("チャンピオン事例 10 件を再投入 (既存があれば追加)"):
        n = seed.seed(force=True)
        st.success(f"{n} 件投入しました")


# --- route -------------------------------------------------------------------

if page.startswith("🏠"):
    render_home()
elif page.startswith("🎯"):
    render_briefing()
elif page.startswith("📝"):
    render_proposal()
elif page.startswith("🔄"):
    render_followup()
elif page.startswith("📚"):
    render_deal_master()
elif page.startswith("📥"):
    render_pipeline()
elif page.startswith("⚙️"):
    render_settings()
