"""Sales-instructor AI — Streamlit prototype.

Run locally:
    streamlit run streamlit_app.py

Deployed at Streamlit Cloud Community; API keys come from `st.secrets`
or are pasted into the Settings page at runtime (kept only in
`st.session_state`, never written to disk).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sales_ai import db_manager, llm_router, pipeline, rag_engine, schemas

st.set_page_config(
    page_title="営業インストラクター AI (Prototype)",
    page_icon="🧑‍🏫",
    layout="wide",
)


# --- session-state bootstrap --------------------------------------------------

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
    ss.setdefault("chat_provider", "claude")
    ss.setdefault("chat_model", llm_router.list_models("claude")[1])
    ss.setdefault("ingest_provider", "gemini")
    ss.setdefault("ingest_model", llm_router.list_models("gemini")[1])
    ss.setdefault("selected_dbs", [s.name for s in schemas.all_schemas()])
    ss.setdefault("messages", [])  # list[{role, content}]


_init_state()


# --- sidebar nav --------------------------------------------------------------

st.sidebar.title("🧑‍🏫 営業インストラクター AI")
st.sidebar.caption("SoyLM prototype — Streamlit edition")

page = st.sidebar.radio(
    "ページ",
    ["💬 チャット", "📥 データ投入", "📚 データ管理", "⚙️ 設定"],
    label_visibility="collapsed",
)

with st.sidebar.expander("DB 状態", expanded=True):
    for sc in schemas.all_schemas():
        try:
            n = db_manager.count_records(sc.name)
        except Exception:
            n = 0
        st.write(f"- {sc.display}: **{n}** 件")


# --- helpers ------------------------------------------------------------------

def _api_key(provider: str) -> str:
    return st.session_state["api_keys"].get(provider, "")


def _provider_model_selector(prefix: str, default_provider: str) -> tuple[str, str]:
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "プロバイダー",
            llm_router.PROVIDERS,
            index=llm_router.PROVIDERS.index(st.session_state.get(f"{prefix}_provider", default_provider)),
            key=f"{prefix}_provider_select",
        )
    with col2:
        models = llm_router.list_models(provider)
        current = st.session_state.get(f"{prefix}_model", models[0])
        if current not in models:
            current = models[0]
        model = st.selectbox("モデル", models, index=models.index(current), key=f"{prefix}_model_select")
    st.session_state[f"{prefix}_provider"] = provider
    st.session_state[f"{prefix}_model"] = model
    if not _api_key(provider):
        st.warning(f"{provider} の API キーが未設定です。⚙️ 設定ページで入力してください。")
    return provider, model


# === ⚙️ Settings ==============================================================

def render_settings() -> None:
    st.title("⚙️ 設定")
    st.write(
        "API キーはこのセッション (`st.session_state`) にのみ保持され、リポジトリには保存されません。"
        " Streamlit Cloud に常時設定したい場合は、Settings → Secrets に "
        "`anthropic_api_key` と `google_api_key` を登録してください。"
    )

    keys = st.session_state["api_keys"]
    with st.form("api_keys_form"):
        claude = st.text_input(
            "Anthropic API key", value=keys.get("claude", ""), type="password",
            help="https://console.anthropic.com/ で取得",
        )
        gemini = st.text_input(
            "Google AI Studio API key", value=keys.get("gemini", ""), type="password",
            help="https://aistudio.google.com/apikey で取得",
        )
        if st.form_submit_button("保存"):
            st.session_state["api_keys"] = {"claude": claude.strip(), "gemini": gemini.strip()}
            st.success("セッションに保存しました")

    st.subheader("接続テスト")
    test_col1, test_col2 = st.columns(2)
    if test_col1.button("Claude にハロー", disabled=not _api_key("claude")):
        try:
            out = llm_router.chat(
                "claude", llm_router.CLAUDE_MODELS[-1], _api_key("claude"),
                system="Respond with exactly: OK",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=16,
            )
            st.success(f"Claude: {out}")
        except Exception as e:
            st.error(f"Claude エラー: {e}")
    if test_col2.button("Gemini にハロー", disabled=not _api_key("gemini")):
        try:
            out = llm_router.chat(
                "gemini", llm_router.GEMINI_MODELS[-1], _api_key("gemini"),
                system="Respond with exactly: OK",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=16,
            )
            st.success(f"Gemini: {out}")
        except Exception as e:
            st.error(f"Gemini エラー: {e}")


# === 📥 Pipeline ==============================================================

def render_pipeline() -> None:
    st.title("📥 データ投入")
    st.write(
        "ファイルをアップロードすると、LLM がスキーマに合わせて構造化し、"
        "対応する SQLite + FTS5 DB に書き込みます。"
    )

    schema_opts = {sc.display: sc.name for sc in schemas.all_schemas()}
    chosen_display = st.selectbox("投入先 DB", list(schema_opts.keys()))
    schema_name = schema_opts[chosen_display]
    sc = schemas.get(schema_name)
    with st.expander("このスキーマのフィールド"):
        st.dataframe(pd.DataFrame(sc.columns, columns=["column", "sql_type"]), hide_index=True)

    st.subheader("構造化に使う LLM")
    provider, model = _provider_model_selector("ingest", "gemini")

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
        log_area = st.container()
        for i, (name, data) in enumerate(items, 1):
            with log_area:
                with st.status(f"処理中: {name}", expanded=False) as status:
                    try:
                        result = pipeline.ingest(
                            schema_name=schema_name,
                            filename=name,
                            data=data,
                            provider=provider,
                            model=model,
                            api_key=_api_key(provider),
                        )
                        status.update(label=f"✅ {name} (id={result['id']})", state="complete")
                        st.json(result["record"])
                    except Exception as e:
                        status.update(label=f"❌ {name}: {e}", state="error")
            progress.progress(i / len(items))
        st.success(f"{len(items)} 件の処理が完了しました")


# === 📚 Data management =======================================================

def render_data() -> None:
    st.title("📚 データ管理")

    tabs = st.tabs([sc.display for sc in schemas.all_schemas()])
    for tab, sc in zip(tabs, schemas.all_schemas()):
        with tab:
            n = db_manager.count_records(sc.name)
            st.caption(f"{n} 件のレコード")

            rows = db_manager.list_records(sc.name, limit=200)
            if rows:
                df = pd.DataFrame([dict(r) for r in rows])
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.info("レコードがありません")

            with st.expander("DB ファイルのエクスポート / インポート"):
                col1, col2 = st.columns(2)
                with col1:
                    data = db_manager.export_db_bytes(sc.name)
                    st.download_button(
                        f"{sc.name}.db をダウンロード",
                        data=data if data else b"",
                        file_name=f"{sc.name}.db",
                        mime="application/octet-stream",
                        disabled=not data,
                    )
                with col2:
                    up = st.file_uploader(
                        f"{sc.name}.db をアップロード (上書き)",
                        type=["db"], key=f"up_{sc.name}",
                    )
                    if up and st.button("上書き実行", key=f"btn_up_{sc.name}"):
                        db_manager.import_db_bytes(sc.name, up.read())
                        st.success("上書きしました。ページを再読み込みしてください。")

            with st.expander("レコード削除"):
                rid = st.number_input(
                    "削除する id", min_value=0, step=1, value=0, key=f"del_{sc.name}",
                )
                if st.button("削除", key=f"btn_del_{sc.name}", disabled=rid <= 0):
                    db_manager.delete_record(sc.name, int(rid))
                    st.success(f"id={rid} を削除しました")
                    st.rerun()


# === 💬 Chat ==================================================================

def render_chat() -> None:
    st.title("💬 チャット")

    cfg_col, dbs_col = st.columns([2, 3])
    with cfg_col:
        st.subheader("回答 LLM")
        provider, model = _provider_model_selector("chat", "claude")
    with dbs_col:
        st.subheader("参照する DB")
        cols = st.columns(len(list(schemas.all_schemas())))
        selected: list[str] = []
        for col, sc in zip(cols, schemas.all_schemas()):
            with col:
                checked = st.checkbox(
                    sc.display, value=sc.name in st.session_state["selected_dbs"],
                    key=f"db_check_{sc.name}",
                )
                if checked:
                    selected.append(sc.name)
        st.session_state["selected_dbs"] = selected

    if st.button("会話をリセット", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("営業に関する質問を入力 (例: 製造業の決裁者攻略法は?)")
    if not prompt:
        return
    if not _api_key(provider):
        st.error(f"{provider} の API キーが未設定です。⚙️ 設定ページで入力してください。")
        return
    if not selected:
        st.warning("参照 DB を 1 つ以上選択してください")
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    hits = rag_engine.preview_hits(prompt, selected)
    with st.chat_message("assistant"):
        if hits:
            with st.expander(f"参照ヒット {len(hits)} 件", expanded=False):
                for h in hits:
                    sc = schemas.get(h.schema_name)
                    st.markdown(f"**{sc.display} #{h.record_id}** (score={h.score:.2f})")
                    st.code(h.snippet, language="text")
        history_for_llm = [m for m in st.session_state["messages"][:-1]]
        try:
            stream = rag_engine.stream_answer(
                question=prompt,
                selected_schemas=selected,
                provider=provider,
                model=model,
                api_key=_api_key(provider),
                chat_history=history_for_llm,
            )
            answer = st.write_stream(stream)
        except Exception as e:
            answer = f"⚠️ エラー: {e}"
            st.error(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})


# --- route --------------------------------------------------------------------

if page.startswith("⚙️"):
    render_settings()
elif page.startswith("📥"):
    render_pipeline()
elif page.startswith("📚"):
    render_data()
else:
    render_chat()
