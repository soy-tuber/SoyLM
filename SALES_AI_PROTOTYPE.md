# 営業インストラクター AI — Streamlit プロトタイプ

SoyLM をベースにした「営業インストラクター AI」設計書 (`設計書` を参照)
の Phase 1〜3 を、Streamlit 1 ファイルで動くプロトタイプとして実装したもの。

そのまま **Streamlit Cloud Community** にデプロイできる構成。

## 構成

```
streamlit_app.py          # エントリポイント (Streamlit Cloud が自動検出)
sales_ai/
  ├ schemas.py            # 5 つの DB スキーマ定義 (提案/トーク/レポート/成果/フォロー)
  ├ db_manager.py         # SQLite + FTS5 (trigram tokenizer for 日本語)
  ├ llm_router.py         # Claude / Gemini ストリーミング統一 IF
  ├ pipeline.py           # ファイル抽出 → LLM 構造化 → DB 書き込み
  └ rag_engine.py         # マルチ DB 並列 FTS5 + コンテキスト合成
.streamlit/
  ├ config.toml           # テーマ・アップロード上限
  └ secrets.toml.example  # API キー雛形 (実ファイルは .gitignore)
```

既存の SoyLM (FastAPI / `app.py`) はそのまま残しています。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud Community へのデプロイ

1. このリポジトリを Streamlit Cloud から選択
2. Main file: `streamlit_app.py`
3. Python version: 3.11 推奨
4. (任意) **Settings → Secrets** に下記を登録:

   ```toml
   anthropic_api_key = "sk-ant-..."
   google_api_key    = "AIza..."
   ```

   登録しない場合は、各ユーザーが UI の「⚙️ 設定」ページで自分のキーを
   貼り付けて利用します (キーは `st.session_state` にのみ保持され、
   ディスク・リポジトリには書き出されません)。

## 機能

- **💬 チャット**: 参照 DB をチェックボックスで選び、Claude/Gemini にストリーミング応答させる
- **📥 データ投入**: PDF / DOCX / PPTX / XLSX / CSV / TXT / MD をアップロード、LLM がスキーマ通りの JSON に構造化して DB に書き込む
- **📚 データ管理**: 各 DB のレコード一覧、`.db` ファイルのダウンロード/アップロード、行削除
- **⚙️ 設定**: API キー設定 + 接続テスト

## データ永続化について

Streamlit Cloud の作業ディスクは **再起動で消える** ため、本プロトタイプは
データを `data/sales_ai/{schema}.db` に置きますが、これは揮発性です。
**「📚 データ管理」ページから `.db` ファイルをダウンロードしてバックアップ**
してください。再デプロイ後は同ページから `.db` をアップロードすれば復元できます。

恒久的な保存が必要になったら、`db_manager.py` の `DATA_DIR` を Turso /
Supabase / S3 などへ差し替えてください。

## 設計書からの差分・注意点

- **vLLM プロバイダーは未実装**: Streamlit Cloud から自前 GPU サーバへ到達できない前提のため、Claude / Gemini のみ。`llm_router.py` に `_vllm_stream` を足せば復活可能。
- **FTS5 トークナイザに `trigram` を採用**: SQLite 標準の `unicode61` は CJK を 1 トークン扱いで日本語検索が壊滅するため、3-gram で部分一致するように切替済 (SQLite 3.34+ で利用可能、Streamlit Cloud の Python ランタイム同梱で動作確認済)。
- **音声入力 (Whisper) は未実装**: Phase 4 扱い。
- **暗号化 (Fernet) は未実装**: プロトタイプでは API キーをセッションメモリのみに保持して回避。
