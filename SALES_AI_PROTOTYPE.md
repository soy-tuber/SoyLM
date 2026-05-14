# 営業インストラクター AI — Streamlit プロトタイプ

SoyLM をベースにした「営業インストラクター AI」プロトタイプ。
Streamlit Cloud Community でそのままデプロイできます。

## 3 つの主機能

| 機能 | 役割 | 推奨モデル |
|---|---|---|
| 🎯 **訪問前ブリーフィング** | 売りたい商品 + 訪問先情報を入力 → 類似する成約事例を抽出し、キーメッセージ・想定質問・反論対応・差別化ポイント・準備チェックリストを提示 | Claude Sonnet |
| 📝 **提案書作成** | 顧客 / 商品 / 課題 / 要件を入力 → チャンピオン事例を引用しながら **Markdown 形式の提案書** を生成 | Claude Opus |
| 🔄 **訪問後フォローアップ** | 訪問レポート本文を入力 → 商談評価 (A〜D)・**ToDo 表 (期限付き)**・推奨フォローメール文案・リスク分析を出力 | Claude Sonnet |

## データモデル

```
deals (案件マスタ)
  ├ A. proposals     (提案資料)
  ├ B. talks         (営業トーク)
  ├ C. reports       (レポート)
  ├ D. results       (成果)
  └ E. followups     (フォローメール/電話)
```

- 「チャンピオン事例」= `deals.outcome = '成約'` のレコード
- すべての子テーブルは `deal_id` を持ち、案件単位で 5 点セットとして紐づく
- 起動時に **サンプルチャンピオン事例 10 件** を自動シード (`sales_ai/seed.py`)

## 起動

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud デプロイ

- Repository: `soy-tuber/SoyLM`
- Branch: `streamlit-prototype`
- Main file path: **`streamlit_app.py`**
- (任意) Settings → Secrets:
  ```toml
  anthropic_api_key = "sk-ant-..."
  google_api_key    = "AIza..."
  ```

未登録でも、UI の「⚙️ 設定」で各ユーザーが自分のキーを貼って動作させられます。

## 構成

```
streamlit_app.py            # 7 ページ (Home / 3 features / 案件マスタ / 投入 / 設定)
sales_ai/
  ├ schemas.py              # deals master + 5 child schemas
  ├ db_manager.py           # 単一 SQLite + FTS5 (trigram for 日本語)
  ├ llm_router.py           # Claude / Gemini ストリーミング統一 IF
  ├ pipeline.py             # ファイル → LLM 構造化 → 子テーブル INSERT
  ├ rag_engine.py           # 類似 deal の取得 + バンドル合成
  ├ features.py             # 3 機能のプロンプト + オーケストレーション
  └ seed.py                 # 10 件のサンプル champion deal
.streamlit/
  ├ config.toml
  └ secrets.toml.example
```

## 案件マスタの育て方 (今後 100 件まで拡大)

1. 「📚 案件マスタ → 新規登録」で案件 (`deals` 行) を作成
2. 「📥 データ投入」でその案件 ID を指定し、A〜E の 5 種類の資料をアップロード
3. LLM が各スキーマに合わせて JSON 構造化し、子テーブルに INSERT
4. 「📚 案件マスタ → 詳細 / 編集」で全コンポーネントが揃ったか確認

クロージングまで到達 (`outcome='成約'`) した案件のみがチャンピオン事例として
3 機能のコンテキストに使われます。

## データ永続化

Streamlit Cloud のディスクは再起動で消えます。「📚 案件マスタ → DB エクスポート」
から `sales_ai.db` をダウンロードして退避してください。再デプロイ後に
同タブからアップロードすれば復元できます。

## 設計書からの差分・注意点

- **vLLM 未実装** — Streamlit Cloud から GPU サーバへ到達できない前提
- **マルチモーダル未実装** — 音声/画像入力は今後の課題 (Phase 4)
- **FTS5 トークナイザ: `trigram`** — `unicode61` は日本語が単一トークン化で検索不能になるため
- **API キー暗号化なし** — セッションメモリのみで運用、永続化しないことで回避
