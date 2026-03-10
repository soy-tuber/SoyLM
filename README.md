# SoyLM — Local-first Research Assistant

Gloss超え。FastAPI + Jinja2 の爆速NotebookLM代替。

## セットアップ

```bash
pip install -r requirements.txt

# 環境変数
export GEMINI_API_KEY="your-key-here"
export NEMOTRON_BASE="http://localhost:8000/v1"        # vLLM endpoint
export NEMOTRON_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1"

# 起動
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│  Browser (Jinja2 SSR + minimal JS)              │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────┐│
│  │ Sources  │ │  Chat (SSE)  │ │  Chat Logs   ││
│  │ Panel    │ │  Streaming   │ │  + Download  ││
│  └──────────┘ └──────────────┘ └──────────────┘│
└──────────────────┬──────────────────────────────┘
                   │ REST + SSE
┌──────────────────┴──────────────────────────────┐
│  FastAPI Backend (single file: app.py)          │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐              │
│  │ Flash Loader │  │ RAG Search   │              │
│  │ (Gemini 2.5  │  │ (FTS5 BM25)  │              │
│  │  Flash)      │  │              │              │
│  └──────┬──────┘  └──────┬───────┘              │
│         │                │                      │
│  ┌──────┴────────────────┴───────┐              │
│  │       SQLite (soylm.db)       │              │
│  │  notebooks | sources | FTS5   │              │
│  │  chatlogs  | messages         │              │
│  └───────────────────────────────┘              │
│                                                 │
│  LLM Providers:                                 │
│  ├─ Nemotron (vLLM localhost, default)          │
│  ├─ Gemini 2.5 Flash (preprocessing + chat)     │
│  └─ Gemini 2.5 Pro (full context mode)          │
│                                                 │
│  Tools:                                         │
│  ├─ DDG Search (Web検索)                         │
│  └─ Fact Checker (Gemini Flash + DDG)           │
└─────────────────────────────────────────────────┘
```

## 機能

- **ソースアップロード**: ファイル / URL / YouTube / テキスト貼付（最大50件）
- **Flash ロード**: Gemini 2.5 Flash で要約・キーポイント抽出 → SQLite格納
- **3カラムUI**: ソース | チャット(SSE) | チャット履歴
- **モデル切替**: Nemotron(ローカル) / Flash / Pro
- **コンテキスト全投入**: Pro選択時、Flash前処理データを全投入
- **DDG検索**: チャット内でWeb検索結果を統合
- **ファクトチェッカー**: Gemini Flash + DDG で回答を検証
- **チャットログ保存 + JSONダウンロード**
- **hash重複排除**: 同一ソースの二重登録防止

## ファイル構成

```
gloss_killer/
├── app.py              # 全バックエンドロジック（約550行）
├── templates/
│   ├── index.html      # ホーム（ノートブック一覧）
│   └── notebook.html   # メイン画面（3カラム）
├── data/
│   ├── soylm.db        # 自動生成
│   └── sources/        # ファイル保存用
├── requirements.txt
└── README.md
```
