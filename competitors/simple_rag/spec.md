# Simple RAG System Specification & Audit

## 1. 概要
- **対象**: Naive Chunk-based RAG
- **アーキテクチャ分類**: Standard Vector Retrieval Baseline

---

## 2. 実装詳細

### (1) 取り込み (Ingest)
- 会話の各発話をそのまま1つのチャンクとしてベクトル化（`all-MiniLM-L6-v2`）。
- メモリの構造化・要約・LLM抽出は一切行わず、生テキストを保存。

### (2) 検索 (Retrieve)
- クエリとのコサイン類似度で上位 Top-k（k=10）を取得。
- 取得したチャンクを結合し、最大2,000トークンの枠内でLLMへ提供。
