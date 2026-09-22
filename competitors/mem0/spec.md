# Mem0 System Specification & Audit

## 1. 概要
- **対象**: Mem0 OSS (v0.1.x)
- **公式リポジトリ**: `https://github.com/mem0ai/mem0`
- **アーキテクチャ分類**: Fact Extraction + Hybrid Vector/Graph Indexing

---

## 2. 実装詳細

### (1) 書き込みパイプライン (Write / Ingestion)
- **方式**: 単一パスの ADD-only 事実抽出 (Single-pass ADD-only fact extraction)。
- **抽出LLMプロンプト**:
  ```text
  You are a personal memory extraction assistant.
  Extract key facts, preferences, dates, tools, locations, and statements about the user or system from the given exchange.
  Output each fact as a bullet point. Do not add commentary.
  ```
- **呼出回数**: 会話の各ターン（User/Assistant）ごとに 1 回の LLM 呼出。
- **保存形式**: 抽出された箇条書きの各ファクトを、独立したセマンティックメモリとして SQLite + FAISS に保存。

### (2) 読み出しパイプライン (Read / Retrieval)
- **検索方式**: 3つのシグナルを決定論的に融合 (Multi-signal Fusion):
  1. **Semantic Score**: `all-MiniLM-L6-v2` によるベクトルコサイン類似度 (重み: 0.5)
  2. **Keyword Score**: Jaccard 類似度 (重み: 0.3)
  3. **Entity Score**: クエリ内の固有名詞・エンティティとの重なり率 (重み: 0.2)
- **Top-k**: `k=10`
- **コンテキスト構築**: 取得された上位メモリを改行区切りで結合し、最大2,000トークンの枠内でLLMへ提供。

---

## 3. なぜこの仕様で固定するのか？
1. **公式推奨との整合性**: Mem0 は「LLMによる構造化事実抽出」を中核価値としており、この書き込み時抽出を忠実に再現することで、Mem0本来の強みを反映。
2. **コストの客観性**: 書き込み時のLLM呼出数とトークン消費を1ターンごとに厳格に記録し、「高精度化のためにどれだけの推論コストを払っているか」を完全に可視化。
