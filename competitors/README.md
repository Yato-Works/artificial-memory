# Competitor Memory Systems Audit & Protocols

このディレクトリは、Artificial Memory (AM) との比較対象となる各外部メモリシステムの**選定根拠、公式アーキテクチャ対比、プロンプト、および再現プロトコル**を完全にドキュメント化したものです。

---

## 1. 競合システムの選定根拠

| システム | 代表文献 / OSS | 選定理由 | 記憶アーキテクチャの型 |
| :--- | :--- | :--- | :--- |
| **Mem0** | `mem0ai/mem0` (2024) | 現在最も広く使われているエージェントメモリOSS。グラフ＋ベクトル検索の代表格。 | 事実抽出 (LLM) + ベクトル・キーワード・エンティティ融合 |
| **MemGPT / Letta** | Packer et al. (2023) | OS型階層メモリの先駆。Working Context と Archival Memory の階層構造。 | 階層型 (Core FIFO + Archival Vector) |
| **MemoryBank** | Zhong et al. (2023) | 認知心理学（エビングハウスの忘却曲線）を導入した長期的対話メモリの代表研究。 | 時間減衰型 (Ebbinghaus Forgetting Curve) |
| **Simple RAG** | Standard Chunking RAG | 最も一般的で素朴なベクトル検索ベースライン。 | チャンク化ベクトル検索 (Cosine Top-k) |

---

## 2. 共通評価プロトコル (Controlled Arena Fairness Rules)

すべての競合システムは、**「記憶・想起アーキテクチャの性能差のみ」** を測定するため、以下の公平性ルール（Fairness Rules）の下で実行されます：

1. **同一の凍結LLM**:
   - 抽出（Write側）および回答生成（Read側）には、すべてのプレイヤーで同一のローカルLLM（`phi4-mini:latest` @ `http://localhost:11434`）を使用。
   - LLM自身の賢さや事前学習知識の差異を排除。
2. **同一の埋め込みモデル**:
   - ベクトル検索には全システム共通で `all-MiniLM-L6-v2` (384次元) を使用。
3. **厳格なコンテキスト予算 (Context Budget)**:
   - 全システム一律で **2,000 トークン** を上限とし、超過時はプロトコル違反（Budget Violation）として自動記録。
4. **完全なコスト・レイテンシ計測**:
   - 書込時LLM呼出数（Write LLM Calls）、生成トークン数、壁時計レイテンシを全システムで厳格に計測。

---

## 3. ディレクトリ構成

- `mem0/`: Mem0 の仕様書 (`spec.md`) およびアダプタ (`adapter.py`)
- `memgpt/`: MemGPT の仕様書 (`spec.md`) およびアダプタ (`adapter.py`)
- `memorybank/`: MemoryBank の仕様書 (`spec.md`) およびアダプタ (`adapter.py`)
- `simple_rag/`: Simple RAG の仕様書 (`spec.md`) およびアダプタ (`adapter.py`)
