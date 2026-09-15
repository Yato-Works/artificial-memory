# Artificial Memory / Context Runtime

> **圧縮による忘却。解像度による想起。出所（Provenance）を伴う推論。**

[English](README.md) | [日本語](README.ja.md)

**Artificial Memory** は、持続的な AI システムのための**認知メモリランタイム（Cognitive Memory Runtime）**です。段階的な圧縮による忘却、必要に応じた適応型の解像度展開、時間推論（タイムトラベル）、および厳密な情報出所（Provenance）追跡を備え、人間の記憶に近い認知基盤を提供します。

---

## 🌟 ビジョン (Vision)

Artificial Memory は、単なる「高性能な RAG（検索拡張生成）」ではありません。AI メモリのライフサイクル全体を一貫して管理する**汎用認知メモリランタイム**です。

* **段階的なメモリ圧縮 (Progressive Memory Compression)** — 忘却とは「完全削除」ではなく「解像度の低下（Resolution Down）」であるという設計思想。
* **適応型解像度による想起 (Adaptive-Resolution Recall)** — トークン消費に対して最大の情報価値（ユーティリティ）が得られる場合のみ、メモリを高解像度に展開。
* **中間表現としてのコンテキスト (Context IR)** — メモリ → 想起 → 優先順位付け → 解像度展開 → トークン予算配分 → Context IR → LLM という一貫したコンパイルパイプライン。
* **第一級プロパティとしての出所追跡 (Provenance)** — すべての記憶がどの会話・どのメッセージから生成されたかを完全に追跡可能。
* **進化する記憶 (Memory Evolution)** — 生成 → アクセス → 圧縮 → 展開 → 矛盾検出 → 改訂 → 統合 → アーカイブ → 再コンパイルという状態遷移。
* **矛盾検出と信念管理 (Belief Management)** — 「客観的証拠（Evidence）」と「解釈・信念（Belief）」を明確に分離し、時間の経過に伴う矛盾を明示的に追跡。
* **時間的記憶とタイムトラベル (Temporal Memory & Time Travel)** — 「ある時点 T において、システムは何を知り、何を信じていたか」を過去に遡って正確に再構築。
* **自己修復と完全性検証 (Integrity & Self-Healing)** — 圧縮時のセマンティック維持スコアを計測し、破損や陳腐化を検知して自動再コンパイル。
* **フェデレーション (Federated Multi-Agent)** — 複数エージェント間での信頼度・有効期限・出所を伴うポリシー駆動型メモリ共有。
* **エンタープライズ・ガバナンス** — データ分類、アクセス権限、監査ログ、テナント分離、保持ポリシーをアーキテクチャ層で担保。

---

## ✨ 主な機能

### 🧠 コア・メモリシステム
* **6段階のメモリ解像度モデル**:
  * **RAW (Level 0)**: 会話原文そのまま（1.0x）
  * **LIGHT (Level 1)**: 冗長性除去・フィラー保持（約2〜3倍圧縮）
  * **EPISODE (Level 2)**: 出来事の流れ・ナラティブ要約（約5〜10倍圧縮）
  * **SEMANTIC (Level 3)**: 決定事項・重要ファクト（約20〜50倍圧縮）
  * **LONG_TERM (Level 4)**: 抽象化されたパターンや原則（約100倍以上圧縮）
  * **DEEP_LONG_TERM (Level 5)**: コアバリュー・アイデンティティ（約500倍以上圧縮）
* **トピック別自動整理**: 会話内容からトピックの自動分類および階層管理
* **Memory IR / Context IR**: コンパイル・想起・コンテキスト構築のための型安全な中間表現

### 🔍 高度な検索・想起機能
* **ベクトル検索 & ハイブリッド検索**: FAISS（ローカル）/ pgvector（PostgreSQL）によるキーワード＋ベクトルのハイブリッド想起
* **時間軸クエリ**: `valid_from` / `valid_until` による特定時点の記憶探索
* **連想記憶グラフ**: メモリ間の依存関係と連想パスの探索
* **反事実的想起 (Counterfactual Recall)**: 単なる類似度だけでなく、「その記憶が最終的な回答に与えた影響度」を計測

### 🤖 LLM & エージェント統合
* **コンテキスト・ランタイム**: トークン予算上限内で最適な情報密度を実現する自動コンテキスト構築
* **マルチプロバイダ対応**: Ollama（ローカルファースト）および OpenAI API
* **MCP (Model Context Protocol) サーバー**: Cline、Claude Desktop、Hermes などの自律型エージェントから直接呼び出し可能な7つの専用ツール群を提供

### 🔐 セキュリティ & マルチテナント
* **ユーザー・権限管理**: ユーザー登録、ロールベースアクセス制御、セッショントークン
* **APIキー**: 有効期限・スコープ付きAPIキー
* **テナント分離**: プロジェクト単位・ユーザー単位での記憶の完全分離

### 📊 観測性と研究基盤
* **リアルタイムメトリクス**: 圧縮比率、想起精度、トークン節約効果の計測
* **再現可能なベンチマーク**: 決定論的想起テスト、Ollama QAベンチマーク、アブレーション評価
* **メモリ・デバッガ**: 「なぜその記憶が選ばれたのか／却下されたのか」「なぜ解像度が展開されたのか」を説明可能

---

## 📈 実証ベンチマーク (Benchmarks)

`scripts/run_readme_benchmark.py` による実測値（Python 3.11, embeddings: all-MiniLM-L6-v2, FAISS IndexFlatIP）：

| 評価指標 | 多様なドメイン (Diverse) | ストレス環境 (Near-duplicate) |
|---|---|---|
| **セマンティック想起精度@1 (ベクトル / ハイブリッド)** | **66.7% / 79.2%** | 29.2% / 41.7% |
| **セマンティック想起精度@5 (ベクトル / ハイブリッド)** | **100% / 100%** | 57.5% / 72.5% |
| **想起レイテンシ (p50 / p95)** | **27 ms / 32 ms** | 26 ms / 28 ms |
| **全文注入に対するトークン削減率** | **69.6% 削減** | **70.9% 削減** |
| **書き込みスループット** | ~530 writes/s | ~550 writes/s |

### ローカル LLM (Ollama) による質問応答精度とトークン効率
プロジェクトの固有名詞や技術選定に関する合成質問（20問）に対する回答精度：

| 使用モデル | 記憶なし | 全文コンテキスト注入 | **Artificial Memory 想起 (600 tok 制限)** |
|---|---|---|---|
| **qwen2.5:1.5b** | 0% | 90% (1141 tok) | **100% (382 tok)** |
| **qwen3:4b** | 5% | 100% (1141 tok) | **100% (382 tok)** |

👉 **約66%〜70% のトークンを削減**しながら、全文コンテキスト注入と同等以上の正解率を達成しています。

---

## 🚀 クイックスタート (Quick Start)

### 1. インストール

```bash
# リポジトリのクローン
git clone https://github.com/Yato-Works/artificial-memory.git
cd artificial-memory

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Web API, LLM, Vector extras を含めてインストール
pip install -e ".[web,llm,vector]"
```

> **Note**: `vector` extra は FAISS と sentence-transformers をインストールします（セマンティック検索に必要）。未インストールの場合はキーワード検索モードで動作します。

### 2. Docker による起動 (推奨)

```bash
docker-compose up -d
```
* Web UI: [http://localhost:8000/ui](http://localhost:8000/ui)
* API Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. CLI による基本操作

```bash
# 会話セッションの開始
python -m artificial_memory start --topic "新規プロジェクト"

# ユーザー発言・アシスタント発言の記録
python -m artificial_memory user "認証方式にはJWTとOAuth2を採用しましょう。"
python -m artificial_memory assistant "了解しました。セキュリティ要件を整理します。"

# 会話終了とメモリコンパイル
python -m artificial_memory end

# 適応型想起
python -m artificial_memory recall "認証方式は何でしたか？"

# 記憶の出所（Provenance）トレース
python -m artificial_memory trace 1
```

---

## 🔌 MCP (Model Context Protocol) 連携

Cline、Claude Desktop、Hermes などのエージェントツールと直接連携できます。

### 提供される 7 つの MCP ツール
1. `memory_remember`: 会話や事実を記憶
2. `memory_recall`: 質問やキーワードに応じた関連記憶の想起
3. `memory_trace`: 記憶の根拠となった発言・会話の出所追跡
4. `memory_timeline`: トピックの時系列変化の取得
5. `memory_explain`: 想起エンジンの決定プロセスの可視化
6. `memory_expand`: 記憶の解像度をより詳細なレベルへ展開
7. `memory_inspect`: Memory IR の内部構造の検査

### 設定例 (`claude_desktop_config.json` または Cline の MCP 設定)

```json
{
  "mcpServers": {
    "artificial-memory": {
      "command": "python",
      "args": ["-m", "artificial_memory.mcp"],
      "cwd": "/path/to/artificial-memory"
    }
  }
}
```

詳細な検証レポートおよび導入ガイドは [docs/mcp-agent-testing.md](docs/mcp-agent-testing.md) および [docs/real-agent-e2e.md](docs/real-agent-e2e.md) をご覧ください。

---

## 🧪 テストの実行

```bash
# 全テストスイートの実行
pytest tests -v

# 高速テスト（コア機能）
pytest tests/test_core.py -v

# コードスタイルのチェック
ruff check src tests

# 型チェック
mypy src/artificial_memory
```

---

## 🌟 導入事例・Showcase (Used by)

Artificial Memory を個人開発、スタートアップ、企業の本番サービス、または研究でご利用ですか？  
ぜひ [SHOWCASE.md](SHOWCASE.md) にご登録いただくか、[Showcase 登録 Issue](https://github.com/Yato-Works/artificial-memory/issues/new?template=showcase.yml) よりお知らせください！（※ 企業名非公開・業界名のみでの登録も歓迎しています）

---

## 📜 ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。

---

## ✍️ 引用 (Citation)

研究やプロジェクトで本システムを利用される場合は、以下の BibTeX をご活用ください：

```bibtex
@software{artificial-memory,
  title = {Artificial Memory / Context Runtime},
  subtitle = {A Cognitive Memory Runtime for Persistent AI Systems},
  author = {Yato-Works},
  year = {2026},
  url = {https://github.com/Yato-Works/artificial-memory}
}
```
