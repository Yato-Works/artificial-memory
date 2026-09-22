# Artificial Memory / Context Runtime

> **決定論的メモリランタイム：構造化想起、時間推論、矛盾検知、および棄権（Abstention）を備えた認知基盤**
> 
> *圧縮による忘却。解像度による想起。出所（Provenance）を伴う推論。*

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
* **矛盾検出と信念管理 (Belief Management / Truth vs. Evidence)** — 「過去に言及された事実（Evidence）」と「現在の有効な状態・決定（Truth）」を明確に分離し、時間の経過に伴う矛盾を明示的に追跡。
* **時間的記憶とタイムトラベル (Temporal Memory & Time Travel)** — 「ある時点 T において、システムは何を知り、何を信じていたか」を過去に遡って正確に再構築。
* **決定論的 IR 抽出とゼロ LLM コール書き込み** — メモリ登録時に高価で不確実な LLM 呼び出しを一切行わず（Write LLM = 0）、構文・意味論的解析により決定論的に構造化 IR を抽出。
* **自己修復と完全性検証 (Integrity & Self-Healing)** — 圧縮時のセマンティック維持スコアを計測し、破損や陳腐化を検知して自動再コンパイル。
* **フェデレーション (Federated Multi-Agent)** — 複数エージェント間での信頼度・有効期限・出所を伴うポリシー駆動型メモリ共有。
* **エンタープライズ・ガバナンス** — データ分類、アクセス権限、監査ログ、テナント分離、保持ポリシーをアーキテクチャ層で担保。

---

## ✨ 主な機能

### 🧠 コア・メモリシステム
* **6段階のメモリ解像度モデル**: RAW (Level 0) から DEEP_LONG_TERM (Level 5)
* **トピック別自動整理**: 会話内容からトピックの自動分類および階層管理
* **Memory IR / Context IR**: コンパイル・想起・コンテキスト構築のための型安全な中間表現
* **Universal IR Extractor & Resolver**: ドメイン固有のヒューリスティックを排除し、エンティティ・時間枠・矛盾ゲートを決定論的に解決

### 🔍 高度な検索・想起・失敗帰属診断 (Failure Attribution)
* **メモリ・LLM 分離評価 (Decoupled Evaluation)**: メモリ想起能力（Test A）と回答生成 LLM（Test B）を厳密に分離して性能測定
* **失敗帰属診断 (Failure Attribution)**: 失敗ケースを 8 項目のトレースログで追跡し、4 分類（想起失敗 / 解決失敗 / LLM 推論失敗 / 評価器失敗）へ自動帰属
* **ベクトル検索 & ハイブリッド検索**: FAISS（ローカル）/ pgvector（PostgreSQL）によるキーワード＋ベクトルのハイブリッド想起
* **時間軸クエリ**: `valid_from` / `valid_until` による特定時点の記憶探索
* **連想記憶グラフ**: メモリ間の依存関係と連想パスの探索
* **反事実的想起 (Counterfactual Recall)**: 単なる類似度だけでなく、「その記憶が最終的な回答に与えた影響度」を計測

### 🤖 LLM & エージェント統合
* **コンテキスト・ランタイム**: トークン予算上限内で最適な情報密度を実現する自動コンテキスト構築
* **マルチプロバイダ対応**: Ollama（ローカルファースト）および OpenAI API
* **MCP (Model Context Protocol) サーバー**: Cline、Claude Desktop、Hermes などの自律型エージェントから直接呼び出し可能な専用ツール群を提供

### 🔐 セキュリティ & マルチテナント
* **ユーザー・権限管理**: ユーザー登録、ロールベースアクセス制御、セッショントークン
* **APIキー**: 有効期限・スコープ付きAPIキー
* **テナント分離**: プロジェクト単位・ユーザー単位での記憶の完全分離

### 📊 観測性と研究基盤
* **リアルタイムメトリクス**: 圧縮比率、想起精度、トークン節約効果の計測
* **再現可能なベンチマーク**: 競合設定（`benchmark_config/*.yaml`）を完全固定し、公開スキーマ（`dataset_public/`）と未知テスト（`dataset_hidden/`）で評価
* **レッドチーム対抗試験**: Entity/Semantic/Temporal 衝突、矛盾、Truth vs Evidence を突く敵対的破壊セット

---

## 📈 実証ベンチマーク (Scientific Memory Benchmark)

固定された競合実装（**Mem0**, **MemGPT**, **MemoryBank**, **Simple RAG**）および凍結されたローカルモデル（`phi4-mini:3.8b`）を用いた厳密な科学的比較実験。

### 1. 敵対的衝突スイート（AM 破壊セット / Adversarial Collision）
エンティティ衝突、意味衝突、時間衝突、矛盾、および **Truth vs. Evidence（提案履歴と現決定の混同）** を突く極限テスト。

| システム | Test A (想起正解率) | Test B (回答正解率) | 偽陽性率 (FPR) | 棄権正解率 (Abstention) | コンテキスト消費 (Tokens/Q) | 登録時 LLM 呼出 (Write LLM) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Artificial Memory (AM)** | **100.0%** | **83.3%** | **16.7%** | **83.3%** | **118** | **0** |
| MemGPT | 66.7% | 83.3% | 0.0% | 100.0% | 311 | 0 |
| Mem0 | 66.7% | 66.7% | 33.3% | 66.7% | 144 | 15 |
| MemoryBank | 66.7% | 66.7% | 16.7% | 83.3% | 200 | 0 |
| Simple RAG | 66.7% | 66.7% | 33.3% | 66.7% | 195 | 0 |

> **アーキテクチャ上の主要な洞察**:
> - **メモリと LLM の分離評価**: AM は **Test A (想起精度) において 100.0%** を達成。Test B で 83.3% となった 1 件の失敗は、**Failure Attribution（失敗帰属診断）** により AM のメモリ欠陥ではなく「3.8B LLM が時間的推論を誤認した（`LLM_Reasoning_Failure`）」と明確に特定されました。
> - **Truth vs. Evidence の分離**: 競合手法は過去の廃案（Evidence）と現在の決定（Truth）を混同し、最大 33.3% の偽陽性率（FPR）を記録。AM はこれらを明確に分離し、誤った断定を回避。
> - **圧倒的なリソース効率**: AM はわずか **118 tokens/Q**（MemGPT の 38%、Mem0 の 82%）しかコンテキストを消費せず、書き込み時の LLM 呼び出しも **0 回** を維持。

### 2. 100問 階層型ベンチマーク (Scaled Suite)
4段階の難易度（**Easy**: 単一事実、**Medium**: マルチホップ、**Hard**: 時間・矛盾、**Adversarial**: 衝突）で構成。
- 公開仕様・スキーマ: `dataset_public/`
- 非公開評価スイート: `dataset_hidden/arena_100.json`

ベンチマークおよび失敗診断の実行:
```bash
# 破壊セットと失敗帰属診断の実行
python scripts/run_scaled_benchmark.py --suite adversarial --diagnose

# 100問スイートの実行
python scripts/run_scaled_benchmark.py --suite 100
```

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
