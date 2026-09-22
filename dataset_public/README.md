# Artificial Memory Arena: Public Benchmark Specification

このディレクトリは、AI Memory Systems を評価するための公開仕様書およびスキーマ定義です。

---

## 1. 難易度階層（4 Difficulty Tiers）

| 難易度 | 定義 | 期待される挙動 |
| :--- | :--- | :--- |
| **Easy** | 単一ターンでの直接言及 | 単純な単語マッチングまたは直接ベクトル検索で想起可能 |
| **Medium** | 複数セッションを跨ぐ言及と言い換え | 会話セッションの横断結合、同義語の理解が必要 |
| **Hard** | 表面的なキーワードなし（文脈依存） | 表層一致に頼らず、文脈と関係性から対象を特定 |
| **Adversarial** | 衝突・多段階更新・証拠/真実分離 | 類似名称・過去の破棄された情報などの罠を回避 |

---

## 2. 評価カテゴリ（10 Categories）

1. `factual_recall`: 事実想起
2. `multi_session_recall`: 複数セッションに跨るエピソード想起
3. `temporal_reasoning`: 時間制約・期間判定
4. `knowledge_updates`: 情報の更新・最新真実の追跡
5. `contradiction_handling`: 発言者間の矛盾・不一致の検知
6. `indirect_recall`: 2ホップ以上の間接想起
7. `distractor_resistance`: 大量の無関係なノイズに対する耐性
8. `abstention`: 未知・未設定プロパティに対する適切な棄権（幻覚防止）
9. `compression_recovery`: 圧縮後の数値・記号・シリアルの正確な復元
10. `reflection_reinterpretation`: 行動ログからの嗜好・傾向の抽象化

---

## 3. Blind Benchmark 規約
評価の公平性を保つため、実際のテストケース・シナリオ（100問）および敵対的テスト（Adversarial）は `dataset_hidden/` に隔離され、評価時のみ読み込まれます。
