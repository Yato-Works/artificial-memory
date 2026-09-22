# Artificial Memory — 総合ベンチマーク・ダッシュボード

生成日: 2026-09-21 / データ源: `benchmark_results/**`, `benchmark/results/**`

---

## 1. ヘッドライン（全体成績）

| ベンチマーク | 問題数 | 精度 | Oracle Recall | 平均tokens | 平均latency |
|---|---:|---:|---:|---:|---:|
| LoCoMo-10 (AM v0.2) | 1,986 | **41.7%** | 45.8% | 154.8 | 341 ms |
| LoCoMo-10 (内蔵 baseline) | 1,986 | 42.5% | 46.0% | 153.7 | 342 ms |
| LongMemEval 500問 | 500 | **64.0%** | 93.0% | 638.9 | 1,050 ms |
| LongMemEval 冻結50問 | 50 | 78.0% | 78.0% | 2,940 | — |
| Temporal Compiler (conv-26) | 37 | **64.9%** | 91.9% | 423.8 | — |
| Answer Guard (Phase Immune) | 32 | **81.3%** | — | 382.5 | — |

## 2. LoCoMo-10 カテゴリ別（AM vs 内蔵baseline）

| カテゴリ | 問題数 | baseline | AM | Δ |
|---|---:|---:|---:|---:|
| adversarial | 446 | 35.4% | **39.7%** | **+4.3pp** |
| open-domain | 96 | 13.5% | **15.6%** | +2.1pp |
| single-hop | 841 | **50.3%** | 47.4% | −2.9pp |
| multi-hop | 282 | **23.4%** | 22.0% | −1.4pp |
| temporal | 321 | **57.6%** | 54.8% | −2.8pp |
| **micro合計** | 1,986 | **42.5%** | 41.7% | −0.8pp |

## 3. LongMemEval 500問 タイプ別（oracle→answer 変換率）

| タイプ | 問題数 | 正答 | Oracle | 変換率 |
|---|---:|---:|---:|---:|
| single-session-user | 70 | 66 | 67 | **94.3%** |
| single-session-preference | 30 | 28 | 18 | 93.3% |
| knowledge-update | 78 | 61 | 77 | 78.2% |
| temporal-reasoning | 133 | 100 | 123 | 75.2% |
| multi-session | 133 | 49 | 125 | **36.8%** |
| single-session-assistant | 56 | 16 | 55 | **28.6%** |

失敗分類 (180件): ABSTENTION 62 / COMPOSITION 53 / TEMPORAL 29 / STATE 17 / RETRIEVAL 8 / VERIFICATION 6 / GROUNDING 4 / ENTITY 1

## 4. 他手法との比較 (コントロールアリーナ 10問)

| プレイヤー | 精度 | pass/partial/fail | GT Coverage | Abstention精度 | 禁則違反 |
|---|---:|---|---:|---:|---:|
| Simple RAG | **60%** | 6/1/2 | 75% | 0% | 10% |
| MemoryBank-style | **60%** | 6/1/2 | 75% | 0% | 10% |
| MemGPT-style | 50% | 5/1/3 | 65% | 0% | 10% |
| Mem0-style | 50% | 5/1/3 | 65% | 0% | 10% |
| AM v0.2 (Hardened) | 40% | 4/1/4 | 55% | 0% | 10% |
| AM v0.2 (素) | 10% | 1/0/9 | 10% | **100%** | **0%** |

## 5. コンポーネント昇段ラダー (conv-26, 32問)

| 段階 | 精度 | 条 |
|---|---:|---|
| baseline (state無し) | 18.8% | ▇▇▇▇ |
| 旧retrieved state | 21.9% | ▇▇▇▇▇ |
| State Compiler (C5/C6) | 43.8% | ▇▇▇▇▇▇▇▇▇▇▇ |
| + Phase Vitamin | 59.4% | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |
| Gold state (上限) | 68.8% | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |
| + Phase Immune | **81.3%** | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |

## 6. Protein 昇段 (P0→P4)

| 段階 | 精度 | Evidence Recall | tokens |
|---|---:|---:|---:|
| P0 Steroid Baseline | 37.5% | 60.0% | 149 |
| P1 +Deduplication | 47.5% | 70.0% | 2,526 |
| P2 +Entity/Relation Fusion | 47.5% | 70.0% | 2,526 |
| P3 +Temporal Supersession | 50.0% | 67.5% | 1,949 |
| P4 +Evidence Ranker | **62.5%** | **85.0%** | 1,962 |

## 7. Overdrive Ablation (抜くと落ちるか)

| 構成 | overall | temporal | adversarial |
|---|---:|---:|---:|
| Full Overdrive Core | **72.5%** | 90% | 70% |
| − Proposition Gate | 65.0% (−7.5) | 90% | 20% |
| − Answer Verifier | 65.0% (−7.5) | 90% | 40% |
| − Temporal Engine | 55.0% (−17.5) | 20% | 70% |
| − Adaptive Search | 72.5% (±0) | 90% | 70% |
| − State Supersession | 72.5% (±0) | 90% | 70% |

## 8. Steroid / 検索再現率カーブ (495問)

| 段階 | Recall | 条 (vs baseline +) |
|---|---:|---|
| 冻結baseline | 57.2% | ▇▇▇▇▇▇▇▇▇▇▇▇ |
| R0 Wide Slicing | 76.2% (+19.0) | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |
| R1 1-Hop | 80.2% (+23.0) | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |
| R2 2-Hop | 83.2% (+26.1) | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |
| R3 3-Hop | **85.1%** (+27.9) | ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ |

Oracle gap 診断 (197問): Oracle回答 60.9% vs AM 37.5% → **検索ギャップ 23.4pp**

## 9. 検索オートプシー (失敗1,423件の原因)

| 原因 | 件数 | 割合 | 条 |
|---|---:|---:|---|
| LEXICAL_MISMATCH | 346 | 24.3% | ▇▇▇▇▇ |
| HAYSTACK_SESSION_DROP | 346 | 24.3% | ▇▇▇▇▇ |
| ENTITY_ALIAS_MISMATCH | 269 | 18.9% | ▇▇▇▇ |
| MULTI_HOP_GAP | 235 | 16.5% | ▇▇▇▇ |
| CANDIDATE_WINDOW_CUTOFF | 121 | 8.5% | ▇▇ |
| TEMPORAL_ANCHOR_MISMATCH | 106 | 7.4% | ▇▇ |

## 10. LLM別 (20問, seed=42)

| モデル | No Memory | Full Context | AM Recall |
|---|---:|---:|---:|
| qwen2.5:1.5b | 0.0% (222ms) | 90.0% (148ms) | **100% (142ms)** |
| qwen3:4b | 5.0% (3,558ms) | 100% (3,742ms) | **100% (3,734ms)** |

## 11. Top-k スイープ (P4, 40問)

| k | 4 | 8 | 10 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| 精度 | 32.5% | 40.0% | **47.5%** | 45.0% | 45.0% | 45.0% |
| tokens | 668 | 1,338 | 1,624 | 1,932 | 2,494 | 3,045 |

→ Top-10 が精度/コストの肩。

---
### 主要インサイト
1. LongMemEval の ceiling は Oracle 93% に対し 64% — ボトルネックは検索ではなく **変換 (ABSTENTION 62 + COMPOSITION 53 = 64%)**。
2. LoCoMo adversarial は +4.3pp 改善する一方、single-hop は −2.9pp — 過剰絞り込み/ranker のトレードオフ。
3. Overdrive ablation では **Temporal Engine が最大の寄与 (−17.5pp)**。Adaptive Search / State Supersession は現状ゼロ寄与。
4. 検索失敗の約半分は LEXICAL_MISMATCH + HAYSTACK_SESSION_DROP — 階層エビデンス索引 (hierarchical evidence index) の主攻目標と一致。
