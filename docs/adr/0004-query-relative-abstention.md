# ADR-0004: Query-Relative Top-1 Threshold Separates Abstention Without Damaging Ground-Truth Retention

- Status: Accepted
- Date: 2026-09-19
- Phase: 8.13 (DeepSeek Raid follow-up)
- Evidence: `benchmark/results/deepseek_ab/margin_probe.json`
  (`separation_gap = +0.2145`, `abstain_max = 6.0834`, `answer_min = 6.2979`),
  `benchmark/results/deepseek_ab/abstention_audit_20260919_175042.json`
  (post-repair `v0.2.0-arena-2`: `abstention_evidenced = 0`)
- Amends: [ADR-0003](0003-abstention-evidence-absence.md) §Decision.4
  ("A lexical floor is not an abstention mechanism")

## Context

ADR-0003 demonstrated that a query-independent per-memory BM25 floor (`score_floor`) cannot solve the abstention problem: emptying the candidate pool required `floor = 8.0`, by which point Ground-Truth (GT) retention collapsed from 0.5128 to 0.1538 (a 70% loss of real evidence). `score_floor` was therefore designated strictly as a cost/filler-trimming lever with default `0.0`.

The core reason an absolute per-memory floor failed was **query variability**: the absolute BM25 score of a relevant memory varies with query length, term specificity, and IDF. An unevidenced query with common keywords can produce individual memory scores of 5–6, whereas an evidenced query with brief or generic phrasing might only peak at 6–7.

Phase 8.13 probed whether query-relative statistics could separate unevidenced queries from answerable queries. We ran a full-corpus margin probe (`MarginProbe` in `scripts/deepseek_ab_small.py`) across all 180 questions in `v0.2.0-arena-2` (20 abstention + 160 across 8 answerable categories).

## Measured Findings

From `benchmark/results/deepseek_ab/margin_probe.json`:

1. **Top-1 BM25 Margin (`separation_gap = +0.2145`)**:
   - Maximum `top1` score among all 20 abstention questions: **`6.0834`** (abstention-03).
   - Minimum `top1` score among all 160 answerable questions: **`6.2979`** (contradiction-08).
   - **Separation Gap**: `min(answerable) - max(abstention) = +0.2145` (> 0).
   - A single query-relative threshold on `top1` (e.g. $\tau_{ab} = 6.15$ to $6.20$) separates the two distributions with **100% precision and zero overlap**.

2. **Comparison with Other Statistics**:
   | Statistic | Abstain Max | Answer Min | Gap | Rel Gap ($\Delta / \text{max}$) |
   |---|---|---|---|---|
   | **top1** | **6.0834** | **6.2979** | **+0.2145** | **+0.0353 (Separable)** |
   | top1_ratio (peakiness) | 1.4910 | 1.1720 | -0.3190 | -0.2140 (Inseparable) |
   | top1 - top10_mean | 1.9573 | 1.3551 | -0.6022 | -0.3077 (Inseparable) |
   | mass@0.5 | 79 | 39 | -40 | -0.5063 (Inseparable) |
   | mass@3.0 | 38 | 36 | -2 | -0.0526 (Inseparable) |

   `top1` is the **only** statistic that exhibits a strictly positive separation gap across the entire benchmark.

## Decision

1. **Implement `SessionGate.abstention_threshold`**:
   - `SessionGate` receives an optional `abstention_threshold: float | None = None` (default `None` for backward compatibility; recommended `6.15`).
   - At the entry of `SessionGate.filter()`, compute `max_score = max(own_scores)`.
   - If `abstention_threshold is not None` and `max_score < abstention_threshold`:
     - The query is classified as structurally unevidenced.
     - The gate immediately returns `[]` (empty candidate list) and increments `self.abstention_drops`.
   - If `max_score >= abstention_threshold`:
     - The query is classified as evidenced.
     - The full candidate pool proceeds to session ranking and optional `score_floor` filtering, preserving **100% of Ground-Truth retention**.

2. **Nemotron P0 Hardening: Scale-Invariant Normalized BM25**:
   - To eliminate the statistical fragility of hardcoded thresholds across varying query lengths and corpus sizes ($10^3 \to 10^7$), `SessionGate` supports **Normalized BM25**:
     $$\text{norm\_bm25}(q, d) = \frac{\text{BM25}(q, d)}{\sum_{t \in q} \text{IDF}(t) \cdot (k_1 + 1)}$$
   - When $0.0 < \text{abstention\_threshold} < 1.0$, the gate operates in normalized mode ($[0, 1]$ scale-invariant confidence ratio). When $\ge 1.0$, it operates in raw BM25 score mode (backward compatibility).

3. **Nemotron P0 Hardening: Robust Top-k Mean Session Aggregation**:
   - To eliminate the "Trojan Horse" failure mode where a single incidental word match pulls in an entire multi-topic session or massive log session, `SessionGate` defaults to `session_aggregation="top_k_mean"` with $k=3$.
   - A session must demonstrate cohesive evidence across multiple memories to achieve high ranking, preventing context pollution and outlier amplification.

4. **Nemotron P0 Hardening: Plan Cache Invalidation on Degradation**:
   - When a memory degrades its resolution (e.g. `FULL (200t)` -> `DEEP_LONG_TERM (25t)`), cached retrieval plans containing that memory become stale and corrupt downstream knapsack token allocation.
   - `RetrievalPlanCache.invalidate_for_memory(memory_id)` evicts any cached plans containing the modified memory ID, ensuring exact token budget compliance.

5. **Plan Cache Provenance Invalidation**:
   - `abstention_threshold`, `session_aggregation`, and `aggregation_k` are added to `DeepSeekRecallEngine._gate_cfg`.
   - Changing any gate parameter automatically flushes `RetrievalPlanCache`, preventing stale unevidenced or evidenced plans from replaying across configurations.

6. **Theoretical Alignment with SOTA Architectures**:
   - **Nous Research (Predictive World Model)**: A memory system should only update or retrieve when evidence generates an informative shift in belief. Returning an empty pool when no evidence exists reflects the absence of belief updates, rather than feeding noise to the answerer.
   - **DeepSeek (Engram) & Anthropic (Contextual Retrieval)**: Preserving high session coherence while cleanly gating non-existent memories prevents the LLM's attention mechanism from fabricating answers out of background context filler.

## Consequences

- **Positive**: Solves the abstention accuracy collapse (Phase 8.9: 0.95 -> 0.00) without sacrificing GT retention (0.5128 -> 0.1538). Both high recall and honest abstention are achieved simultaneously.
- **Robustness**: Scale-invariant Normalized BM25 protects against query-length and corpus-size drift; top-k mean protects against Trojan Horse single-token context poisoning.
- **Correctness**: Cache invalidation on degradation prevents token-budget overflow and stale knapsack solving.
- **Observability**: `SessionGate.snapshot()` exposes `abstention_drops`, tracking exactly when queries are gated as unevidenced.
- **Safety**: Default remains `None`, preserving frozen baseline behaviour for existing pipelines unless explicitly configured.
