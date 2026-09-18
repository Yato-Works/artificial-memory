# ADR-0002: The Cheap Recall Layer Must Exploit Session Structure, Not Memory Similarity

- Status: Accepted
- Date: 2026-09-19
- Phase: 8.7/8.8 (DeepSeek Raid follow-up)
- Evidence: `benchmark/results/deepseek_ab/scorer_sweep_*.json`, `summary_20260919_022054.json`, `summary_20260919_023013.json`
- Supersedes: the open gate question in [ADR-0001](0001-candidate-window-retrieval-policy.md) (Decision #4)

## Context

ADR-0001 left the gate as the sole remaining recall bottleneck: with the
candidate window fully open (pool recall 100%), the cheap Jaccard gate
retained only 25–30% of ground-truth evidence at operationally useful pool
sizes. Its Decision #4 required either (a) a cheap scorer with higher
retention at equal reduction, or (b) an explicit, documented trade.

Phase 8.7 ran the scorer sweep to settle it: five cheap scorers (Jaccard,
BM25, embedding-cosine, hybrid, and a session-max BM25) over windows
{50, 200, 440} × pools {10, 20, 32, 64, 100}. The result was not a tuning
outcome but a category error in how the gate scored candidates:

| scorer | unit of scoring | best retention @ pool 100 |
|---|---|---|
| jaccard | single memory vs query | ~30% |
| bm25 | single memory vs query | ~30% |
| embedding | single memory vs query | 14–16% (inverted: semantic clustering evicts evidence) |
| hybrid | single memory vs query | ~31% |
| **session** | **conversation session containing the memory** | **85%** |

Every memory-level similarity scorer destroyed 62–86% of evidence regardless
of lexical or semantic flavour. The session-granular scorer — score each
memory by the best-matching *session* it belongs to — retained **85%** at the
same 77% pool reduction, and was ~4× faster than the embedding scorer.

Phase 8.8 then validated the full pipeline end-to-end with the real frozen
LLM (10-question E2E subset, 3 answer repeats):

| metric | classic | session (`deepseek_session`) |
|---|---|---|
| answer accuracy | 0.0 | **0.6** |
| mean score | 0.0 | 0.65 |
| GT coverage | 0.1 | **0.75** |
| repeat selection stability | 0.0 | **1.0** |
| gate traffic per query | — | 440 → 100 (77.3% reduction) |
| plan tiering | — | full=10, reuse=40 (80% cache hit) |

## Decision

1. **Session granularity is the unit of the Cheap Recall Layer.** The gate
   scores candidates by session-level structure (the best-matching
   conversation session), not by per-memory similarity to the query.
   Implemented as `SessionGate` (`recall/retrieval_cache.py`), selected via
   `retrieval_strategy="deepseek_session"`.
2. **Memory-level similarity scorers are rejected for gating.** The sweep
   showed they are not weak-but-tunable; they are structurally wrong for this
   pipeline (embedding scoring is actively anti-correlated). They remain
   available behind `--scorer-sweep` for reproduction.
3. **`deepseek_session` is a distinct opt-in strategy**, compositional with
   the existing dimensions: classic (frozen default) → `deepseek` (CSA2 cache
   ± gate) → `deepseek_session` (cache + full-store window + session gate).
   `BasicRecallEngine` remains untouched.
4. **The window-and-gate operating point is explicit**: full-store window
   (`candidate_window=None`), gate pool 100. This is recorded as a policy
   pair, per ADR-0001 Decision #1, not as a hidden default.

## Consequences

- **Positive**: the 8.5→8.8 evidence chain closes. Answer accuracy moved
  0.0 → 0.6, repeat stability 0.0 → 1.0, at a 77.3% pool reduction. The
  retrieval bottleneck identified in ADR-0001 is resolved end-to-end.
- **Note on stability**: `repeat_selection_stability` is implemented as the
  fraction of adjacent answer-repeat pairs with *identical* memory-id
  sequences (exact-match, not a Jaccard-overlap formula). Any note or report
  should quote the implemented definition, not the Jaccard form.
- **Cost**: the session gate reads the full store per query (440 memories
  here). At 77% reduction and p50 ≈ 50 ms this is acceptable at benchmark
  scale; a store-orders-of-magnitude-larger deployment needs the session
  index bounded (e.g. recency- or project-scoped sessions) before reuse.
- **Risk**: the 10-question E2E is a small sample. The 3,000-call production
  run must reproduce the delta before this operating point is frozen as a
  recommended configuration.

## Non-goals

- Abstention / temporal / contradiction categories remain unsolved; they are
  a *reasoning*-layer problem (model discrimination), now cleanly separated
  from the retrieval layer this ADR covers ("retrieval solved ≠ reasoning
  solved").
- Replacing semantic ranking, the budget selector, or the CSA2 cache.
- `EphemeralStore` runtime integration (separate Replay-vs-Persist experiment).
