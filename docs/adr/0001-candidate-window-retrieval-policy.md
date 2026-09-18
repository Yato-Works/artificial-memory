# ADR-0001: Candidate Window is a Retrieval Policy

- Status: Accepted
- Date: 2026-09-18
- Phase: 8.5/8.6 (DeepSeek Raid follow-up)
- Evidence: `benchmark/results/deepseek_ab/` (see Report §11, §13)

## Context

The frozen AM v0.2.0 recall pipeline narrows the search space at the store
entrance: `BasicRecallEngine._get_candidates` reads a small, unranked,
recency-ordered slice of the store (the "candidate window") and hands it to
semantic ranking. Nothing in the pipeline chose this window deliberately; it
is an accident of `store.get_memories` pagination.

Phase 8.5 quantified the consequence with an LLM-free ground-truth funnel
(store → candidates → gate → selected). At the frozen window the candidate
pool contains **2.6%** of the ground-truth evidence. Every downstream stage —
semantic ranking, budget selection, the CSA2 plan cache, the HSI gate —
operates on that 2.6%. The live 4-arm run confirmed the ceiling: no arm could
exceed `answer_accuracy = 0.1` until the window was widened, and the widened
arm jumped to **0.5** (`gate_wide`, gt coverage 0.1 → 0.6).

The window × gate grid (Phase 8.6) then separated the two bottlenecks:

| window | pool | evidence recall (pool) | evidence recall (gate) | gate out/q | reduction |
|-----|-----|-----|-----|-----|-----|
| frozen (~20) | any | **2.6%** | ≤ pool | 10–20 | 0–50% |
| 440 (full) | 100 | **100%** | 29.6% | 100 | 77% |
| 440 (full) | 64 | 100% | 25.2% | 64 | 85% |
| 300 | 20 | 79.1% | 13.5% | 20 | 93% |

Two independent defects, in series:

1. **Window**: an unranked recency slice discards ~97% of evidence before any
   ranked stage runs.
2. **Gate**: once the window is open (pool recall 100%), the cheap Jaccard
   lexical gate retains only 25–30% of evidence at operationally useful pool
   sizes (64–100). It is cheap, but its precision is the binding constraint.

## Decision

1. **`candidate_window` is promoted from an internal accident to an explicit
   Retrieval Policy parameter** of the recall pipeline — on the same level as
   `retrieval_strategy`. Policies are expressed as composable dimensions:
   - `candidate_window`: how much of the store is eligible (recent-first,
     full-scan, time-bounded, project-scoped, …)
   - `retrieval_strategy`: how the eligible space is narrowed (classic,
     deepseek cache, deepseek cache+gate)
2. **The window must be measured, never assumed.** Every change to the
   retrieval pipeline must pass the LLM-free GT-funnel audit
   (`scripts/deepseek_ab_small.py --gate-audit / --window-sweep /
   --grid-sweep`) before any LLM-cost experiment is spent on it.
3. **Default behaviour stays frozen.** `classic` keeps the historical
   recency-slice behaviour byte-identical. The policy dimensions are opt-in
   via `retrieval_strategy="deepseek"`, exactly as the CSA2 cache was.
4. **The gate is a candidate for replacement, not just retuning.** With the
   window open, pool-level recall is 100% and the gate's 70–75% damage is the
   sole remaining recall bottleneck. The next experiment must either (a) find
   a cheap scorer with higher retention at equal reduction, or (b) accept the
   reduction/retention trade explicitly with a documented operating point.

## Consequences

- **Positive**: recall becomes a first-class, tunable, measurable property.
  The 3,000-call A/B can now compare arms that differ by one policy dimension
  each, instead of all being capped at the same 2.6% ceiling.
- **Cost**: a widened window passes more candidates to ranking and (without a
  gate) to the context budget — the `gate_wide` live arm measured ~1.7× p50
  and 2.8× context tokens vs classic. The operating point must be chosen on
  the Pareto frontier (recall ↑, out/q ↓), not on recall alone.
- **Risk**: the current best frontier point (window 440, pool 100) retains
  only ~30% of evidence. Answer-level accuracy at that point (0.5) shows the
  *direction* is right, but the frontier's ceiling is set by the gate, not by
  the window. Do not freeze a production default until the gate's retention
  is improved or its cost accepted deliberately.

## Non-goals

- Replacing semantic ranking or the budget selector.
- Runtime integration of `EphemeralStore` (separate Replay-vs-Persist experiment).
- MoE-style routing, speculative retrieval.
