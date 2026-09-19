# ADR-0003: Abstention Must Be Structural, and a Lexical Floor Cannot Manufacture It

- Status: Accepted
- Date: 2026-09-19
- Phase: 8.11/8.12 (DeepSeek Raid follow-up)
- Evidence: `benchmark/results/deepseek_ab/abstention_audit_20260919_170618.json`
  (dataset `v0.2.0-arena-2`: evidence-absence check + floor dose-response),
  `benchmark/results/deepseek_ab/floor_validation_20260919_153600.json`
  (dataset `v0.2.0-arena-1`, pre-repair record of the same sweep: the curve
  matches the repaired one at floors 3.0/5.0 (GT 0.4872/0.3077) and differs
  only in the last digits, because the repair rewrote 34 distractor turns and
  therefore shifted the GT-term evidence sets)
- Amends: the non-goal "Abstention ... remains unsolved" in
  [ADR-0002](0002-session-granular-cheap-recall.md)

## Context

Phase 8.9's 3,000-call production run measured the abstention category
collapsing from 0.95 (classic) to 0.00 for the session-gated arms. The
explanation on the table was *fabrication*: the gate always fills its pool, so
an unevidenced question still receives ~100 memories of filler which the
answer LLM pattern-matches into a confident wrong value. Phase 8.11 therefore
added `SessionGate.score_floor` -- a per-memory BM25 floor, not backfilled --
so an unevidenced pool can come back empty.

Three defects were found while validating that lever, and only the third is
about the lever itself:

1. **`SessionGate.filter` short-circuited before the floor.** The
   `len(memories) <= pool_size` early return fired whenever the candidate
   window was small (the level-0 window is 50), so `floor` was never
   evaluated. Detected because floor=8 produced `floor_drops=0`; sixty
   consecutive validation runs had been evidence-free no-ops.
2. **Swapping `engine.gate` did not invalidate the CSA2 plan cache.** Plans
   seeded under floor=0.0 kept replaying under floor=8.0
   (`floor_drops=0` persisted after defect 1 was fixed). The gate signature is
   now part of the plan's provenance; `DeepSeekRecallEngine.recall` flushes the
   cache when it changes.
3. **The abstention questions were not unevidenced at all.** The S4 scenario
   says "For project atlas, the user only discussed scheduling and never a
   debugger", while S6 says "For project atlas, the user set the debugger to
   tokei". The abstention plot planted its near-miss fact on
   `PROJECTS[index + 7]`; with 20 questions and 20 projects that shift is
   surjective over `PROJECTS`, so **every** queried project eventually received
   a debugger from some *other* question's plot.

Defect 3 invalidates the fabrication reading of Phase 8.9. `gate_wide`
surfacing "tokei" for "atlas" was not hallucination: the assertion existed in
the history, planted after the abstention plot's denial by an unrelated plot.
The scorer, seeing the higher-session evidence, marked the answer
`false_positive` and booked it against the gate.

## Decision

1. **Evidence-absence is a dataset invariant, not an emergent property.**
   Abstention distractors are drawn from `PROJECTS_UNQUERIED`, a 20-entity
   pool disjoint from `PROJECTS`. The near-miss ("some project does have a
   debugger") is preserved; a queried project receiving evidence becomes
   impossible by construction rather than by luck of the index shift.
   `ARENA_VERSION` moves `v0.2.0-arena-1` -> `v0.2.0-arena-2` and the frozen
   hash is regenerated (`eed68804...`).
2. **The invariant is tested, not trusted.**
   `tests/test_v02_phase7.py` asserts pool disjointness *and* greps the
   generated history for `"for project <P>, the user set the debugger to"` for
   every abstention question (pre-repair: 7 of 20 offenders; post-repair: 0).
3. **`score_floor` stays, as a cost lever, not as the abstention fix.** The
   measured dose-response (20 abstention + 9 GT questions, pool=100,
   window=450) is monotone and has a usable knee:

   | floor | abstention context | empty | GT retention |
   |---|---|---|---|
   | 0.0 | 83.5 | 0% | 0.5128 |
   | 0.5-1.0 | 78.0 | 0% | 0.5128 |
   | 2.0 | 78.0 | 0% | **0.5385** |
   | 3.0 | 37.7 | 0% | 0.4872 |
   | 4.0 | 2.2 | 0% | 0.3333 |
   | 6.0 | 0.1 | 95% | 0.2564 |
   | 8.0 | 0.0 | 100% | 0.1538 |

   floor in [0.5, 2] trims filler at zero measured retention cost (2.0 is
   retention-neutral-to-positive), while floor >= 4 trades away real GT
   evidence. Default stays `0.0` (Phase 8.7/8.9 behaviour).
4. **A lexical floor is not an abstention mechanism.** Emptying the context
   requires floor=8, by which point GT retention is 0.1538. Abstention and
   evidence questions share the same lexical profile ("for project X ...
   debugger"), so no query-independent lexical threshold can separate them.
   Abstention must be decided from evidence *presence* (the denial memories
   exist and are distinguishable), not from evidence *strength*.

## Consequences

- **Positive**: the Phase 8.9 abstention delta is now correctly attributed --
  partly a dataset defect, not purely a gate defect. The repaired dataset makes
  the category actually measurable, which is a precondition for any claim
  about fabrication.
- **Re-run required**: every Phase 8.9/8.10/8.11 number computed on
  `arena-1` abstention questions is invalid *for that category only*. The
  retrieval-side findings (window, session granularity, cost curve) are
  unaffected -- they never depended on abstention.
- **Interpretation rule**: `No Evidence != Evidence Not Found`. A system that
  answers an abstention question can now only do so from the denial statement
  or from parametric knowledge; both are legitimate measured failures of
  abstention, and neither is a retrieval artefact.
- **Temporal dimension**: the atlas case is a *denial at S5, assertion at S6*
  pair -- exactly the structure the `knowledge_updates` and
  `contradiction_handling` categories test. Abstention and temporal recall
  therefore share a representation requirement (state as of time), and are
  best repaired together rather than as an "abstention-only" patch.

## Non-goals

- Changing the abstention scorer (`false_positive` on any answer remains
  correct).
- Making the answer LLM abstain by prompt engineering; the measured question
  is whether the retrieval layer can present absence honestly.
- Re-tuning the floor per category (the sweep shows no category-separating
  threshold exists).