# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **v0.2.0 Phase 8.12 - Abstention dataset repair (arena-2) + abstention semantics audit**
  - **Dataset defect (root cause of the Phase 8.9 abstention collapse)**:
    `_plot_abstention` planted its near-miss distractor on
    `PROJECTS[index + 7]`.  With 20 questions and 20 projects that shift is
    surjective over `PROJECTS`, so *every* queried project eventually received
    a debugger from some other question's plot -- 7 of 20 abstention questions
    had real evidence in the history (e.g. the `nectar` question planted
    "atlas ... set the debugger to tokei", directly contradicting S5's "never
    a debugger").  Phase 8.9's `gate_wide` did not fabricate "tokei"; it
    surfaced a planted assertion and the scorer booked it as
    `false_positive`
  - **Repair**: new `PROJECTS_UNQUERIED` pool (20 entities, disjoint from
    `PROJECTS`) supplies abstention distractors, making evidence-absence a
    structural guarantee.  `ARENA_VERSION` -> `v0.2.0-arena-2`, frozen hash
    regenerated (`eed68804...`)
  - **Invariant tests** (`tests/test_v02_phase7.py`): pool disjointness, plus
    a history grep asserting no turn says "for project <P>, the user set the
    debugger to" for any abstention project (pre-repair 7 offenders ->
    post-repair 0)
  - **`--abstention-audit`** (`scripts/deepseek_ab_small.py`): Part 1 counts
    asserting vs denying memories per abstention question (post-repair:
    `abstention_evidenced=0`, `structurally_unwinnable=true`); Part 2 sweeps
    `score_floor` for the dose-response
  - **Floor dose-response** (20 abstention + 9 GT questions, pool=100,
    window=450): floor 0.0 -> ctx 83.5 / GT 0.5128; floor 2.0 -> 78.0 /
    **0.5385** (retention-neutral-to-positive, the knee); floor 3.0 -> 37.7 /
    0.4872; floor 4.0 -> 2.2 / 0.3333; floor 6.0 -> 0.1 (95% empty) / 0.2564;
    floor 8.0 -> 0.0 (100% empty) / 0.1538
  - **Finding**: a query-independent *lexical* floor cannot implement
    abstention -- abstention and evidence questions share the same lexical
    profile, so emptying the context costs 85% of GT retention.  Abstention
    must come from evidence *presence* (denial memories), not evidence
    *strength*.  `score_floor` is retained as a cost lever only (default 0.0)
  - Evidence: `abstention_audit_20260919_170618.json`,
    [ADR-0003](docs/adr/0003-abstention-evidence-absence.md)

### Added
- **v0.2.0 Phase 8.11 - Score-floor abstention lever + three-layer forensic audit**
  - `SessionGate.score_floor`: memories whose own BM25 is below the floor are
    excluded individually (not backfilled), so the gate may return *fewer*
    than pool_size memories; an empty context is the correct answer to a
    question the store cannot evidence
  - Dose-response (LLM-free, 20 abstention + 9 GT questions, pool=100):
    floor=0.0 abstention_ctx=84.8 / gt_retention=0.513; floor=1-2 78.0 /
    0.539 (retention unchanged); floor=3 38.1 / 0.487; floor=8 0.0 (100%
    empty) / 0.128.  Usable window: floor in [1, 2] trims filler with zero
    retention cost
  - **Fix (silent no-op #1)**: `SessionGate.filter` returned early when
    `len(memories) <= pool_size`, which disabled the floor entirely whenever
    the candidate window was small (measured: floor=8 produced
    `floor_drops=0`, 60 consecutive validation runs were evidence-free
    no-ops).  Floor evaluation now runs BEFORE the size short-circuit
  - **Fix (silent no-op #2)**: swapping `engine.gate` did not invalidate the
    CSA2 plan cache -- plans seeded under floor=0.0 kept replaying under
    floor=8.0 (`DeepSeekRecallEngine.recall` now flushes the plan cache when
    the gate type/pool/max_sessions/score_floor signature changes)
  - **Fix (validation harness)**: a per-floor `SessionGate` built without
    `corpus_provider` computes IDF over the *pool*, not the store -- a
    different scorer from the shipped one; harnesses must wire
    `corpus_provider=runtime._store_snapshot` to measure production
    behaviour
  - **Dataset finding (misattribution)**: the S4 abstention questions are
    *not* unevidenced -- S6 contains "For project <P>, the user set the
    debugger to <X>" for all 20 projects, directly contradicting S5's
    "only discussed scheduling and never a debugger".  The Phase 8.9
    abstention accuracy collapse (0.95 -> 0.00) was therefore partly
    misattributed: gate_wide surfaced the S6 answer and the scorer marked
    answering as `false_positive`.  The floor lever is mechanically
    correct (dose-response above) but the abstention dataset must be
    repaired before it can validate the fabrication hypothesis
  - Evidence: `floor_validation_20260919_153600.json` (valid run; the 60
    earlier no-op evidence files were deleted, see comment in
    `scripts/deepseek_ab_small.py` history)
### Added
- **v0.2.0 Phase 8.10 - Session-granular cheap-recall cost curve**
  - `SessionGate.max_sessions`: top-K whole sessions retained, pool-size cap
    removed (sessions kept whole; deterministic tie-break on first-index)
  - LLM-free cost audit: 95.2% cost reduction (pool=10) / 43.7% retention at
    pool=100 / 43.7% retention, 1394 mean tokens at pool=60 (max_sessions=1
    gives 95.6% of the accuracy at 4.6% of tokens)
  - `scripts/deepseek_ab_small.py`: `--cost-audit` (pool x K grid),
    `--gate-audit` (funnel), `--window-sweep`
  - Finding: the pool=60 "free win" was **refuted** under stratified
    sampling -- the winning run's question set was easy-category only
    (`factual_recall` + `multi_session_recall`); on a stratified 40-question
    set pool=60 scores 0.50 / 0.613 (accuracy / GT coverage) vs 0.55 / 0.725
    at pool=100. The `max_sessions` lever is inactive at pool=100 (the
    pool_size cap binds first); real leverage is at pool <= 60 OR K=1
  - Evidence: `benchmark/results/deepseek_ab/session_cost_audit_*.json`,
    `production_20260919_093444.json` (pool=60), `production_20260919_095444.json`
    (pool=100), `production_20260919_095745.json` (both, same stratified set)
- **v0.2.0 Phase 8.9 - Production A/B validation (3,000-call real-LLM run)**
  - `scripts/deepseek_ab_production.py`: chunked three-arm production
    experiment -- 200 questions x 5 repeats x 3 arms (`classic` / `cache` /
    `session`) = 3,000 answer calls in 30 crash-isolated chunks (evidence
    flushed per chunk, `failures=0` across all 30)
  - Medium-scale confirmation (40 questions, 600 calls): session-gate
    accuracy 1.0 vs classic 0.0, GT coverage 1.0, repeat stability 1.0,
    plan reuse 960/1000 (96%); cost: p50 93 ms vs 32 ms, tokens 1,991 vs
    511 (the correctness-for-cost trade-off that motivates Phase 8.10)
  - Budget audit: every one of the 1,800 answer contexts <= 2,000 tokens
    (0 violations); `violation=None` clarified as "key absent from
    summary", re-verified from raw traces
  - Window/scorer sweep evidence retained under
    `benchmark/results/deepseek_ab/` (see ADR-0002 for the full chain)
- **v0.2.0 Phase 8.8 - Session-granular gate: runtime wiring + real-LLM validation**
  - `SessionGate` in `recall/retrieval_cache.py`: session-level BM25-style
    scoring where a session's *maximum* evidence score is assigned to every
    memory in it -- session membership is the recall-bearing structure, not
    per-memory lexical similarity
  - `RuntimeConfig.retrieval_strategy="deepseek_session"`: session gate +
    widened candidate window wired into the runtime (opt-in, additive);
    `DeepSeekRecallEngine` gained a configurable candidate window
    (`candidate_window`, default = frozen recent-50)
  - 10-question real-LLM validation: accuracy 0.0 -> 0.6, GT coverage
    0.1 -> 0.75, repeat stability -> 1.0 at 440 -> 100 candidate reduction
- **v0.2.0 Phase 8.6/8.7 - Bottleneck forensics + cheap-scorer sweep**
  - GT-funnel audit (`--gate-audit`): store -> candidates -> gate ->
    selected attribution; finding: the frozen candidate window (recent-50
    of 440 memories) loses 89% of ground-truth evidence *before* the gate
    runs -- the original Jaccard gate was innocent (fired 0 times)
  - `--window-sweep`: candidate-window decision table (50/100/250/450)
  - Cheap-scorer sweep (Jaccard / BM25 / embedding / hybrid / session):
    memory-level scorers keep destroying 62-86% of evidence at useful pool
    sizes; only session-granular scoring achieves non-destructive reduction
    (the "right unit of retrieval was not a Memory" finding, ADR-0002)

### Fixed
- `memory/vector_search.py`: bounded retry (50-400 ms exponential backoff)
  around the atomic `os.replace` in `_save_index`; on Windows, antivirus
  scanners briefly hold a shared-read lock on freshly written
  `.tmp`/FAISS index files, making `os.replace` fail with WinError 5 even
  though permissions are correct.  This flaked 5 `test_phase8_arena`
  tests intermittently (all passed in isolation); with the retry the full
  suite is stable: 435 passed, 11 skipped, 0 failed.  No semantic change.

- **v0.2.0 Phase 8.4 - DeepSeek Raid (DeepSeek V4.1-inspired retrieval acceleration)**
  - `recall/retrieval_cache.py`: three DeepSeek V4.1 systems patterns
    translated to memory-runtime mechanisms:
    - `RetrievalPlanCache` (CSA2 analogue): FULL / REINDEX / REUSE
      candidate-set reuse for repeated query families, with deterministic
      punctuation-stripping signatures, insertion-order eviction and TTL
    - `HierarchicalGate` (HSI analogue): cheap lexical Jaccard pre-filter
      narrowing the candidate pool before semantic ranking; order-stable
      tie-breaks keep recorded runs replayable
    - `EphemeralStore` (SWA Bounded Replay analogue): bounded session
      scratch state that is discarded, never persisted, and rebuilt from
      the source log through a replay closure
  - `recall/deepseek_engine.py`: `DeepSeekRecallEngine` composing the plan
    cache + gate on top of the frozen `BasicRecallEngine` (subclass, never
    in-place modification)
  - `RuntimeConfig.retrieval_strategy`: opt-in switch (`"classic"` default =
    frozen behavior; `"deepseek"` wires the raid engine); S0-S3 baseline
    players and the frozen Scorer are untouched
  - `tests/test_deepseek_raid.py`: 23 tests covering tiering, gating,
    replay, composition and runtime wiring (full suite: 423 passed,
    11 skipped, zero regressions)
- **v0.2.0 Phase 8.5 - DeepSeek Raid A/B experiment harness**
  - `scripts/deepseek_ab_small.py`: three-arm controlled E2E over the
    10-question Phase 8.3.4 subset -- `classic` (frozen baseline) vs
    `cache` (CSA2 plan cache) vs `gate` (plan cache + HierarchicalGate);
    EphemeralStore stays out of every arm so deltas attribute cleanly
  - `AMv020Player`: `retrieval_strategy` config passthrough (default
    `"classic"` keeps frozen behaviour) and `memory_ids` in Answer.metadata
    for selection-level determinism evidence
  - Evidence per arm: frozen-Scorer aggregates, retrieval latency
    p50/p95, context tokens, FULL/REINDEX/REUSE counters, gate traffic,
    repeat-selection stability, cross-arm selection agreement
  - Dry-run validation findings: REUSE freezes selection across repeats
    (stability 1.0) where classic re-ranking drifts (0.0); gate is a no-op
    at pool_size=64 with 20-candidate pools (use a smaller pool to arm it)
  - First real-LLM run (phi4-mini, frozen config, gate-pool=10):
    - quality identical across arms (answer_accuracy 0.1 = abstention only;
      phi4-mini fails all knowledge questions under the 2,000 budget, so
      raid mechanisms are quality-neutral here -- no recall lost to REUSE)
    - cache arm: retrieval p50 32.1 ms vs classic 42.9 ms (~25% faster);
      plan tiering full=10 / reuse=240 (96% reuse)
    - gate arm: pool narrowed 200 -> 100 candidates, context tokens
      522 -> 262.6 (-50%), retrieval p50 25.8 ms (fastest); selection
      agreement vs classic drops to 0.0 (gate changes candidate pools by
      design), repeat-selection stability stays 1.0
    - cross-arm agreement confirms determinism: cache replays stable
      selections, classic re-ranks drift (stability 0.05)
    - evidence: benchmark/results/deepseek_ab/summary_20260918_161229.json

### Added
- **v0.2.0 Phase 1 - Memory Evolution Core**
  - Runtime memory state vector (`MemoryStateVector`) with `importance` /
    `confidence` / `currentness` / `future_utility` / `contradiction_risk`
    and the deterministic `StateSignalCalculator` behind it
    (`state_signals.py`)
  - New first-class lifecycle operations in `MemoryEvolutionEngine`:
    `REINFORCE` (usage/evidence raises confidence), `REINTERPRET` (new
    higher-order interpretation keeping the original evidence), `RESTORE`
    (rebuild the richest available resolution from stored versions),
    `KEEP` and `REJECT` (explicit retention decisions; reject means DORMANT,
    never deletion)
  - `evolution_policy.py`: proposal -> validation -> commit -> audit split.
    `EvolutionProposal` is validated by `EvolutionGovernor` (9 deterministic
    rules) before any mutation; AI-proposed mutations are proposals, not
    truth (plan #26)
- **v0.2.0 Phase 2 - Temporal Management & Contradiction Handling**
  - `temporal_validity.py`: explicit validity windows
    (`valid_from` / `valid_until`) with half-open interval semantics,
    monotonic invalidation and `apply_supersession_bounds` (plan #21)
  - `supersession.py`: `SupersessionManager` with explicit
    `SUPERSEDES` edges, bidirectional chain walking and current-head
    resolution; history is preserved, never overwritten
  - `contradiction_edges.py`: explicit `CONTRADICTS` edges with severity,
    audit events on both sides, and temporal resolution of genuine state
    changes (`apply_temporal_resolution`)
  - `lifecycle_transitions.py`: validated `MemoryStatus` state machine
    (`ALLOWED_TRANSITIONS`), `InvalidTransitionError`, and transition
    history read back from the audit log
  - `provenance.py`: evidence chain walking (Semantic -> Episode ->
    Conversation -> Message) and `explain()` combining validity,
    provenance, supersession, contradictions and audit history (plan #22)
  - Governor dispatch for `CONTRADICT` / `SUPERSEDE` / `TRANSITION`;
    Phase 1's `CONTRADICT` routing placeholder is now a real dispatch
- Tests: `tests/test_v02_phase1.py` (34 tests) and
  `tests/test_v02_phase2.py` (35 tests)
- **v0.2.0 Phase 3 - Reconstruction Layer**
  - `graph.py`: deterministic memory-graph traversal (plan #15 / #16).
    Breadth-first expansion with visited-set cycle guards, total ordering
    (strength, association id, memory id) and path/seed/strength bookkeeping;
    supporting relations (RELATED / ELABORATES / SUMMARIZES / CAUSES /
    FOLLOWS) are expandable while CONTRADICTS / SUPERSEDES are never
    implicitly expanded
  - `evidence.py`: LLM-free evidence ranking (plan #17) combining lexical
    query coverage (Latin words + CJK bigrams), graph proximity, the Phase 1
    state vector and temporal validity, with contradiction and non-current
    penalties; every item carries human-readable inclusion reasons;
    `keep_ids` pins caller-supplied seed chains into the package
  - `reconstruction.py`: the full pipeline "candidates -> graph expansion ->
    provenance expansion -> temporal filter -> ranking -> contradiction
    checks -> multi-memory synthesis -> evidence package" (plan #15),
    producing an auditable `ReconstructedEvidencePackage` with chains,
    conflicts (resolved vs unresolved), provenance traces and deterministic
    text synthesis; `reconstruct_multi_hop()` recovers known A -> B -> C
    chains
  - Fixed a SQL operator-precedence bug in
    `SQLiteMemoryStore.get_associations` / `PostgresMemoryStore`:
    the `association_type` filter silently leaked to only one side of the
    `OR`, making type-filtered queries return unrelated associations
- Tests: `tests/test_v02_phase3.py` (53 tests)
- **v0.2.0 Phase 4 - Context Allocator**
  - `context/allocator.py`: the context window as a constrained resource
    (plan #18). Every candidate receives one of five representations
    (`FULL / COMPRESSED / SUMMARY / TAG / OMIT`) chosen by priority, so the
    allocator optimizes *useful information per token* rather than raw
    context volume
  - Priority covers every plan #18 factor: query relevance, importance,
    confidence, temporal relevance, contradiction risk and resolution; token
    cost enters through the representation choice
  - Greedy budget allocation with a strict downgrade path (FULL ->
    COMPRESSED -> SUMMARY -> TAG); COMPRESSED / SUMMARY texts come from
    stored `MemoryVersion`s (never generated on the fly) and parts always
    report the *effective* level after degradation; TAGs are synthesized
    deterministically from the memory's own tokens plus its id
  - Every OMIT carries an auditable reason (below relevance floor /
    superseded / budget exhausted); `ContextAllocation` reports token usage,
    the representation distribution and utility-per-token stats
  - `allocate_from_reconstruction()` consumes a Phase 3 evidence package
    directly, closing the retrieval -> reconstruction -> allocation pipeline
  - Fixed an import cycle (`context` -> `memory` -> `compiler` ->
    `context`) by importing the memory modules lazily inside the allocator
- Tests: `tests/test_v02_phase4.py` (28 tests)
- **v0.2.0 Phase 5 - Background Reflection**
  - `memory/reflection.py`: memory management outside the critical response
    path (plan #13 / #14). The `BackgroundReflector` runs the full pipeline —
    candidate selection -> evidence retrieval -> review / state assessment ->
    proposed mutation -> policy validation -> commit -> audit
  - Seven triggers (`idle` / `session_end` / `low_load` / `memory_pressure`
    / `contradiction_detected` / `scheduled` / `unresolved_clusters`); review
    is bounded (`max_candidates`), prioritizing frequently-used, stale,
    low-utility and low-confidence memories
  - Five reflection workers, each producing an `EvolutionProposal` — never a
    direct mutation (plan #26): `REINFORCE` (frequently accessed + valuable),
    `MERGE` (duplicate clusters via lexical overlap), `REINTERPRET`
    (deterministic synthesis from related evidence), `ARCHIVE` (`TRANSITION`
    to archived — never deletion), `KEEP` (explicit retention for rarely
    accessed high-value memories)
  - All proposals flow through the Phase 1 `EvolutionGovernor` (the exact
    reason the gate was built first); commits write auditable evolution
    events with `triggered_by="reflection"`; `dry_run` mode reviews without
    committing; `ReflectionReport` is fully serializable
  - LLM reviewer hook: the heuristic proposer can later be replaced by an
    LLM submitting through the same proposal interface and validation gate
- Tests: `tests/test_v02_phase5.py` (16 tests)
- **v0.2.0 Phase 6 - Predictive Recall**
  - `recall/predictive.py`: `TopicPredictor` ranks the topics a conversation is
    likely to move to next from three deterministic, LLM-free signals
    (association links into the topic, topic recency, query/topic term
    overlap); a zero-total case predicts nothing rather than guessing
  - `PrefetchCache` + `PredictiveRecallEngine.prefetch()` warm the likely
    topics into a TTL- and size-bounded cache; `recall()` serves them only
    when the conversation actually moves there (cache hit -> `reused`,
    otherwise a normal recall) — deferred context injection, never eager
    injection of speculative context
  - Prefetched memories are chosen by the Phase 1 `StateSignalCalculator`
    (`future_utility`), so prediction cost and memory value share one model
  - `PrefetchResult` / `stats()` report fetched, reused, hit rate and
    evictions, keeping the cost/benefit of prediction measurable
  - Fixed an import cycle introduced by the new module (recall ->
    `memory.__init__` -> `memory.compiler` -> `compiler` -> `context` ->
    recall) by importing `memory.evidence` / `memory.state_signals` lazily,
    following the Phase 4 `allocator.py` cycle-guard convention
- Tests: `tests/test_v02_phase6.py` (19 tests)
- **v0.2.0 Phase 7 - Benchmark Infrastructure (Track B Arena)**
  - `research/benchmarks/arena.py`: deterministic, LLM-free Track B dataset —
    one shared 200-question conversation history (10 categories x 20
    questions) planted into 8+ session buckets, with question-level ground
    truth expressed as term groups ("must mention X and Y, not Z") so scoring
    needs no judge model (plan #33 / #34)
  - Categories map 1:1 to the capability axes: factual recall,
    multi-session recall, temporal reasoning, knowledge updates,
    contradiction handling, indirect recall, distractor resistance,
    abstention, compression recovery, reflection/reinterpretation
  - Session buckets are data-derived (`day // SESSION_INTERVAL_DAYS`): one
    scenario per bucket with unique ids, turns merged and day-ordered;
    cross-session follow-ups (`+15` / `+30`) may open trailing sessions
    beyond the base 8-session grid
  - `write_dataset` / `read_frozen_hash` / `verify_freeze`: hash freeze so
    results are only comparable against an unchanged dataset (Fairness Rule
    8); the frozen dataset is committed at `benchmark/dataset/arena.json`
    with its sha256 (`84dd96f224ea6fe8...`)
  - Dataset integrity fixes during implementation: early-session facts are
    clamped to day >= 0 (no `session--1` bucket), per-day scenarios no
    longer collide on one bucket id, and `required_sessions` is derived
    from the real distinct evidence sessions instead of a hardcoded claim
- Tests: `tests/test_v02_phase7.py` (28 tests)
- **v0.2.0 Phase 8.1 - Arena Core** (`Benchmark_Plan.txt` Rev.2 spec freeze)
  - `research/benchmarks/player.py`: `Player` protocol (`ingest` / `answer` /
    `finalize` / `reset_costs`), `Answer` (full last-assistant-message text per
    the Answer Extraction Protocol), and `CostReport` covering the mandatory
    write-side cost (write LLM calls / tokens / latency) plus read-side
    retrieval, context construction, answer tokens and total cost —
    Total Cost = Write LLM + Retrieval + Context Construction + Answer LLM
  - `ControlledPlayer` (fixed 2,000-token Official Context Budget, budget
    violations recorded) vs `RealSystemPlayer` (native context management,
    actual tokens measured) — the two arenas are never mixed
  - `research/benchmarks/runner.py`: `ArenaRunner` with the §9 Repeat
    Protocol (3 answer repeats with majority vote, >= 5 latency samples with
    p50/p95), per-run output tree (manifest.json, metrics/latency/tokens CSV,
    failures.jsonl, traces, environment.txt) and `create_arena_manifest`
    reusing `research/experiments/manifest.py`
  - `research/benchmarks/players.py`: skeleton of the five Controlled
    players (Simple RAG / MemoryBank-style / MemGPT-style / Mem0-style /
    AM v0.2.0) with real ingest (embedding + store) and deterministic mock
    answer generation until the fixed LLM is wired in Phase 8.2
  - Robustness fixes found while integrating: `VectorSearchEngine` now
    tolerates a corrupt/empty `metadata.json` (rebuilds instead of crashing)
    and writes its index atomically; Controlled players get isolated
    ephemeral FAISS index dirs (the shared repo-level `vector_index/`
    previously bled memories across players and runs); `AMv020Player` was
    calling the runtime facade's async `remember` without awaiting it (a
    silent no-op — AM ingested nothing) and non-existent
    `recall_adaptive`/`build_context` methods; it now drives
    `asyncio.run(remember(...))` and uses `recall()` + `RecallResult`
    - Tests: `tests/test_phase8_arena.py` (31 tests; budget enforcement,
    repeat protocol, failure recording, real-player native behavior,
    non-LLM write cost semantics, Mem0 write-side token accounting,
    index dir isolation + cleanup)
    - Tests: `tests/test_phase8_leakage.py` (9 tests; Ground Truth Sentinel
    never reaches the LLM prompt, answer API accepts text+context only,
    frozen config / shared instance / config hash stability assertions)
  - Tests: `tests/test_phase8_scorer.py` (27 tests; §10 synthetic case table,
    normalization, repeat-majority + tie-break determinism, pure-function
    invariance, aggregate math, run-replay determinism + hash-mismatch guard,
    scores.csv header validation)
  - Live smoke: `tests/smoke_llm_answer.py` (phi4-mini real run,
    `f5317d523d06a65d...` config hash)
- **v0.2.0 Phase 8.2 - Fixed LLM / Leakage Boundary / Evaluation Boundary**
  - `research/benchmarks/llm.py`: single shared `OllamaAnswerer` for all five
    Controlled players with a frozen configuration (model, temperature 0.0,
    seed 42, prompts, max output tokens) hashed into
    `config_sha256` / `prompt_sha256` for the run manifest
  - Generation signature is `answer(question_text, context)` only, so ground
    truth / forbidden terms / abstention flags cannot reach the prompt by
    design (Leakage Boundary); extraction is a second frozen entry point
    `extract(exchange_text)` for Mem0-style write-side strategies
  - Frozen model changed to `phi4-mini:latest` before the freeze: the original
    candidate `qwen3:4b` ignores both `think: false` and `/no_think` on this
    Ollama install and emitted 300-500 reasoning tokens into `content`
    (~26 s/call), which would have poisoned forbidden-term scoring; recorded in
    `Benchmark_Plan.txt` §5 as a pre-freeze LLM configuration decision
  - Mock answer generation removed: all five players now retrieve real
    context and answer through the shared frozen LLM (`_generate_answer`);
    deterministic abstention is used only when no answerer is injected
    (unit-test mode), so a run cannot silently produce ground-truth-flavoured
    answers
  - Mem0-style player now implements its observed OSS core: single-pass
    ADD-only LLM extraction on write plus semantic + BM25-style keyword +
    entity multi-signal fusion retrieval (0.5 / 0.3 / 0.2)
  - Write-side cost truthfulness fixes: `CostReport.add_write_cost()` now
    counts LLM calls only (`calls=0` for embedding-only ingest and
    deterministic consolidation/paging, latency still recorded), and Mem0-style
    no longer double counts per-turn extraction latency inside E2E write
    latency — so `write_llm_calls` / `write tokens` actually separate
    LLM-write strategies from non-LLM ones
  - Leakage Guard tests (`tests/test_phase8_leakage.py`, 9 tests): a
    `GROUND_TRUTH_SENTINEL_9f83a` planted into dataset ground truth never
    appears in any recorded prompt, `answer` accepts only text + context, and
    the frozen config values / hash stability / shared instance are asserted
  - Arena tests grew to 31 (write-cost semantics regression guards)
  - Ephemeral player index dirs are now tracked in-process and removed at
    interpreter exit (`atexit`), so repeated benchmark runs no longer leave one
    FAISS index per player ingest behind in the system temp dir; a regression
    test asserts per-ingest isolation and removal
  - `tests/smoke_llm_answer.py`: live smoke of the frozen LLM
    (real Ollama answer + one end-to-end player question, manually run)
- **v0.2.0 Phase 8.3 - Scoring & E2E Validation (8.3.1-8.3.4)**
  - `research/benchmarks/scorer.py`: judge-free Scorer per the frozen §10 spec
    (SCORING_SPEC_VERSION `v0.2.0-scoring-1`): NFKC+lowercase+whitespace
    normalization (no punctuation stripping so `f#` survives), word-boundary
    term matching, synonym term groups, forbidden-term zeroing, abstention
    lexicon + frozen `ABSTENTION_TEXT`, outcome table
    (pass/partial/fail/false_positive), blank answers stay out of the
    false-positive bucket
  - Scorer is a pure post-step, fully separated from the runner:
    `score_run_dir()` replays a persisted run (`traces.jsonl` + frozen
    dataset) into `scores.csv` / `aggregates.csv` / `aggregates.json`,
    rejects a dataset-hash mismatch, and can re-score old traces with an
    improved scorer without invalidating the run
  - Official repeat reduction per §10: majority answer by exact text with the
    shared deterministic `majority_index` tie-break; all repeat scores kept
    for `score_consistency`
  - §10 aggregates: answer_accuracy, partial/false_positive/forbidden rates,
    mean coverage, abstention_accuracy, per-category accuracy
  - Scorer unit tests (`tests/test_phase8_scorer.py`, 27): the full §10
    synthetic case table (case/compound/boundary/synonym/symbol), repeat
    reduction determinism, aggregate math, replay determinism, hash-mismatch
    rejection
  - `tests/smoke_e2e_arena.py` (8.3.4): 10 questions (one per category) x
    Simple RAG x 1 repeat with the shared frozen answerer, then replay scoring;
    the §12 output tree (manifest/metrics/latency/tokens/traces/scores) is
    asserted complete and `failures.jsonl` now always exists (empty when clean)
  - Critical retrieval bug found by the E2E (regression test added): all three
    vector-index baselines queried `BasicRecallEngine`, which never touches
    the vector index (insertion-order candidates + word-overlap ranking) —
    the dataset's target fact ranked #1 (0.808 cosine) in the index the
    players had built, yet was never returned, so Simple RAG scored 0.1/10
    before the fix and 6.5/10 after. Baselines now retrieve through
    `VectorSearchEngine` directly (Simple RAG / MemoryBank: dense top-20;
    MemGPT: main-context turns first, then dense archival hits; Mem0 keeps
    its own multi-signal fusion), and baseline context is built by a plain
    budget-capped join instead of the AM `ContextAllocator` (SUT machinery
    must not re-rank a baseline's retrieved lines)
  - E2E observation recorded for 8.3.5+ (not a harness bug): Simple RAG
    asserts concrete values on abstention questions (false_positive) and
    misses cross-session multi-hop evidence within top-20 dense retrieval

## [0.1.0] - 2026-09-15

### Added
- Initial release of Artificial Memory / Context Runtime
- Core memory system with 6 resolution levels (RAW to DEEP_LONG_TERM)
- Progressive memory compression (Forget = Resolution Down)
- Adaptive-resolution recall with provenance tracking
- Topic-based organization with automatic classification
- Memory IR / Context IR formal definitions
- Vector search with FAISS and pgvector backends (`vector` extra)
- Shared SentenceTransformer model cache and lazy module exports (import time reduced from 19s to 0.3s)
- Temporal queries, belief management, and contradiction detection
- Memory integrity checking and self-healing
- Federated multi-agent memory exchange and enterprise governance
- MCP interface for AI agents with 7 verified tools and real-agent cross-session E2E support
- CLI, WebSocket, and REST API
- Reproducible benchmark harness (deterministic recall, Ollama LLM QA, MCP smoke test)
- Docker, docker-compose, and Kubernetes Operator with Helm chart

### Changed
- Pinned `psycopg[binary]` and `mcp>=1.0,<2.0` for stability
- CI enhanced with matrix testing, vector/mcp extras, and non-blocking mypy strict adoption

### Fixed
- Fixed MCP tool response nested Pydantic `MemoryIR` JSON serialization
- Fixed `memory_inspect` event loop deadlock in MCP tools
- Fixed PostgreSQL test timeout with fast connectivity check and graceful skip
- Resolved all Ruff linting issues and majority of type annotations

### Infrastructure
- Docker and docker-compose support
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Comprehensive test suite (113 tests)
- Type checking with mypy (strict mode)
- Linting with ruff

---

## Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create a git tag: `git tag v<version>`
4. Push tag: `git push origin v<version>`
5. GitHub Actions will build and publish release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

MIT License - see [LICENSE](LICENSE) for details.