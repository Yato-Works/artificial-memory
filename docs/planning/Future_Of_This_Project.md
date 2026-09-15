# Artificial Memory V1 — Future Of This Project

> **Cognitive Memory Runtime for Persistent AI Systems**  
> *Forgetting is not deletion. Forgetting is resolution reduction.*

---

## 1. Executive Summary

This document defines the complete implementation roadmap for **Artificial Memory V1** — transforming the current feature-rich prototype into a **research-grade, reproducible, operationally reliable** Cognitive Memory Runtime.

**V1 Scope**: Single-node correctness + semantic validity + empirical evidence + OSS release readiness  
**V1 Non-Scope**: Distributed correctness proof, enterprise governance, autonomous healing, LLM-based compiler

**Core Thesis Implementation**:
```
Conversation → Compiler → MemoryIR → Progressive Resolution → Temporal/Provenance/Confidence
                                                      ↓
                                    Adaptive Recall → ContextIR → LLM
                                                      ↓
                                    Memory Evolution (Belief/Contradiction/Healing)
```

---

## 2. Six Architectural Boundaries (V1 Decisions)

| # | Decision | Rationale |
|---|----------|-----------|
| **1. NLI** | `deberta-v3-base` as **optional extra** (`pip install -e ".[nli]"`) | Core runs fully on rule-based compression. Heavy deps opt-in. |
| **2. Belief Persistence** | **JSONB** in `belief_states.evidence_ids` | V1 ships fast. Normalized schema (`belief_evidence`, `belief_conflict`, `belief_revision`) in V2. |
| **3. Compression Policy Version** | **Mandatory** on `memory_version` | `policy_version` + `algorithm_version` columns enable full provenance traceability. |
| **4. Benchmark Model** | **3-tier**: Local 7B-14B (experiments) → Kaggle (reproducibility) → Nemotron 550B (Judge/Analyst) | Nemotron reviews *results*, doesn't *run* experiments. Cost-effective + credible. |
| **5. Kubernetes Operator** | **Retain (A)** with explicit disclaimer | "Deployment/orchestration primitives only. Not proof of distributed memory consistency." |
| **6. Interface Priority** | **MCP → Python SDK → REST** | MCP demonstrates "AI-native memory" instantly. SDK for integration. REST for compatibility. |

---

## 3. Phase Plan

### PHASE 0: P0 Critical Fixes (Week 1-2)
*Foundation: secrets, persistence, correctness, reproducibility*

| ID | Task | Files | Acceptance |
|----|------|-------|------------|
| P0-1 | Remove hardcoded secrets from `RuntimeConfig` | `src/artificial_memory/runtime/facade.py:106-136` | `grep -r "dev-secret\|default_.*key" src/` → 0 hits |
| P0-2 | Add LICENSE, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md | Root | GitHub community standards pass |
| P0-3 | Fix `ConfidenceProfile.overall` to combine 4 components | `src/artificial_memory/core/ir/common.py:120-127` | `overall = weighted_sum(memory, retrieval, temporal, source)` |
| P0-4 | Persist `BeliefEngine._beliefs` to DB (JSONB) | `src/artificial_memory/memory/belief.py`, new migration | Restart preserves beliefs; `belief_at(timestamp)` works |
| P0-5 | Persist `EvolutionEvent` log (currently stubbed) | `src/artificial_memory/memory/evolution.py:474-482`, migration | Every revise/merge/split/heal creates event row |
| P0-6 | Populate `ProvenanceChain.compilation_chain` in compiler | `src/artificial_memory/compiler/pipeline.py`, all stages | `MemoryIR.provenance.compilation_chain` non-empty |
| P0-7 | Add `policy_version`, `algorithm_version` to `memory_version` | Migration + `src/artificial_memory/core/models.py` | Every compressed version records how it was made |
| P0-8 | Fix Windows test flakiness (tempfile cleanup) | `tests/conftest.py` or test fixtures | CI green on Windows runner |
| P0-9 | Create `uv.lock` / pinned dependencies | `pyproject.toml` → `uv lock` | `uv sync --frozen` works cleanly |
| P0-10 | RuntimeConfig raises clear error for required secrets | `src/artificial_memory/runtime/facade.py` | No silent defaults for JWT, keys |

---

### PHASE 1: Core Architecture Stabilization (Week 2-4)
*IR packages, Context separation, Facade audit, Compiler formalization*

| ID | Task | Files | Acceptance |
|----|------|-------|------------|
| P1-1 | Create `core/ir/` package with schema_version | `src/artificial_memory/core/ir/{memory,context,belief,provenance,temporal,common}.py` | All models: `strict=True, extra="forbid", schema_version` |
| P1-2 | Split `ContextIR` → `ContextPackage` + `ContextBuildTrace` | `src/artificial_memory/core/ir/context.py` | Package = LLM payload; Trace = diagnostics/reproducibility |
| P1-3 | Update `EnhancedContextBuilder.build_context_ir()` to return both | `src/artificial_memory/context/builder.py:245-289` | Returns `(ContextPackage, ContextBuildTrace)` |
| P1-4 | Audit `ArtificialMemoryRuntime` — remove internal leakage | `src/artificial_memory/runtime/facade.py` | Public methods return IR types only; no legacy models |
| P1-5 | Add `async close()` + context manager to Facade | `src/artificial_memory/runtime/facade.py:558-566` | `async with ArtificialMemoryRuntime() as rt:` works |
| P1-6 | Formalize Compiler Stage protocol (name, version, config_schema) | `src/artificial_memory/compiler/pipeline.py:52-58` | Stages configurable for ablation; deterministic path valid |
| P1-7 | Create NLI optional extra structure | `src/artificial_memory/core/compression/nli/`, `pyproject.toml` | `pip install -e ".[nli]"` adds transformers+torch |
| P1-8 | Document Facade methods with OpenAPI-compatible docstrings | `src/artificial_memory/runtime/facade.py` | Auto-generates REST/MCP schemas |

---

### PHASE 2: Semantic Compression & Validation (Week 4-6) ★ **Most Important Research**
*Compression with semantic guarantees, not just word overlap*

| ID | Task | Files | Acceptance |
|----|------|-------|------------|
| P2-1 | `CompressionPolicy` class + versioning | `src/artificial_memory/core/compression/policies.py` | `preserve: [decision, reason, constraints, temporal_validity]`, `allow_loss: [filler, repetition]` |
| P2-2 | `SemanticVerifier` with 5 preservation metrics | `src/artificial_memory/core/compression/verifier.py` | `verify_entailment()`, `verify_entities()`, `verify_relations()`, `verify_temporal()`, `verify_decisions()` |
| P2-3 | NLI integration (DeBERTa-v3-base) with rule-based fallback | `src/artificial_memory/core/compression/nli/` | Optional extra; caches results; gracefully degrades |
| P2-4 | Compression pipeline: Candidate → Verify → Accept/Reject | `src/artificial_memory/compiler/stages/compression.py` | Rejects compression if `preservation_score < policy.min_preservation_score` |
| P2-5 | Record `policy_version` + `algorithm_version` on `MemoryVersion` | `src/artificial_memory/core/models.py`, migration | Full trace: "compressed by rule-compressor-v1 under compression-v1.2" |
| P2-6 | Update `IntegrityMetrics.semantic_preservation_score` to use NLI | `src/artificial_memory/memory/integrity.py:74-115` | Word overlap → entailment + entity + relation + temporal + decision |

---

### PHASE 3: Belief Engine & Contradiction Detection (Week 5-7)
*Explicit belief/evidence separation with temporal validity*

| ID | Task | Files | Acceptance |
|----|------|-------|------------|
| P3-1 | `belief_states` table migration (JSONB evidence) | `scripts/schema_postgres.sql` + SQLite equivalent | `belief_id, proposition, confidence, evidence_ids JSONB, contradicting_evidence JSONB, valid_from, valid_until, revision` |
| P3-2 | Persist `BeliefEngine` to DB (replace in-memory dict) | `src/artificial_memory/memory/belief.py:66-71` | Restart preserves beliefs; `belief_at(timestamp)` uses persisted state |
| P3-3 | `ContradictionPipeline` with 5 stages | `src/artificial_memory/memory/contradiction/pipeline.py` | EntityNorm → PredicateNorm → TemporalCheck → RuleDetector → NLIDetector |
| P3-4 | Contradiction type classification | `src/artificial_memory/memory/contradiction.py:13-21` | `Contradiction / Supersession / TemporalChange / Uncertainty / Ambiguity` |
| P3-5 | Belief revision with revision counter | `src/artificial_memory/memory/belief.py:117-134` | `belief.update(evidence_delta)` increments revision; `belief_at(t)` works |
| P3-6 | `belief_conflicts()` and `belief_history()` APIs | `src/artificial_memory/memory/belief.py:329-384` | Returns full conflict timeline + resolution status |

---

### PHASE 4-9: Parallelizable Hardening (Week 7-13)
*Can run in parallel once Phases 1-3 stabilize*

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| **P4** | Confidence Calibration | ECE, reliability diagrams, Brier score; `ConfidenceProfile.overall` uses all 4 components |
| **P5** | Provenance Completion | `compilation_chain` populated; `evolution_events` table; full trace: Context → Memory → Version → Event → Stage → Conversation → Message |
| **P6** | Temporal Disambiguation | 6 `TimePointType` enums; historical algorithm versioning in `TimeTravelEngine` |
| **P7** | Fused Recall + Adaptive Allocation | RRF fusion (vector + keyword + metadata); MMR diversity; utility-driven token allocation (replaces fixed 50/35/15) |
| **P8** | Integrity Checks + Controlled Healing | 6 check types; healing = re-expand → recompile → validate → new version (human-in-loop for V1) |
| **P9** | Distributed Foundation | Idempotent workers (Redis keys); Vector index sync (event-driven); Transaction boundaries; Status conditions (`IndexOutOfSync`, `StorageUnavailable`, etc.) |

---

### PHASE 10: Research Benchmark + Red-Team (Week 13-16)
*Reproducible empirical validation*

| ID | Task | Specification |
|----|------|---------------|
| P10-1 | Fixed synthetic dataset generator | 100-1000 sessions, known ground truth (facts, beliefs, decisions, contradictions per timestamp) |
| P10-2 | 12 Baseline implementations | Raw / Fixed-window / Summary / Vector RAG / Summary+Vector / MemGPT / AM w/o compression / AM w/o temporal / AM w/o belief / AM w/o provenance / AM w/o adaptive / Full AM |
| P10-3 | 19 Core metrics instrumented | Long-term recall, temporal consistency, contradiction resolution, belief consistency, semantic preservation, decision preservation, false/poisoned/stale memory resistance, confidence calibration (ECE), provenance completeness, token efficiency, latency, storage growth |
| P10-4 | Statistical methodology | Paired design, bootstrap CI, effect sizes (Cohen's d), Bonferroni correction, pre-registered analysis |
| P10-5 | 20 Red-team scenarios as regression tests | Each: reproduction script + detection metric + mitigation test; runs in CI |
| P10-6 | Nemotron 550B result review | Judge analyzes: "Are conclusions warranted? Are baselines fair? Is methodology sound?" |

---

### PHASE 11: OSS Release Preparation (Week 16-18)
*Public release hardening*

| ID | Task | Deliverable |
|----|------|-------------|
| P11-1 | **MCP Server** (7 tools) | `memory.remember`, `memory.recall`, `memory.expand`, `memory.trace`, `memory.timeline`, `memory.inspect`, `memory.explain` |
| P11-2 | Python SDK | `ArtificialMemoryRuntime` async API with typed returns |
| P11-3 | Hugging Face Space Demo | PostgreSQL backend + visualization: RAW → LIGHT → EPISODE → SEMANTIC → LONG_TERM progression |
| P11-4 | Documentation site (mkdocs) | Quick Start, Architecture, Memory Model, Compiler, Recall, Context, Temporal, Belief, Provenance, Benchmarking, MCP, Kubernetes, Troubleshooting, Contributing |
| P11-5 | Example notebooks | `basic_usage.ipynb`, `time_travel.ipynb`, `contradiction_demo.ipynb` |
| P11-6 | Release automation | GitHub Actions: test → build → PyPI → Docker Hub → GH Release (semver) |
| P11-7 | Benchmark reproduction guide | `BENCHMARKS.md` with exact commands, seeds, hardware spec |

---

### PHASE 12: Production Hardening (Week 18-22)
*Operational maturity for enterprise adoption*

- Stability soak tests (1000 sessions)
- Prometheus metrics + Grafana dashboards + alerting
- Alembic migration framework + custom data migrations
- Backup/recovery procedures (tested)
- Security audit (deps, secrets, pen test)
- Operator upgrade/rollback + chaos engineering

---

## 4. Key Technical Specifications

### 4.1 MemoryIR (Canonical Representation)
```python
class MemoryIR(BaseModel):
    schema_version: str = "memory-ir-v1.0.0"
    identity: MemoryIdentity
    type: MemoryType
    resolution: ResolutionLevel
    content: str
    semantic_metadata: SemanticMetadata
    temporal_scope: TemporalScope
    confidence: ConfidenceProfile  # 4-component, computed
    provenance: ProvenanceChain    # compilation_chain populated
    relations: list[MemoryRelation]
    dependencies: list[MemoryDependency]
    lifecycle: LifecycleState
    compression_history: list[CompressionRecord]
    access_summary: AccessSummary
```

### 4.2 ContextIR Separation
```python
# LLM-facing payload
class ContextPackage(BaseModel):
    schema_version: str
    query: str
    temporal_snapshot: datetime
    parts: list[ContextPart]
    token_budget: int
    token_count: int
    model_target: ModelTarget

# Diagnostic/reproducibility artifact
class ContextBuildTrace(BaseModel):
    schema_version: str
    query: str
    recall_strategy_version: str
    ranking_algorithm_version: str
    compression_policy_version: str
    considered_parts: list[ContextCandidate]
    selected_parts: list[ContextPart]
    rejected_parts: list[RejectedContextPart]
    expansion_decisions: list[ExpansionDecision]
    compression_decisions: list[CompressionDecision]
    temporal_snapshot: datetime
    statistics: ContextStats
    runtime_version: str
```

### 4.3 Compression Policy (Versioned)
```python
class CompressionPolicy(BaseModel):
    policy_id: str
    version: str
    preserve: list[str]           # decision, reason, constraints, temporal_validity
    allow_loss: list[str]         # filler, repetition, incidental_context
    min_preservation_score: float # per-category thresholds
    # Example: "compression-v1.2" with "decision_memory" profile
```

### 4.4 Confidence Profile (Calibrated)
```python
class ConfidenceProfile(BaseModel):
    memory_confidence: float
    retrieval_confidence: float
    temporal_confidence: float
    source_confidence: float
    overall: float  # COMPUTED: weighted geometric mean, not copied
```

### 4.5 MemoryVersion Extended Provenance
```sql
ALTER TABLE memory_versions ADD COLUMN policy_version VARCHAR(50);
ALTER TABLE memory_versions ADD COLUMN algorithm_version VARCHAR(50);
ALTER TABLE memory_versions ADD COLUMN compilation_provenance JSONB;  -- stage chain
```

---

## 5. Benchmark Methodology (Locked for V1)

### 5.1 Dataset
- **Fixed synthetic conversations** generated once, replayed identically
- **Ground truth per timestamp**: facts, beliefs, decisions, contradictions
- **Scales**: 10 (smoke) → 50 (debug) → 100 (dev) → 250 (stability) → 500 (stress) → 1000 (final)

### 5.2 Baselines (12, all same model/data/queries/budget)
1. Raw conversation context
2. Fixed-window memory
3. Traditional summarization
4. Vector RAG
5. Summary + Vector RAG
6. Existing LT memory (MemGPT/Letta)
7. AM without progressive compression
8. AM without temporal reasoning
9. AM without belief management
10. AM without provenance
11. AM without adaptive recall
12. Full AM

### 5.3 Metrics (19 Core)
| Category | Metrics |
|----------|---------|
| **Memory Quality** | Long-term recall accuracy, Cross-session fact recall, Temporal consistency, Contradiction resolution, Belief consistency, Semantic preservation, Decision preservation |
| **Reliability** | False-memory resistance, Poisoned-memory resistance, Stale-memory resistance, Confidence calibration (ECE), Provenance completeness |
| **Efficiency** | Input tokens, Context tokens, Storage size, Recall latency, End-to-end latency, Memory compilation cost |
| **Scalability** | Memory growth/session, Index size growth, Throughput, Worker scaling, Recovery time |

### 5.4 Statistical Rigor
- Paired experimental design (same conversations × all conditions)
- Bootstrap confidence intervals (10,000 resamples)
- Effect sizes (Cohen's d) reported alongside p-values
- Bonferroni correction for 12 baseline comparisons
- Pre-registered analysis config (seeds, model settings, benchmark version)

---

## 6. V1 Definition of Done

### Architecture
- [ ] Memory IR stable, versioned, strict validation
- [ ] Context Package + Build Trace separated
- [ ] Legacy ↔ IR adapters lossless round-trip tested
- [ ] Runtime Facade coherent, no internal leakage
- [ ] Memory/Evidence/Belief/Context/Agent State clearly separated

### Memory Semantics
- [ ] Progressive resolution consistent (RAW→DEEP_LONG_TERM)
- [ ] Compression preserves critical semantics (NLI-verified)
- [ ] Memory history traceable (version chain + policy_version)
- [ ] Beliefs persistent + temporally valid
- [ ] Contradictions explicit typed objects
- [ ] Evolution events persisted for all meaningful changes

### Research Integrity
- [ ] Compression quality measured semantically (not word overlap)
- [ ] Confidence calibrated (ECE, reliability diagrams)
- [ ] Benchmark methodology reproducible (fixed dataset, seeds, config)
- [ ] Baselines + ablations defined before final evaluation
- [ ] Red-team scenarios repeatable + regression tested
- [ ] Raw experiment artifacts preserved (JSONL per conversation)

### Operational Reliability
- [ ] Storage migrations versioned (Alembic + data migrations)
- [ ] No hardcoded secrets
- [ ] Distributed execution has explicit consistency semantics (documented)
- [ ] Worker processing idempotent (keys: `compile:conv_123:compiler_v1.2`)
- [ ] Failure/recovery behavior tested

### Public Release
- [ ] Documentation complete (13 sections)
- [ ] License + security policy present
- [ ] Clean install from fresh environment works
- [ ] Examples work
- [ ] Benchmark reproduction documented
- [ ] MCP server functional (7 tools)
- [ ] HF Space demo live

---

## 7. V2 Vision (Not V1 Scope)

```
V1: Cognitive Memory Runtime
    ↓
V2: Cognitive State Runtime
```

**Future IRs**: `GoalIR`, `PlanIR`, `TaskIR`, `SkillIR`, `SelfModelIR`, `WorldStateIR`  
**Future Engines**: Reflection, Experience Learning, World Model, Self Model, Goal Management, Cognitive Federation, Trust/Delegation

---

## 8. Immediate Next Steps (For You)

1. **Save this document** as `Future_Of_This_Project.md`
2. **Start Phase 0** — 10 tasks, ~1-2 weeks, all P0 correctness
3. **Create GitHub Issues** from the task tables above (copy-paste ready)
4. **Set up project board** with Phases as columns
5. **Run benchmark smoke test** (10 sessions) after Phase 2 to validate direction

---

## 9. Repository Hygiene Checklist (Pre-Release)

- [ ] `git-filter-repo` audit — no secrets in history
- [ ] `uv.lock` committed
- [ ] `SECURITY.md` with vulnerability reporting email
- [ ] `CODE_OF_CONDUCT.md` (Contributor Covenant)
- [ ] Semver strategy documented (`MAJOR.MINOR.PATCH`, V1 = 1.0.0)
- [ ] Release checklist in `RELEASE.md`
- [ ] Dependabot / Renovate configured
- [ ] CI: unit + integration + typecheck + lint on every PR

---

**This plan is complete. Every task is actionable, every decision bounded, every phase has clear acceptance criteria. Start Phase 0.**