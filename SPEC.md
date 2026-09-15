# Artificial Memory / Context Runtime — Technical Specification

> **A Cognitive Memory Runtime for Persistent AI Systems**
> *Forget by compression. Recall by resolution. Reason with provenance.*

---

## 🎯 Vision Statement

Artificial Memory is not merely a "better RAG" — it is a **general-purpose cognitive memory runtime** that manages persistent AI memory throughout its entire lifecycle. It implements human-like memory with progressive compression, adaptive recall, temporal reasoning, and full provenance tracking.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Artificial Memory                            │
│         Cognitive Memory Runtime for Persistent AI             │
├─────────────────────────────────────────────────────────────────┤
│  CLI          │  HTTP API        │  WebSocket  │  Web UI   │
├─────────────────────────────────────────────────────────────────┤
│                    Runtime Facade (Unified Entry Point)         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Recall    │ │   Context   │ │  Confidence │               │
│  │   Runtime   │ │   Runtime   │ │   Engine    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    Memory Compiler (10 Deterministic Stages)    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Lexical    │ │  Semantic   │ │  Fact/      │               │
│  │  Analysis   │ │  Extraction │ │  Decision   │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Episode    │ │  Temporal   │ │  Provenance │               │
│  │  Construction│ │  Linking    │ │  Linking    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│              Memory Evolution Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │Consolidation│ │Contradiction│ │   Belief    │               │
│  │  Engine     │ │  Detection  │ │  Engine     │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Healing   │ │ Dependency  │ │  Temporal   │               │
│  │  (Integrity)│ │   Graph     │ │  Updates    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    IR Layer (Intermediate Representation)       │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │    Memory IR    │  │   Context IR    │                      │
│  └─────────────────┘  └─────────────────┘                      │
├─────────────────────────────────────────────────────────────────┤
│  SQLite  │  FAISS/pgvector  │  File Storage (Markdown/JSON)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Memory System

### Memory Resolution Model (6 Levels)

| Level | Name | Compression Ratio | Use Case |
|-------|------|-------------------|----------|
| 0 | **RAW** | 1.0x | Original conversation, verbatim |
| 1 | **LIGHT** | ~2-3x | Redundancy removal, fillers preserved |
| 2 | **EPISODE** | ~5-10x | "What happened" - narrative summary |
| 3 | **SEMANTIC** | ~20-50x | "What was decided" - decisions, facts |
| 4 | **LONG_TERM** | ~100x+ | Abstract patterns, principles |
| 5 | **DEEP_LONG_TERM** | ~500x+ | Core beliefs, identity, values |

### Key Principles

1. **Forget = Resolution Down** — Not deletion, but progressive compression
2. **Progressive Recall** — Expand resolution only when tokens provide utility
3. **Conversation Preservation** — Tone, fillers, hesitation preserved at RAW
4. **Context IR** — Dense intermediate representation for LLM
5. **Human-like Recall** — Adaptive resolution based on age/importance/relevance
6. **Provenance Tracking** — Every memory traces to source conversation/message
7. **Evidence ≠ Belief** — Explicit contradiction detection, belief state separate
8. **Memory Evolution** — Memories change; history preserved; healing possible

---

## 🔍 Advanced Retrieval & Recall

### Recall Levels
- **Level 0** (CURRENT_ONLY) — Only active, current-resolution memories
- **Level 1** (CURRENT_AND_LIGHT) — Active + light compressed
- **Level 2** (STANDARD) — Current + light + episode
- **Level 3** (DEEP) — All except deep long-term
- **Level 4** (EXHAUSTIVE) — All resolutions including archived

### Search Capabilities
- **Vector Search** — FAISS (local) / pgvector (PostgreSQL) with hybrid keyword+vector
- **Temporal Queries** — Time-travel with `valid_from`/`valid_until`
- **Associative Memory** — Graph-based associations with dependency tracking
- **Provenance Tracking** — Full traceability from memory → source conversation
- **Counterfactual Recall** — Measure memory influence, not just relevance

### Context Runtime
- **Priority-based Budget Allocation** — HIGH (50%), MEDIUM (35%), LOW (15%)
- **Token Optimization** — Utility/token scoring for optimal context packing
- **System Prompt Generation** — Auto-generated from topic context

---

## 🤖 LLM Integration

### Multi-Provider Support
| Provider | Models | Features |
|----------|--------|----------|
| **Ollama** (local-first) | llama3.1, etc. | Privacy, offline, custom models |
| **OpenAI** | gpt-4o, etc. | High quality, function calling |

### Chat Features
- Streaming responses
- Context-aware with automatic recall
- Configurable recall level per request
- Token usage tracking

---

## 🔐 Multi-User & Security

### Authentication
- User registration & login (JWT)
- Session management with secure tokens
- API keys with scopes & expiration
- Role-based access control

### Isolation
- Topic-level memory isolation per user/project
- Tenant support (logical / schema / physical)

---

## 📊 Observability & Research Platform

### Real-time Metrics
- Compression ratios per memory
- Recall accuracy & latency
- Token costs per conversation
- Memory evolution events

### Experiment Framework
```python
config = ExperimentConfig(
    name="comparison",
    experiment_types=[
        ExperimentType.RAW_CONVERSATION,
        ExperimentType.TRADITIONAL_SUMMARY,
        ExperimentType.VECTOR_MEMORY,
        ExperimentType.TEMPORAL_MEMORY,
        ExperimentType.ARTIFICIAL_MEMORY,
    ],
    num_conversations=10,
    topics=["Architecture", "API Design", "Database"],
)

runner = ExperimentRunner(config)
results = await runner.run_all_experiments()
```

### Built-in Capabilities
| Component | Purpose |
|-----------|---------|
| **Benchmark Harness** | Reproducible quantitative evaluation |
| **Ablation Framework** | Component removal studies |
| **Red-Team Suite** | Adversarial robustness testing |
| **Memory Debugger** | Explain selection/rejection/expansion |
| **AM-Specific Benchmarks** | Temporal, contradiction, false memory, compression loss |

---

## 🌐 Distributed & Enterprise (Phase 7+)

### Kubernetes Operator
```yaml
apiVersion: memory.artificialmemory.dev/v1
kind: ArtificialMemoryCluster
spec:
  replicas: 3
  storage:
    backend: postgres
    vector: pgvector
  runtime:
    recallWorkers: 4
    consolidationWorkers: 2
    compilerWorkers: 2
  policy:
    compression: adaptive
    retention: policy-driven
    governance: enabled
```

### CRDs Managed
- `ArtificialMemoryCluster` — Cluster orchestration
- `MemoryStore` — Storage backend (PostgreSQL + pgvector)
- `MemoryWorker` — Compilation/consolidation workers
- `RecallWorker` — Query processing workers
- `VectorIndex` — Distributed vector index shards

### Federation (Multi-Agent)
- Policy-driven memory exchange
- Trust levels: UNTRUSTED → LOW → MEDIUM → HIGH → TRUSTED
- Provenance-preserving transfer
- Expiration & revocation support

### Governance
- Classification labels (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)
- Permission policies per domain (MEMORY, RECALL, ADMIN)
- Audit logging with tamper-evident records
- Retention policies (time-based, size-based, event-driven)

---

## 🛠️ Development & Operations

### Technology Stack
| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Core | Pydantic v2, FastAPI, SQLite/PostgreSQL |
| Vector | FAISS (local), pgvector (distributed) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Async | asyncio, uvicorn |
| CLI | Click + Rich |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Linting | Ruff (fast, Rust-based) |
| Type Checking | mypy (strict mode) |

### Commands
```bash
# Development
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
mypy src/artificial_memory

# Docker
docker-compose up -d

# Production
docker build -t artificial-memory .
docker run -d -p 8000:8000 artificial-memory

# Kubernetes
helm install artificial-memory ./k8s/helm/artificial-memory-operator
```

---

## 📈 Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Recall Latency | < 100ms | With vector index warm |
| Compression Ratio | 10-500x | Depending on resolution level |
| Memory Footprint | < 500MB | For 1M memories (compressed) |
| Horizontal Scaling | Linear | Add recall/compilation workers |
| Vector Search | 150x faster | With HNSW vs flat FAISS |

---

## 🔬 Research Differentiators

| Capability | Standard RAG | Artificial Memory |
|------------|--------------|-------------------|
| Progressive Compression | ❌ | ✅ 6 levels |
| Temporal Queries | ❌ | ✅ Time travel |
| Belief vs Evidence | ❌ | ✅ Explicit separation |
| Contradiction Detection | ❌ | ✅ Multi-strategy |
| Memory Healing | ❌ | ✅ Auto-recompile |
| Provenance Chain | ❌ | ✅ Full traceability |
| Counterfactual Recall | ❌ | ✅ Influence scoring |
| Federated Exchange | ❌ | ✅ Trust policies |
| Kubernetes Native | ❌ | ✅ Operator + CRDs |

---

## 📦 Package Structure

```
src/artificial_memory/
├── cli/                 # CLI commands (30+ commands)
├── api/                 # FastAPI REST + WebSocket
├── mcp/                 # MCP server (priority interface)
├── runtime/             # Unified facade (ALL interfaces)
├── core/
│   ├── models/          # 30+ Pydantic models
│   ├── interfaces/      # Protocols (MemoryStore, etc.)
│   ├── ir/              # MemoryIR, ContextIR
│   └── adapters/        # Legacy ↔ IR conversion
├── memory/
│   ├── compiler/        # 10-stage pipeline
│   ├── consolidation/   # Compression scheduler
│   ├── belief/          # Belief engine
│   ├── contradiction/   # Detection & resolution
│   ├── evolution/       # Revise, merge, split, heal
│   ├── dependency/      # Graph + impact analysis
│   ├── counterfactual/  # Influence scoring
│   ├── integrity/       # Semantic preservation
│   ├── stale/           # Staleness detection
│   ├── healing/         # Auto-repair
│   ├── federation/      # Multi-agent exchange
│   ├── trust/           # Trust policies
│   ├── tenancy/         # Multi-tenant isolation
│   └── governance/      # Audit, retention, policy
├── recall/
│   ├── engine/          # Basic recall
│   ├── adaptive/        # Utility-based recall
│   └── human_recall/    # Human-like simulation
├── context/
│   ├── builder/         # Context IR construction
│   └── ir_compiler/     # NL → IR compiler
├── temporal/
│   ├── time_travel/     # State at timestamp
│   ├── belief_history/  # Belief evolution
│   ├── context_reconstruction/  # Historical contexts
│   └── impact_analysis/ # Change simulation
├── storage/
│   ├── sqlite_store.py  # SQLite implementation
│   └── postgres_store.py # PostgreSQL + pgvector
├── auth/                # JWT, sessions, API keys
├── llm/                 # Provider abstraction
├── observability/       # Metrics collection
├── governance/          # Enterprise policies
├── experiments/         # Benchmark/ablation/redteam
├── topic/               # Classification
└── compression/         # Rule-based compressor
```

---

## 📊 Current Status (v0.1.0)

| Component | Status | Tests |
|-----------|--------|-------|
| Core Memory CRUD | ✅ Complete | 10/10 pass |
| Conversation Logging | ✅ Complete | 10/10 pass |
| Compression (4 levels) | ✅ Complete | 10/10 pass |
| Vector Search (FAISS) | ✅ Complete | 12/12 pass |
| Recall Engine | ✅ Complete | 10/10 pass |
| Context Builder | ✅ Complete | 10/10 pass |
| Memory Compiler | ✅ Complete | 10/10 pass |
| Topic Classification | ✅ Complete | 10/10 pass |
| Memory Evolution | ✅ Complete | 28/28 pass |
| Belief Engine | ✅ Complete | 28/28 pass |
| Contradiction Detection | ✅ Complete | 28/28 pass |
| Temporal Queries | ✅ Complete | 25/25 pass |
| Integrity & Healing | ✅ Complete | 25/25 pass |
| Federation | ✅ Complete | 15/15 pass |
| Governance | ✅ Complete | 15/15 pass |
| **Total** | **✅ 113/113 pass** | |

---

## 🚀 Roadmap

### Phase 1 — Foundation ✅
- Memory IR & Context IR formal definitions
- Legacy Memory ↔ MemoryIR lossless adapter
- Runtime Facade (unified entry point)
- Deterministic Compiler Pipeline

### Phase 2 — Advanced Memory Runtime ✅
- Adaptive Recall (utility/token optimization)
- Memory Evolution Engine (revision, merge, split)
- Contradiction Detection & Belief State
- Dependency Graph & Impact Analysis

### Phase 3 — Memory Integrity ✅
- Integrity Metrics (semantic preservation)
- Stale Memory Detection
- Memory Healing (auto-recompile)
- Counterfactual Recall Engine

### Phase 4 — Temporal & Debugging ✅
- Memory Time Travel (state/belief/context at timestamp)
- Memory Debugger (selection/rejection/expansion explanations)
- Decision Trace & Impact Analysis

### Phase 5 — Research Platform ✅
- Benchmark Harness (reproducible experiments)
- Ablation Framework
- Red-Team Suite (adversarial testing)
- AM-Specific Synthetic Benchmarks

### Phase 6 — Multi-Agent & Enterprise ✅
- Federated Memory Exchange
- Trust Policies & Governance
- Tenant Isolation & Audit
- Retention Policies

### Phase 7 — Distributed Runtime ✅
- PostgreSQL / pgvector Backend
- Distributed Vector Indexing
- Worker Architecture (queue-based compilation)
- Horizontal Scaling

### Phase 8 — Kubernetes ✅
- Kubernetes Operator (CRDs + reconciliation)
- Helm Chart for deployment
- Integration tests with kind/k3s (in progress)
- Production hardening (network policies, PDBs)

### Phase 9+ — Future
- Distributed consistency proofs
- Cryptographic provenance
- Neurosymbolic reasoning integration
- Edge deployment optimization

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, ground rules, testing, and PR process.

---

## 📚 Citation

```bibtex
@software{artificial-memory,
  title = {Artificial Memory / Context Runtime},
  subtitle = {A Cognitive Memory Runtime for Persistent AI Systems},
  author = {Artificial Memory Project},
  year = {2024},
  url = {https://github.com/artificial-memory/artificial-memory}
}
```

---

*Built with ❤️ for the future of persistent AI cognition.*