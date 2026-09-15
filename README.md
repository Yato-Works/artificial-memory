# Artificial Memory / Context Runtime

> **Forget by compression. Recall by resolution. Reason with provenance.**

A **Cognitive Memory Runtime for Persistent AI Systems** that implements human-like memory with progressive compression, adaptive recall, temporal reasoning, and full provenance tracking.

## Vision

Artificial Memory is not merely a "better RAG" — it is a **general-purpose cognitive memory runtime** that manages persistent AI memory throughout its entire lifecycle:

- **Progressive Memory Compression** — Forgetting = Resolution Down, not Deletion
- **Adaptive-Resolution Recall** — Expand only when tokens provide maximum utility
- **Context as Intermediate Representation** — Memory → Recall → Prioritization → Resolution Expansion → Budget Allocation → Context IR → LLM
- **Provenance as First-Class Property** — Every memory traces back to source conversation/message
- **Memory as Evolving Object** — Created → Accessed → Compressed → Expanded → Contradicted → Revised → Consolidated → Archived → Recompiled
- **Contradiction Detection & Belief Management** — Evidence ≠ Belief; explicit contradiction tracking with temporal validity
- **Temporal Memory & Time Travel** — Reconstruct "what the system knew/believed at time T"
- **Memory Integrity & Self-Healing** — Semantic preservation scoring, automatic recompilation on corruption
- **Federated Multi-Agent Memory** — Policy-driven memory exchange with trust, provenance, expiration
- **Enterprise Governance** — Classification, permissions, audit, tenancy, retention as architectural layer

## Features

### 🧠 Core Memory System
- **Memory Resolution Model**: 6 levels from RAW (0) to DEEP_LONG_TERM (5)
- **Progressive Forgetting**: Forget = Resolution Down, not Delete
- **Progressive Recall**: Recall = Resolution Up, expand only when needed
- **Topic-based Organization**: Automatic topic classification and hierarchy
- **Memory IR / Context IR**: Formal intermediate representations for compilation, recall, and context building

### 🔍 Advanced Retrieval
- **Vector Search**: FAISS-based semantic search with hybrid keyword+vector
- **Temporal Queries**: Time-travel queries with valid_from/valid_until
- **Associative Memory**: Graph-based memory associations with dependency tracking
- **Provenance Tracking**: Full traceability from memory to source conversation
- **Counterfactual Recall**: Measure memory influence, not just relevance

### 🤖 LLM Integration
- **Context Runtime**: Automatic context building with priority-based budget (utility/token optimization)
- **Multi-provider**: Ollama (local-first) and OpenAI support
- **Confidence Scoring**: 4-component confidence with natural language expression
- **Human-like Recall**: Adaptive resolution based on memory age/importance/relevance

### 🔐 Multi-user & Security
- **User Management**: Registration, authentication, roles
- **API Keys**: Scoped keys with expiration
- **Session Management**: Secure token-based sessions
- **Topic Isolation**: User/project-level memory isolation

### 📊 Observability & Research
- **Real-time Metrics**: Compression ratios, recall accuracy, token costs
- **Experiment Framework**: Quantitative evaluation with ablation studies
- **Memory Debugger**: Explain why memory was selected/rejected, why resolution expanded/not
- **Red-Team Suite**: Adversarial testing for robustness
- **Web UI**: Dashboard for memory visualization
- **WebSocket**: Real-time updates

## Quick Start

## Benchmarks

Measured with `python scripts/run_readme_benchmark.py --memories 120 --queries 50 --seed 42`
(Python 3.11 on Windows 10, Intel i7-11700; embeddings: all-MiniLM-L6-v2, FAISS IndexFlatIP).
Reproduce on your machine — numbers below are from the dev workstation, not a benchmark lab.

| Metric | Diverse domains | Stress (near-duplicate) |
|---|---|---|
| Semantic recall accuracy@1 (vector / hybrid) | **66.7% / 79.2%** | 29.2% / 41.7% |
| Semantic recall accuracy@5 (vector / hybrid) | **100% / 100%** | 57.5% / 72.5% |
| Recall latency p50 / p95 | 27 / 32 ms | 26 / 28 ms |
| Token savings vs full-context injection | **69.6%** | 70.9% |
| Vector index build (120 memories) | ~1.2 s | ~1.6 s |
| Remember throughput | ~530 writes/s | ~550 writes/s |

Two synthetic datasets are benchmarked deliberately:
**"diverse"** — memories from distinct domains (realistic retrieval setup);
**"stress"** — 120 near-identical templates that differ only in entity/tech (adversarial).
The gap between them quantifies how resolution/hybrid recall degrades under
homogeneous content — exactly the failure mode we want to keep visible.

### Installation
```bash
git clone https://github.com/your-org/artificial-memory
cd artificial-memory
pip install -e ".[web,llm,vector]"
```

> The `vector` extra installs FAISS + sentence-transformers for semantic
> vector search (numpy is installed as a core dependency). Without it, the
> package still works with keyword-only recall.

### Start with Docker (Recommended)
```bash
docker-compose up -d
# Access UI at http://localhost:8000/ui
# API at http://localhost:8000/docs
```

### Use with AI Agents (MCP — Priority Interface)

Interface priority for V1: **MCP → Python SDK → REST**.

```bash
pip install -e ".[mcp]"
# Then register with your MCP client (Claude Desktop, IDE, agent):
python -m artificial_memory.mcp
```

Exposed tools (all routed through the Runtime Facade):

| Tool | Purpose |
|------|---------|
| `memory_remember` | Store durable facts / decisions with provenance |
| `memory_recall` | Adaptive-resolution recall with full provenance |
| `memory_expand` | Expand a compressed memory to higher resolution |
| `memory_trace` | Trace a memory back to its source conversation |
| `memory_explain` | Explain recall selection decisions |
| `memory_timeline` | Chronological timeline of a topic |
| `memory_inspect` | Inspect IR, provenance chain, and versions |

### Manual Start
```bash
# Start the API server
python -m artificial_memory.api.server

# Or use CLI
python -m artificial_memory start "Projects/MyProject"
python -m artificial_memory user "Hello, how are you?"
python -m artificial_memory assistant "I'm doing well, thank you!"
python -m artificial_memory recall "what did we discuss"
python -m artificial_memory end
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `am start <topic>` | Start new conversation |
| `am user <message>` | Log user message |
| `am assistant <message>` | Log assistant message |
| `am end` | End conversation & compile memories |
| `am memory` | Show current memories |
| `am context` | Show LLM context |
| `am recall <query>` | Recall memories |
| `am expand <id> --target N` | Expand memory resolution |
| `am trace <id>` | Trace memory provenance |
| `am temporal_state` | Show temporal state |
| `am confidence` | Check recall confidence |
| `am human_recall` | Human-like recall |
| `am vector_search` | Vector similarity search |
| `am associate` | Analyze semantic associations |
| `am style_profile` | Show conversation style profile |
| `am consolidation_status` | Show consolidation engine status |

## API Endpoints

### Conversations
- `POST /conversations/start` - Start conversation
- `POST /conversations/{id}/messages` - Add message
- `POST /conversations/{id}/end` - End & compile

### Memory & Recall
- `GET /memory?topic_path=...` - List memories
- `POST /recall` - Standard recall
- `POST /recall/human` - Human-like recall
- `POST /recall/explain` - Explain recall process
- `POST /memory/{id}/expand` - Expand resolution

### Context & Vector
- `POST /context` - Build optimized context
- `POST /vector/search` - Vector similarity search
- `POST /vector/hybrid` - Hybrid vector+keyword search
- `GET /vector/stats` - Index statistics

### Temporal
- `POST /temporal/state` - State at timestamp
- `POST /temporal/changes` - Changes between timestamps
- `POST /temporal/timeline` - Topic timeline

### Confidence & Style
- `POST /confidence` - Compute confidence
- `GET /confidence/memory/{id}` - Memory confidence
- `POST /style/profile` - Style profile
- `POST /style/reconstruct` - Reconstruct with style

### Human-like Recall
- `POST /recall/human` - Human-like recall
- `POST /recall/explain` - Explain recall process

### Auth
- `POST /auth/register` - Register user
- `POST /auth/login` - Login
- `GET /auth/me` - Current user
- `POST /auth/api-keys` - Create API key
- `GET /auth/api-keys` - List API keys

### Chat
- `POST /chat` - Chat with LLM
- `POST /chat/stream` - Streaming chat

### Metrics & Admin
- `GET /metrics` - System metrics
- `GET /metrics/export` - Export metrics
- `WS /ws` - WebSocket

## Configuration

Environment variables:
```bash
# Database
DATABASE_PATH=memory.db

# LLM Providers
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=your-key

# Auth
JWT_SECRET=your-secret

# Vector Search
VECTOR_INDEX_PATH=vector_index
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Artificial Memory                        │
│         Cognitive Memory Runtime for Persistent AI         │
├─────────────────────────────────────────────────────────────┤
│  CLI          │  HTTP API        │  WebSocket  │  Web UI   │
├─────────────────────────────────────────────────────────────┤
│                    Runtime Facade                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Recall    │ │   Context   │ │  Confidence │           │
│  │   Runtime   │ │   Runtime   │ │   Engine    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                    Memory Compiler                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Lexical    │ │  Semantic   │ │  Fact/      │           │
│  │  Analysis   │ │  Extraction │ │  Decision   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Episode    │ │  Temporal   │ │  Provenance │           │
│  │  Construction│ │  Linking    │ │  Linking    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│              Memory Evolution Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Consolidation│ │Contradiction│ │   Belief    │           │
│  │  Engine     │ │  Detection  │ │  Engine     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Healing   │ │ Dependency  │ │  Temporal   │           │
│  │  (Integrity)│ │   Graph     │ │  Updates    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                    IR Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │    Memory IR    │  │   Context IR    │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  SQLite  │  FAISS/pgvector  │  File Storage (Markdown/JSON) │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture: Distributed / Kubernetes

```
                    AI Application
                          │
                          ▼
                 Artificial Memory
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  Memory Runtime    Context Runtime    Governance
       │                  │                  │
  Recall             Context             Audit
  Evolution          Allocation          Policy
  Temporal           Provenance          Security
  Healing            Debugging           Tenancy
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                    Storage Layer
              (PostgreSQL / pgvector / Object Store)
                          │
                    Kubernetes Operator
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Recall Workers    Compiler Workers   Vector Index Workers
   Consolidation     Memory Governance   Runtime Nodes
```

**Long-term goal**: Artificial Memory aims to provide a **Kubernetes Operator** for deploying and managing distributed cognitive memory runtimes:

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

## Experiment Framework

Run quantitative evaluations with ablation studies:

```python
from artificial_memory.experiments import ExperimentRunner, ExperimentConfig, ExperimentType

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

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check .

# Type check
mypy src/artificial_memory

# Format
ruff format .
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# With OpenAI
OPENAI_API_KEY=your-key docker-compose up -d

# Production
docker build -t artificial-memory .
docker run -d -p 8000:8000 -v ./data:/app/data artificial-memory
```

## Kubernetes Deployment (Phase 9)

> **⚠️ Scope Disclaimer (V1)**
>
> The Kubernetes Operator provides **deployment and operational orchestration
> primitives** only — CRDs, reconciliation, worker deployment, scaling, and
> lifecycle management. It **does not** constitute proof of distributed memory
> consistency, distributed correctness, or production readiness. Single-node
> (SQLite / single Postgres) operation is the validated path in V1. Distributed
> consistency semantics (idempotency, transaction boundaries, index
> synchronization, recovery) are being designed in Phase 9.5 and must not be
> assumed from the presence of the Operator.

### Quick Start with Helm

```bash
# Add the chart repository (or use local chart)
helm repo add artificial-memory ./k8s/helm/artificial-memory-operator

# Create namespace
kubectl create namespace artificial-memory

# Install with default values
helm install artificial-memory artificial-memory/artificial-memory-operator \
  -n artificial-memory

# Or install with custom values
helm install artificial-memory artificial-memory/artificial-memory-operator \
  -n artificial-memory \
  -f custom-values.yaml
```

### Deploy via kubectl (CRDs + Operator)

```bash
# Install CRDs
kubectl apply -f k8s/crds/

# Install operator
kubectl apply -f k8s/operator/rbac.yaml
kubectl apply -f k8s/operator/deployment.yaml
```

### Create an ArtificialMemoryCluster

```yaml
# cluster.yaml
apiVersion: memory.artificialmemory.dev/v1
kind: ArtificialMemoryCluster
metadata:
  name: my-cluster
  namespace: artificial-memory
spec:
  replicas: 3
  storage:
    backend: postgres
    vector: pgvector
    postgres:
      host: artificial-memory-postgres
      port: 5432
      database: artificial_memory
      secretRef: artificial-memory-postgres-secret
  runtime:
    recallWorkers: 4
    consolidationWorkers: 2
    compilerWorkers: 2
  policy:
    compression: adaptive
    retention: policy-driven
    governance: enabled
```

```bash
kubectl apply -f cluster.yaml
```

### Check Cluster Status

```bash
# Get cluster status
kubectl get artificialmemorycluster -n artificial-memory

# Get all resources
kubectl get amc,ms,mw,rw,vi -n artificial-memory

# Check operator logs
kubectl logs -n artificial-memory -l app.kubernetes.io/component=operator

# Port-forward to access API
kubectl port-forward -n artificial-memory svc/artificial-memory-recall-service 8000:8000
```

### Custom Values

```yaml
# custom-values.yaml
cluster:
  name: production-cluster
  replicas: 5
  runtime:
    recallWorkers: 8
    consolidationWorkers: 4
  storage:
    postgres:
      secretRef: production-postgres-secret

postgresql:
  cnpg:
    instances: 5
    storageSize: 100Gi

monitoring:
  enabled: true

ingress:
  enabled: true
  hosts:
    - host: memory.example.com
      paths:
        - path: /
          pathType: Prefix
```

### Local Development with kind

```bash
# Start local Kubernetes with kind
kind create cluster --name artificial-memory

# Or use docker-compose with kind
docker-compose -f docker-compose.kind.yml up -d

# Deploy to kind
helm install artificial-memory ./k8s/helm/artificial-memory-operator -n artificial-memory --create-namespace
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a PR

## Design Philosophy

Core principles:

1. **Forget = Resolution Down** — Not deletion, but progressive compression
2. **Progressive Recall** — Expand resolution only when needed
3. **Conversation Preservation** — Tone, fillers, hesitation preserved
4. **Context IR** — Dense intermediate representation for LLM
5. **Human-like Recall** — Adaptive resolution based on age/importance
6. **Provenance Tracking** — Every memory traces to source conversation
7. **Evidence ≠ Belief** — Explicit contradiction detection, belief state separate from memory
8. **Memory Evolution** — Memories change; history preserved; healing possible

## Roadmap

### Phase 1 — Foundation (Current)
- [x] Memory IR & Context IR formal definitions
- [x] Legacy Memory ↔ MemoryIR lossless adapter
- [x] Runtime Facade (unified entry point)
- [ ] Deterministic Compiler Pipeline

### Phase 2 — Advanced Memory Runtime
- [ ] Adaptive Recall (utility/token optimization)
- [ ] Memory Evolution Engine (revision, merge, split)
- [ ] Contradiction Detection & Belief State
- [ ] Dependency Graph & Impact Analysis

### Phase 3 — Memory Integrity
- [ ] Integrity Metrics (semantic preservation, temporal consistency)
- [ ] Stale Memory Detection
- [ ] Memory Healing (auto-recompile on corruption)
- [ ] Counterfactual Recall Engine

### Phase 4 — Temporal & Debugging Research
- [ ] Memory Time Travel (state/belief/context at timestamp)
- [ ] Memory Debugger (selection/rejection/expansion explanations)
- [ ] Decision Trace & Impact Analysis

### Phase 5 — Research Platform
- [ ] Benchmark Harness (reproducible experiments)
- [ ] Ablation Framework
- [ ] Red-Team Suite (adversarial testing)
- [ ] AM-Specific Synthetic Benchmarks (temporal, contradiction, false memory, compression loss)

### Phase 6 — Multi-Agent & Enterprise
- [ ] Federated Memory Exchange
- [ ] Trust Policies & Governance
- [ ] Tenant Isolation & Audit
- [ ] Retention Policies

### Phase 7 — Distributed Runtime
- [x] PostgreSQL / pgvector Backend
- [x] Distributed Vector Indexing
- [x] Worker Architecture (queue-based compilation)
- [x] Horizontal Scaling

### Phase 8 — Kubernetes
- [x] Kubernetes Operator (ArtificialMemoryCluster, MemoryStore, MemoryWorker, RecallWorker, VectorIndex)
- [x] Operator-managed: deployment, scaling, storage, config, upgrades, health, migration
- [x] Helm Chart for deployment
- [ ] Integration tests with kind/k3s
- [ ] Production hardening (network policies, pod disruption budgets, priority classes)

## Citation

If you use this in research, please cite:

```
@software{artificial-memory,
  title = {Artificial Memory / Context Runtime},
  subtitle = {A Cognitive Memory Runtime for Persistent AI Systems},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-org/artificial-memory}
}
```