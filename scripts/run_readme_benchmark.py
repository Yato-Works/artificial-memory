"""README benchmark: deterministic, LLM-free system benchmarks.

Runs the full ArtificialMemoryRuntime facade against a synthetic dataset
(fixed seed) and produces README-ready metrics:

- remember throughput (writes/s)
- vector index build time
- semantic recall accuracy@1 / @5 (paraphrased queries)
- recall latency p50 / p95
- token savings vs naive full-context injection
- compression ratio (rule-based compressor)

Usage:
    python scripts/run_readme_benchmark.py [--memories 120] [--queries 50]

Output:
    benchmark_results/readme_benchmark.json
    benchmark_results/README_BENCHMARK.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from artificial_memory.compression.compressor import RuleBasedCompressor  # noqa: E402
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig  # noqa: E402
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore  # noqa: E402

_T0_GLOBAL = time.perf_counter()

# ---------------------------------------------------------------- synthetic data

_ADJECTIVES = ["legacy", "modern", "distributed", "experimental", "critical", "internal"]
_NOUNS = ["service", "pipeline", "dashboard", "cluster", "module", "agent"]
_TECHS = ["PostgreSQL", "SQLite", "Redis", "Kafka", "gRPC", "REST"]
_REASONS = [
    "horizontal scaling is required",
    "the team already knows it well",
    "latency SLOs are strict",
    "operational cost must stay low",
    "multi-region replication matters",
    "the ecosystem support is mature",
]


def _make_dataset(n: int, rng: random.Random) -> list[dict]:
    """Generate n synthetic memories, each with a distinct key fact."""
    cases = []
    for i in range(n):
        adj = rng.choice(_ADJECTIVES)
        noun = rng.choice(_NOUNS)
        tech = _TECHS[i % len(_TECHS)]
        reason = _REASONS[i % len(_REASONS)]
        entity = f"{adj} {noun} #{i:04d}"
        content = (
            f"Architecture decision: the {entity} must use {tech}. "
            f"The reason is that {reason}. This decision was reviewed "
            f"and approved by the platform team. The rollout plan, the "
            f"migration risk review, and the rollback strategy were all "
            f"documented in the platform wiki before the change window, "
            f"and the on-call rotation was briefed about the new setup."
        )
        query = f"Which technology does the {entity} use?"
        cases.append({"entity": entity, "content": content, "query": query})
    return cases


def _make_diverse_dataset(n: int, rng: random.Random) -> list[dict]:
    """Generate n memories from distinct domains (realistic retrieval setup)."""
    codewords = ["Aurora", "Basalt", "Cinder", "Dynamo", "Ember", "Fathom", "Glacier",
                 "Halcyon", "Iris", "Jigsaw", "Kestrel", "Lumen", "Mosaic", "Nimbus"]
    components = ["billing module", "auth service", "ETL pipeline", "search indexer",
                  "notification queue", "cache layer", "report engine", "sync worker"]
    stacks = ["PostgreSQL", "Kafka", "Redis", "MongoDB", "DynamoDB", "RabbitMQ",
              "Elasticsearch", "ClickHouse"]
    benefits = ["regional failover", "write throughput", "cost efficiency",
                "query latency", "operational simplicity", "compliance needs"]
    cases = []
    for i in range(n):
        codeword = codewords[i % len(codewords)] + f"-{i // len(codewords):02d}"
        component = rng.choice(components)
        stack = rng.choice(stacks)
        benefit = rng.choice(benefits)
        content = (
            f"Project {codeword} migrated its {component} to {stack} "
            f"during the last quarter to improve {benefit}. The rollout "
            f"was coordinated with the platform team, a rollback plan "
            f"was documented before the cutover window, and the on-call "
            f"rotation was briefed on the new operational characteristics."
        )
        query = f"What stack does the {codeword} project use for its {component}?"
        cases.append({"content": content, "query": query})
    return cases


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
    return s[idx]

# ---------------------------------------------------------------- benchmark

async def _bench_dataset(
    am: ArtificialMemoryRuntime,
    engine,
    cases: list[dict],
    topic_name: str,
    n_queries: int,
    compressor: RuleBasedCompressor,
) -> dict:
    """Benchmark one dataset end-to-end and return its metrics."""
    store = am.store
    topic = am.conversation_manager.get_or_create_topic(topic_name)
    topic_id = topic.id

    # ---- write throughput ----
    write_times: list[float] = []
    contents: list[str] = []
    for case in cases:
        t0 = time.perf_counter()
        await am.remember(case["content"], topic=topic_name)
        write_times.append((time.perf_counter() - t0) * 1000)
        contents.append(case["content"])

    # ---- vector index build ----
    memories = [m for m in store.get_memories(limit=10000) if m.topic_id == topic_id]
    by_content = {m.content: m.id for m in memories}
    t0 = time.perf_counter()
    for m in memories:
        engine.add_memory(m)
    index_build_ms = (time.perf_counter() - t0) * 1000

    # ---- recall accuracy @1 / @5 (paraphrased queries) ----
    hits1 = hits5 = hits1_h = hits5_h = 0
    for case in cases:
        target = by_content.get(case["content"])
        if target is None:
            continue
        ids = [r.memory_id for r in engine.search(case["query"], topic_id=topic_id, k=5)]
        if ids and ids[0] == target:
            hits1 += 1
        if target in ids:
            hits5 += 1
        hids = [
            r.memory_id
            for r in engine.hybrid_search(case["query"], topic_id=topic_id, k=5)
        ]
        if hids and hids[0] == target:
            hits1_h += 1
        if target in hids:
            hits5_h += 1

    # ---- recall latency via facade ----
    latencies: list[float] = []
    tokens_am: list[int] = []
    for case in cases[:n_queries]:
        t0 = time.perf_counter()
        rr = await am.recall(case["query"], topic=topic_name, level=2, max_tokens=4000)
        latencies.append((time.perf_counter() - t0) * 1000)
        tokens_am.append(rr.tokens)

    # ---- token savings vs naive full-context ----
    full_ctx = sum(compressor.count_tokens(c) for c in contents)
    avg_tokens = statistics.mean(tokens_am) if tokens_am else 0

    # ---- compression savings (light / semantic) ----
    light_savings: list[float] = []
    semantic_savings: list[float] = []
    for content in contents[:50]:
        orig = compressor.count_tokens(content)
        if orig > 0:
            light_c, _ = compressor.compress_light(content, {})
            semantic_c, _ = compressor.compress_semantic(content, {})
            light_savings.append(1 - compressor.count_tokens(light_c) / orig)
            semantic_savings.append(1 - compressor.count_tokens(semantic_c) / orig)

    return {
        "n": len(cases),
        "remember_throughput_writes_per_s": round(len(cases) / (sum(write_times) / 1000), 2),
        "vector_index_build_ms": round(index_build_ms, 2),
        "recall_accuracy_at_1": round(hits1 / len(cases), 4),
        "recall_accuracy_at_5": round(hits5 / len(cases), 4),
        "hybrid_accuracy_at_1": round(hits1_h / len(cases), 4),
        "hybrid_accuracy_at_5": round(hits5_h / len(cases), 4),
        "recall_latency_p50_ms": round(_pct(latencies, 0.50), 2),
        "recall_latency_p95_ms": round(_pct(latencies, 0.95), 2),
        "full_context_tokens": full_ctx,
        "am_avg_recall_tokens": round(avg_tokens, 1),
        "token_savings_ratio": round(1 - avg_tokens / full_ctx, 4) if full_ctx else 0.0,
        "compression_savings_light": round(statistics.mean(light_savings), 4)
        if light_savings else 0.0,
        "compression_savings_semantic": round(statistics.mean(semantic_savings), 4)
        if semantic_savings else 0.0,
    }


async def run_benchmark(n_memories: int, n_queries: int, seed: int) -> dict:
    rng = random.Random(seed)
    stress_cases = _make_dataset(n_memories, rng)
    diverse_cases = _make_diverse_dataset(n_memories, rng)

    tmp = Path(tempfile.mkdtemp(prefix="am_bench_"))
    cfg = RuntimeConfig(
        database_path=str(tmp / "bench.db"),
        memory_files_path=str(tmp / "memory_files"),
        vector_index_path=str(tmp / "vector_index"),
        audit_log_dir=str(tmp / "audit_logs"),
        metrics_output_dir=str(tmp / "metrics"),
    )
    am = ArtificialMemoryRuntime(cfg)
    store: SQLiteMemoryStore = am.store  # type: ignore[assignment]
    model_load_s = time.perf_counter() - _T0_GLOBAL
    result: dict = {"meta": {}, "metrics": {}}

    try:
        stress = await _bench_dataset(
            am, am.vector_search_engine, stress_cases, "Benchmarks/stress",
            n_queries, RuleBasedCompressor(),
        )
        diverse = await _bench_dataset(
            am, am.vector_search_engine, diverse_cases, "Benchmarks/diverse",
            n_queries, RuleBasedCompressor(),
        )
        result["metrics"] = {
            **{f"stress_{k}": v for k, v in stress.items() if k != "n"},
            **{f"diverse_{k}": v for k, v in diverse.items() if k != "n"},
        }
        result["meta"] = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "n_memories": n_memories,
            "n_queries": n_queries,
            "runtime_init_and_model_load_s": round(model_load_s, 2),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
        }
    finally:
        am.close()
        shutil.rmtree(tmp, ignore_errors=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memories", type=int, default=120)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    print(f"Running README benchmark: {args.memories} memories, {args.queries} queries...")
    result = asyncio.run(run_benchmark(args.memories, args.queries, args.seed))

    json_path = out_dir / "readme_benchmark.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["metrics"]
    meta = result["meta"]
    md = f"""# Artificial Memory — README Benchmark

> Forget by compression. Recall by resolution. Reason with provenance.

| Metric | Diverse domains | Stress (near-duplicate) |
|---|---|---|
| Semantic recall accuracy@1 (vector / hybrid) | **{m['diverse_recall_accuracy_at_1'] * 100:.1f}% / {m['diverse_hybrid_accuracy_at_1'] * 100:.1f}%** | {m['stress_recall_accuracy_at_1'] * 100:.1f}% / {m['stress_hybrid_accuracy_at_1'] * 100:.1f}% |
| Semantic recall accuracy@5 (vector / hybrid) | **{m['diverse_recall_accuracy_at_5'] * 100:.1f}% / {m['diverse_hybrid_accuracy_at_5'] * 100:.1f}%** | {m['stress_recall_accuracy_at_5'] * 100:.1f}% / {m['stress_hybrid_accuracy_at_5'] * 100:.1f}% |
| Recall latency p50 / p95 | {m['diverse_recall_latency_p50_ms']:.0f} / {m['diverse_recall_latency_p95_ms']:.0f} ms | {m['stress_recall_latency_p50_ms']:.0f} / {m['stress_recall_latency_p95_ms']:.0f} ms |
| Token savings vs full-context | **{m['diverse_token_savings_ratio'] * 100:.1f}%** | {m['stress_token_savings_ratio'] * 100:.1f}% |
| Vector index build | {m['diverse_vector_index_build_ms']:.0f} ms | {m['stress_vector_index_build_ms']:.0f} ms |
| Remember throughput | {m['diverse_remember_throughput_writes_per_s']:.1f} writes/s | {m['stress_remember_throughput_writes_per_s']:.1f} writes/s |

*Seed {meta['seed']}, {meta['n_memories']} synthetic memories and {meta['n_queries']} queries per dataset.
"diverse" = memories from distinct domains (realistic retrieval), "stress" = near-identical templates (adversarial).
Python {meta['python']} on {meta['platform']} ({meta['cpu']}). Embeddings: all-MiniLM-L6-v2 (384-d, FAISS IndexFlatIP).*
"""
    md_path = out_dir / "README_BENCHMARK.md"
    md_path.write_text(md, encoding="utf-8")

    print(json.dumps(result["metrics"], indent=2))
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

