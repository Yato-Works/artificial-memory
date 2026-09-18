"""LLM benchmark: QA accuracy with/without Artificial Memory (local Ollama models).

Compares three answering conditions against the same memory store:

  1. no_memory    - LLM answers from its own knowledge only (no context)
  2. full_context - ALL memories injected into the prompt
  3. am_recall    - only memories retrieved via ArtificialMemoryRuntime.recall()

The synthetic facts (project codenames + tech stacks) are unknowable without
memory, so the gap between conditions 1 and 2/3 measures how much the memory
runtime contributes, and the token counts measure what it costs.

Usage:
    python scripts/run_llm_benchmark.py [--models qwen2.5:1.5b,qwen3:4b]
                                        [--questions 20] [--seed 42]

Output:
    benchmark_results/llm_benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_readme_benchmark import _make_diverse_dataset  # noqa: E402

from artificial_memory.compression.compressor import RuleBasedCompressor  # noqa: E402
from artificial_memory.llm.manager import OllamaProvider  # noqa: E402
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_WITH_CONTEXT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    'provided context. If the context does not contain the answer, say "I '
    "don't know.\" Answer in one short sentence."
)

SYSTEM_NO_MEMORY = (
    "You are a precise assistant. You have no notes and no context. If you do "
    "not know the answer from your own knowledge, say \"I don't know.\" "
    "Answer in one short sentence."
)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


_STACKS = ["PostgreSQL", "Kafka", "Redis", "MongoDB", "DynamoDB", "RabbitMQ",
           "Elasticsearch", "ClickHouse"]


async def _ask(provider: OllamaProvider, question: str, context: str | None,
               compressor: RuleBasedCompressor) -> tuple[str, float, int]:
    """Ask one question; returns (answer, latency_ms, prompt_tokens)."""
    if context is None:
        system = SYSTEM_NO_MEMORY
        user = question
    else:
        system = SYSTEM_WITH_CONTEXT
        user = f"Context:\n{context}\n\nQuestion: {question}"
    prompt_tokens = compressor.count_tokens(system) + compressor.count_tokens(user)
    t0 = time.perf_counter()
    raw = await provider.generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return _strip_think(raw), latency_ms, prompt_tokens


async def run(models: list[str], n_questions: int, seed: int) -> dict:
    rng = __import__("random").Random(seed)
    cases = _make_diverse_dataset(n_questions, rng)

    tmp = Path(tempfile.mkdtemp(prefix="am_llm_bench_"))
    cfg = RuntimeConfig(
        database_path=str(tmp / "llm.db"),
        memory_files_path=str(tmp / "memory_files"),
        vector_index_path=str(tmp / "vector_index"),
        audit_log_dir=str(tmp / "audit_logs"),
        metrics_output_dir=str(tmp / "metrics"),
    )
    am = ArtificialMemoryRuntime(cfg)
    store = am.store
    topic = "LlmBenchmark/qa"

    # Populate memory + index
    by_content = {}
    for case in cases:
        await am.remember(case["content"], topic=topic)
    engine = am.vector_search_engine
    for m in store.get_memories(limit=10000):
        engine.add_memory(m)
        by_content[m.content] = m.id

    compressor = RuleBasedCompressor()
    all_contents = [c["content"] for c in cases]
    full_context = "\n".join(all_contents)

    results: dict = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "n_questions": n_questions,
            "n_memories": len(cases),
            "full_context_tokens": compressor.count_tokens(full_context),
        },
        "models": {},
    }

    try:
        for model_name in models:
            # Thinking models (qwen3, deepseek-r1, ...) return empty content
            # unless thinking is disabled via the Ollama API flag.
            is_thinker = any(t in model_name for t in ("qwen3", "deepseek-r1"))
            provider = OllamaProvider(model=model_name, think=False if is_thinker else None)
            model_res: dict = {}
            for condition in ("no_memory", "full_context", "am_recall"):
                correct = 0
                latencies: list[float] = []
                prompt_tokens_list: list[int] = []
                details = []
                for case in cases:
                    expected = case["content"].split(" to ")[-1].split(" during")[0].strip()
                    # expected is the stack name inside the content
                    expected = next(
                        (s for s in _STACKS if s in case["content"]), expected
                    )
                    if condition == "no_memory":
                        context = None
                    elif condition == "full_context":
                        context = full_context
                    else:
                        rr = await am.recall(
                            case["query"], topic=topic, level=2, max_tokens=600
                        )
                        recalled = [
                            m.semantic_content.content for m in rr.memories
                        ]
                        context = "\n".join(recalled) if recalled else "(no memories found)"
                        recall_tokens = rr.tokens
                    answer, latency_ms, p_tokens = await _ask(
                        provider, case["query"], context, compressor
                    )
                    hit = expected.lower() in answer.lower()
                    if hit:
                        correct += 1
                    latencies.append(latency_ms)
                    prompt_tokens_list.append(p_tokens)
                    details.append({
                        "q": case["query"],
                        "expected": expected,
                        "answer": answer[:160],
                        "correct": hit,
                        **({"recall_tokens": recall_tokens} if condition == "am_recall" else {}),
                    })
                model_res[condition] = {
                    "accuracy": round(correct / len(cases), 4),
                    "avg_latency_ms": round(statistics.mean(latencies), 1),
                    "avg_prompt_tokens": round(statistics.mean(prompt_tokens_list), 1),
                    "details": details,
                }
                recall_tokens_list = [
                    d["recall_tokens"] for d in details if "recall_tokens" in d
                ]
                if recall_tokens_list:
                    model_res[condition]["avg_recall_tokens"] = round(
                        statistics.mean(recall_tokens_list), 1
                    )
                print(
                    f"  [{model_name}] {condition}: "
                    f"acc={model_res[condition]['accuracy'] * 100:.1f}% "
                    f"tokens={model_res[condition]['avg_prompt_tokens']:.0f} "
                    f"lat={model_res[condition]['avg_latency_ms']:.0f}ms"
                )
            results["models"][model_name] = model_res
    finally:
        am.close()
        shutil.rmtree(tmp, ignore_errors=True)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=str,
        default="qwen2.5:1.5b,qwen3:4b",
        help="Comma-separated Ollama model names",
    )
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    print(f"LLM benchmark: {len(models)} models x 3 conditions x {args.questions} questions")
    results = asyncio.run(run(models, args.questions, args.seed))

    json_path = out_dir / "llm_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for model_name, conds in results["models"].items():
        print(f"\n{model_name}:")
        for cond, m in conds.items():
            print(
                f"  {cond:13s} acc={m['accuracy'] * 100:5.1f}%  "
                f"prompt_tokens={m['avg_prompt_tokens']:6.0f}  "
                f"lat={m['avg_latency_ms']:6.0f}ms"
            )
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


