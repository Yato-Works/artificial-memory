"""Top-K Context Budget Sweep Study for AM Apex Protein Phase.

Evaluates how varying the evidence capacity K in {4, 8, 10, 12, 16, 20}
affects Evidence Recall, Downstream Accuracy, and Context Token Cost.

Outputs results to benchmark_results/protein/topk_sweep_study.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_top_k(
    k: int,
    locomo_adapter: LoCoMoAdapter,
    longmem_adapter: LongMemEvalAdapter,
    answerer: OllamaAnswerer,
    locomo_turns: list,
    locomo_qs: list,
    locomo_ir: list,
    longmem_items: list,
) -> dict[str, float]:
    name = f"Top-{k} Evidence"
    print(f"\n--- Running: {name} ---")
    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=k,
    )
    locomo_adapter.compiler = compiler
    longmem_adapter.compiler = compiler

    t0 = time.perf_counter()

    # 1. LoCoMo subset (10 Multi-Hop + 10 Temporal)
    test_qs = [q for q in locomo_qs if q.category in [1, 2]][:20]
    locomo_correct = 0
    locomo_ora = 0
    locomo_tokens = 0

    for q in test_qs:
        res = locomo_adapter.evaluate_question(q, locomo_turns, locomo_ir, answerer)
        if res.is_correct:
            locomo_correct += 1
        if res.oracle_recall:
            locomo_ora += 1
        locomo_tokens += res.tokens_used

    # 2. LongMemEval subset (10 User + 10 Temporal)
    test_items = [it for it in longmem_items if it.question_type in ["single-session-user", "temporal-reasoning"]][:20]
    longmem_correct = 0
    longmem_ora = 0
    longmem_tokens = 0

    for it in test_items:
        res = longmem_adapter.evaluate_item(it, answerer)
        if res.is_correct:
            longmem_correct += 1
        if res.oracle_recall:
            longmem_ora += 1
        longmem_tokens += res.tokens_used

    total_q = len(test_qs) + len(test_items)
    total_correct = locomo_correct + longmem_correct
    total_ora = locomo_ora + longmem_ora
    mean_tok = (locomo_tokens + longmem_tokens) / total_q if total_q else 0.0

    acc = total_correct / total_q * 100.0 if total_q else 0.0
    ora = total_ora / total_q * 100.0 if total_q else 0.0
    elapsed = time.perf_counter() - t0
    lat_per_q = (elapsed / total_q * 1000.0) if total_q else 0.0

    print(f"[{name}] Acc: {acc:5.1f}% ({total_correct}/{total_q}) | Recall: {ora:5.1f}% ({total_ora}/{total_q}) | Tok/Q: {mean_tok:5.1f} | Latency: {lat_per_q:5.1f}ms")

    return {
        "k": k,
        "name": name,
        "accuracy": acc,
        "evidence_recall": ora,
        "mean_tokens": mean_tok,
        "latency_ms": lat_per_q,
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 80)
    print("      AM APEX PROTEIN PHASE — TOP-K CONTEXT BUDGET SWEEP")
    print("=" * 80)

    answerer = OllamaAnswerer()
    locomo_adapter = LoCoMoAdapter()
    longmem_adapter = LongMemEvalAdapter()

    print("Loading datasets...")
    locomo_turns, locomo_qs, locomo_ir = locomo_adapter.load_conversation(conv_idx=0)
    longmem_items = longmem_adapter.load_dataset()
    print("Datasets loaded successfully.\n")

    k_values = [4, 8, 10, 12, 16, 20]
    results = {}

    for k in k_values:
        results[f"Top-{k}"] = evaluate_top_k(
            k, locomo_adapter, longmem_adapter, answerer,
            locomo_turns, locomo_qs, locomo_ir, longmem_items
        )

    print("\n" + "=" * 85)
    print("                    TOP-K CONTEXT PARETO SWEEP SUMMARY")
    print("=" * 85)
    print(f"{'K Value':<15} | {'Accuracy':<10} | {'Recall':<10} | {'Tokens/Q':<12} | {'Latency':<10}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<15} | {r['accuracy']:5.1f}%    | {r['evidence_recall']:5.1f}%    | {r['mean_tokens']:6.1f} tok   | {r['latency_ms']:5.1f} ms")
    print("=" * 85)

    out_file = RESULTS_DIR / "topk_sweep_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
