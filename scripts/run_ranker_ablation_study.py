"""Ranker Component Ablation Study for AM Apex Protein Phase.

Systematically measures the individual contribution of each ranking signal:
- P4-Full: All 5 signals active (w_s=1.0, w_e=2.5, w_t=1.5, w_r=1.2, w_g=1.0)
- P4 - Semantic: w_s = 0.0 (No token/lexical overlap)
- P4 - Entity: w_e = 0.0 (No entity/kinship anchor matching)
- P4 - Temporal: w_t = 0.0 (No temporal/recency scoring)
- P4 - Relation: w_r = 0.0 (No action/predicate matching)
- P4 - Graph: w_g = 0.0 (No hop-distance proximity scoring)

Outputs results to benchmark_results/protein/ranker_ablation_study.json.
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


def evaluate_ablation(
    name: str,
    weights: dict[str, float],
    locomo_adapter: LoCoMoAdapter,
    longmem_adapter: LongMemEvalAdapter,
    answerer: OllamaAnswerer,
    locomo_turns: list,
    locomo_qs: list,
    locomo_ir: list,
    longmem_items: list,
) -> dict[str, float]:
    print(f"\n--- Running Ablation: {name} ---")
    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        ranker_weights=weights,
        top_k_evidence=10,
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
        "ablation": name,
        "weights": weights,
        "accuracy": acc,
        "evidence_recall": ora,
        "mean_tokens": mean_tok,
        "latency_ms": lat_per_q,
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 80)
    print("      AM APEX PROTEIN PHASE — EVIDENCE RANKER COMPONENT ABLATION")
    print("=" * 80)

    answerer = OllamaAnswerer()
    locomo_adapter = LoCoMoAdapter()
    longmem_adapter = LongMemEvalAdapter()

    print("Loading datasets...")
    locomo_turns, locomo_qs, locomo_ir = locomo_adapter.load_conversation(conv_idx=0)
    longmem_items = longmem_adapter.load_dataset()
    print("Datasets loaded successfully.\n")

    base_weights = {"w_s": 1.0, "w_e": 2.5, "w_t": 1.5, "w_r": 1.2, "w_g": 1.0}

    ablation_configs = {
        "P4-Full (All Signals)": base_weights.copy(),
        "P4 - Semantic (w_s=0)": {**base_weights, "w_s": 0.0},
        "P4 - Entity (w_e=0)": {**base_weights, "w_e": 0.0},
        "P4 - Temporal (w_t=0)": {**base_weights, "w_t": 0.0},
        "P4 - Relation (w_r=0)": {**base_weights, "w_r": 0.0},
        "P4 - Graph (w_g=0)": {**base_weights, "w_g": 0.0},
    }

    results = {}
    for name, weights in ablation_configs.items():
        results[name] = evaluate_ablation(
            name, weights, locomo_adapter, longmem_adapter, answerer,
            locomo_turns, locomo_qs, locomo_ir, longmem_items
        )

    print("\n" + "=" * 85)
    print("                    EVIDENCE RANKER ABLATION SUMMARY")
    print("=" * 85)
    print(f"{'Ablation':<28} | {'Accuracy':<10} | {'Recall':<10} | {'Tokens/Q':<10} | {'Delta Acc':<10}")
    print("-" * 85)
    full_acc = results["P4-Full (All Signals)"]["accuracy"]
    for name, r in results.items():
        delta = r["accuracy"] - full_acc
        delta_str = f"{delta:+5.1f}pt" if name != "P4-Full (All Signals)" else "(baseline)"
        print(f"{name:<28} | {r['accuracy']:5.1f}%    | {r['evidence_recall']:5.1f}%    | {r['mean_tokens']:5.1f} tok   | {delta_str}")
    print("=" * 85)

    out_file = RESULTS_DIR / "ranker_ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
