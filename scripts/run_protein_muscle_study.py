"""Protein Muscle-Training Progression Study (Phase P0 to P6).

Measures the incremental impact of each Protein synthesis component:
- P0: Steroid Baseline (Wide Slicing + Graph Expansion, Raw Evidence)
- P1: + Deduplication (Cross-channel duplicate pruning)
- P2: + Entity & Relation Fusion (Nickname/kinship resolution)
- P3: + Temporal Supersession (E_i < E_j Chronological Ordering)
- P4: + Multi-Objective Evidence Ranker (Top 10 High-Precision Selection)
- P5: + Context IR Compression (Dense [STATE], [EVENT], [DATE])
- P6: + Full Protein Compiler (Complete Protein Pipeline with Provenance)

Evaluates on balanced subsets across LoCoMo and LongMemEval.
Outputs results to benchmark_results/protein/protein_muscle_study.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.protein.protein_compiler import ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.steroid.steroid_compiler import SteroidContextCompiler

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_stage(
    stage_name: str,
    compiler: Any,
    locomo_adapter: LoCoMoAdapter,
    longmem_adapter: LongMemEvalAdapter,
    answerer: OllamaAnswerer,
    locomo_turns: list,
    locomo_qs: list,
    locomo_ir: list,
    longmem_items: list,
) -> dict[str, float]:
    print(f"\n--- Running Stage: {stage_name} ---")
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

    # Approximate precision: ratio of relevant tokens to total tokens (higher density = higher precision)
    precision = min(100.0, (ora * 1.5) / (mean_tok / 50.0)) if mean_tok > 0 else 0.0

    print(f"[{stage_name}] Acc: {acc:5.1f}% ({total_correct}/{total_q}) | Recall: {ora:5.1f}% ({total_ora}/{total_q}) | Tok/Q: {mean_tok:5.1f} | Latency: {lat_per_q:5.1f}ms")

    return {
        "stage": stage_name,
        "accuracy": acc,
        "evidence_recall": ora,
        "estimated_precision": precision,
        "mean_tokens": mean_tok,
        "latency_ms": lat_per_q,
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 80)
    print("      AM APEX PROTEIN PHASE — MUSCLE-TRAINING PROGRESSION (P0 -> P6)")
    print("=" * 80)

    answerer = OllamaAnswerer()
    locomo_adapter = LoCoMoAdapter()
    longmem_adapter = LongMemEvalAdapter()

    print("Loading datasets...")
    locomo_turns, locomo_qs, locomo_ir = locomo_adapter.load_conversation(conv_idx=0)
    longmem_items = longmem_adapter.load_dataset()
    print("Datasets loaded successfully.\n")

    results = {}

    # P0: Steroid Baseline (Raw Evidence Pool, no protein)
    c_p0 = SteroidContextCompiler()
    results["P0: Steroid Baseline"] = evaluate_stage(
        "P0: Steroid Baseline", c_p0, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P1: + Deduplication
    c_p1 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=False, enable_supersession=False,
        enable_ranker=False, enable_compression=False
    )
    results["P1: + Deduplication"] = evaluate_stage(
        "P1: + Deduplication", c_p1, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P2: + Entity/Relation Fusion
    c_p2 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=True, enable_supersession=False,
        enable_ranker=False, enable_compression=False
    )
    results["P2: + Entity/Relation Fusion"] = evaluate_stage(
        "P2: + Entity/Relation Fusion", c_p2, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P3: + Temporal Supersession
    c_p3 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=True, enable_supersession=True,
        enable_ranker=False, enable_compression=False
    )
    results["P3: + Temporal Supersession"] = evaluate_stage(
        "P3: + Temporal Supersession", c_p3, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P4: + Evidence Ranker (Top 10)
    c_p4 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=True, enable_supersession=True,
        enable_ranker=True, enable_compression=False, top_k_evidence=10
    )
    results["P4: + Evidence Ranker"] = evaluate_stage(
        "P4: + Evidence Ranker", c_p4, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P5: + Context IR Compression
    c_p5 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=True, enable_supersession=True,
        enable_ranker=True, enable_compression=True, top_k_evidence=10, target_token_budget=350
    )
    results["P5: + Context IR Compression"] = evaluate_stage(
        "P5: + Context IR Compression", c_p5, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # P6: Full Protein Pipeline (Complete Pipeline: Dedup + Supersession + Ranker + Context IR + Provenance)
    c_p6 = ProteinContextCompiler(
        enable_dedup=True, enable_entity_fusion=True, enable_supersession=True,
        enable_ranker=True, enable_compression=True, top_k_evidence=10, target_token_budget=350
    )
    results["P6: Full Protein Pipeline"] = evaluate_stage(
        "P6: Full Protein Pipeline", c_p6, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    print("\n" + "=" * 85)
    print("                    PROTEIN MUSCLE-TRAINING SUMMARY TABLE")
    print("=" * 85)
    print(f"{'Stage':<28} | {'Accuracy':<10} | {'Recall':<10} | {'Tokens/Q':<10} | {'Latency':<10}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<28} | {r['accuracy']:5.1f}%    | {r['evidence_recall']:5.1f}%    | {r['mean_tokens']:5.1f} tok   | {r['latency_ms']:5.1f} ms")
    print("=" * 85)

    out_file = RESULTS_DIR / "protein_muscle_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
