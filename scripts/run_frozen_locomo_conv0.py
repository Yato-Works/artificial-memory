"""AM Apex Phase EXPEDITION: Full LoCoMo-10 Conv 0 Benchmark (199 Questions).

Evaluates the complete 199 questions of LoCoMo-10 Conv 0 with phi4-mini:3.8b
using the 100% Frozen AM Apex configuration:
  - STEROID: Wide Slicing & Adaptive Graph Expansion
  - PROTEIN: Evidence Ranker & Fusion (w_r=0, top-10)
  - STATE COMPILER: Full deterministic graph-path state compilation (VITAMIN V1-V5)
  - IMMUNE: Answer Verifier with State Refusal Guard & Numeric Normalizer

Breaks down performance across all 5 official LoCoMo categories:
  - Category 1: Multi-Hop (32 Qs)
  - Category 2: Temporal (43 Qs)
  - Category 3: Open-Domain (19 Qs)
  - Category 4: Single-Hop (85 Qs)
  - Category 5: Adversarial (20 Qs)

Saves complete telemetry to benchmark_results/frozen/locomo_conv0_frozen.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/frozen")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 90)
    print("      AM APEX PHASE EXPEDITION: FULL LOCOMO CONV 0 BENCHMARK (199 QUESTIONS)")
    print("=" * 90)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
    print(f"Loaded {len(turns)} turns and {len(questions)} questions across 40 sessions.")

    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )
    adapter.compiler = compiler

    results = []
    cat_results = defaultdict(list)
    t0 = time.perf_counter()

    print("\n" + f"{'Q#':<5} | {'Cat':<11} | {'Status':<6} | {'Ora':<4} | {'Tokens':<8} | Question")
    print("-" * 90)

    for i, q in enumerate(questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results.append(res)
        cat_results[q.category].append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora = "YES" if res.oracle_recall else "NO"
        cat_name = LoCoMoAdapter.CATEGORY_NAMES.get(q.category, f"Cat-{q.category}")

        # Live telemetry every 10 questions
        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"[{i+1:03d}/{len(questions):03d}] | {cat_name:<11} | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {q.question[:35]}")
            print(f"       >>> Rolling Acc: {curr_acc:5.1f}% | Rolling Ora: {curr_ora:5.1f}% | Rolling Tok: {curr_tok:5.1f}")
        else:
            print(f"[{i+1:03d}/{len(questions):03d}] | {cat_name:<11} | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {q.question[:35]}")

    elapsed = time.perf_counter() - t0
    total_correct = sum(1 for r in results if r.is_correct)
    total_ora = sum(1 for r in results if r.oracle_recall)
    overall_acc = total_correct / len(results) * 100
    overall_ora = total_ora / len(results) * 100
    overall_mean_tok = sum(r.tokens_used for r in results) / len(results)

    print("\n" + "=" * 90)
    print("          FULL LOCOMO CONV 0 EVALUATION SUMMARY (199 QUESTIONS)")
    print("=" * 90)
    print(f"Total Questions Evaluated:  {len(results)}")
    print(f"Overall Accuracy:           {overall_acc:5.1f}% ({total_correct}/{len(results)})")
    print(f"Overall Oracle Recall:      {overall_ora:5.1f}% ({total_ora}/{len(results)})")
    print(f"Mean Tokens / Question:     {overall_mean_tok:5.1f} tokens")
    print(f"Total Elapsed Time:         {elapsed:5.1f}s ({elapsed/len(results):.2f}s / question)")
    print("=" * 90)

    # Category Breakdown
    print("\n" + "=" * 90)
    print("               CATEGORY-BY-CATEGORY PERFORMANCE BREAKDOWN")
    print("=" * 90)
    print(f"{'Category':<20} | {'Questions':<10} | {'Accuracy':<12} | {'Oracle Recall':<14} | {'Mean Tokens':<12}")
    print("-" * 90)

    category_summary = {}
    for cat_id in sorted(cat_results.keys()):
        c_items = cat_results[cat_id]
        c_name = LoCoMoAdapter.CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")
        c_correct = sum(1 for r in c_items if r.is_correct)
        c_ora = sum(1 for r in c_items if r.oracle_recall)
        c_acc = c_correct / len(c_items) * 100
        c_ora_pct = c_ora / len(c_items) * 100
        c_tok = sum(r.tokens_used for r in c_items) / len(c_items)

        category_summary[c_name] = {
            "category_id": cat_id,
            "num_questions": len(c_items),
            "accuracy": c_acc,
            "num_correct": c_correct,
            "oracle_recall": c_ora_pct,
            "num_oracle": c_ora,
            "mean_tokens": c_tok,
        }
        print(f"{c_name:<20} | {len(c_items):<10} | {c_acc:5.1f}% ({c_correct:2d}/{len(c_items):2d}) | {c_ora_pct:5.1f}% ({c_ora:2d}/{len(c_items):2d})  | {c_tok:5.1f} tok")

    print("=" * 90)

    # Save to file
    out_file = RESULTS_DIR / "locomo_conv0_frozen.json"
    serializable_details = [
        {
            "question_id": r.question_id,
            "category": r.category,
            "category_name": LoCoMoAdapter.CATEGORY_NAMES.get(r.category, f"Cat-{r.category}"),
            "oracle_recall": r.oracle_recall,
            "predicted_answer": r.predicted_answer,
            "ground_truth": r.ground_truth,
            "tokens_used": r.tokens_used,
            "latency_ms": r.latency_ms,
            "is_correct": r.is_correct,
        }
        for r in results
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(results),
                "overall_accuracy": overall_acc,
                "overall_oracle_recall": overall_ora,
                "overall_mean_tokens": overall_mean_tok,
                "elapsed_seconds": elapsed,
                "category_summary": category_summary,
                "details": serializable_details,
            },
            f,
            indent=2,
        )
    print(f"\nSaved full benchmark results to {out_file}")


if __name__ == "__main__":
    main()
