"""AM Apex Phase EXPEDITION: LongMemEval 50Q Frozen Validation.

Evaluates 50 questions of LongMemEval with phi4-mini:3.8b using the
exact same Frozen AM Apex configuration (ProteinContextCompiler + StateCompiler + AnswerVerifier).

Verifies that the new state compilation and answer guard mechanisms do not introduce
any regressions on the previously established 78.0% frozen baseline.

Saves results to benchmark_results/frozen/longmemeval_50_frozen.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/frozen")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 85)
    print("      AM APEX PHASE EXPEDITION: LONGMEMEVAL 50Q FROZEN VALIDATION")
    print("=" * 85)

    adapter = LongMemEvalAdapter()
    answerer = OllamaAnswerer()

    items = adapter.load_dataset()
    target_items = items[:50]
    print(f"Loaded {len(items)} total LongMemEval items. Evaluating first {len(target_items)} items.")

    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )
    adapter.compiler = compiler

    results = []
    type_results = defaultdict(list)
    t0 = time.perf_counter()

    print("\n" + f"{'#':<3} | {'Type':<18} | {'Status':<6} | {'Ora':<4} | {'Tokens':<8} | Question")
    print("-" * 85)

    for i, item in enumerate(target_items):
        res = adapter.evaluate_item(item, answerer)
        results.append(res)
        type_results[item.question_type].append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora = "YES" if res.oracle_recall else "NO"

        if (i + 1) % 10 == 0 or (i + 1) == len(target_items):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"[{i+1:02d}/50] | {item.question_type:<18} | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {item.question[:32]}")
            print(f"       >>> Rolling Acc: {curr_acc:5.1f}% | Rolling Ora: {curr_ora:5.1f}% | Rolling Tok: {curr_tok:5.1f}")
        else:
            print(f"[{i+1:02d}/50] | {item.question_type:<18} | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {item.question[:32]}")

    elapsed = time.perf_counter() - t0
    total_correct = sum(1 for r in results if r.is_correct)
    total_ora = sum(1 for r in results if r.oracle_recall)
    overall_acc = total_correct / len(results) * 100
    overall_ora = total_ora / len(results) * 100
    overall_mean_tok = sum(r.tokens_used for r in results) / len(results)

    print("\n" + "=" * 85)
    print("          LONGMEMEVAL 50Q FROZEN EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Questions Evaluated:      {len(results)}")
    print(f"Overall Accuracy:         {overall_acc:5.1f}% ({total_correct}/{len(results)})")
    print(f"Oracle Recall:            {overall_ora:5.1f}% ({total_ora}/{len(results)})")
    print(f"Mean Tokens / Question:   {overall_mean_tok:5.1f} tokens")
    print(f"Elapsed Time:             {elapsed:.1f}s ({elapsed/len(results):.2f}s / question)")
    print(f"Frozen Baseline Ref:      78.0% (39/50)")
    print(f"Delta vs Baseline:        {overall_acc - 78.0:+5.1f} pt")
    print("=" * 85)

    # Type Breakdown
    print("\n" + "=" * 85)
    print("               QUESTION TYPE PERFORMANCE BREAKDOWN")
    print("=" * 85)
    print(f"{'Question Type':<25} | {'Count':<6} | {'Accuracy':<12} | {'Oracle Recall':<14}")
    print("-" * 85)

    type_summary = {}
    for q_type in sorted(type_results.keys()):
        t_items = type_results[q_type]
        t_corr = sum(1 for r in t_items if r.is_correct)
        t_ora = sum(1 for r in t_items if r.oracle_recall)
        t_acc = t_corr / len(t_items) * 100
        t_ora_pct = t_ora / len(t_items) * 100
        type_summary[q_type] = {
            "num_questions": len(t_items),
            "accuracy": t_acc,
            "num_correct": t_corr,
            "oracle_recall": t_ora_pct,
            "num_oracle": t_ora,
        }
        print(f"{q_type:<25} | {len(t_items):<6} | {t_acc:5.1f}% ({t_corr:2d}/{len(t_items):2d}) | {t_ora_pct:5.1f}% ({t_ora:2d}/{len(t_items):2d})")
    print("=" * 85)

    # Save to file
    out_file = RESULTS_DIR / "longmemeval_50_frozen.json"
    serializable_details = [
        {
            "question_id": r.question_id,
            "question_type": r.question_type,
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
                "baseline_accuracy": 78.0,
                "delta_vs_baseline": overall_acc - 78.0,
                "type_summary": type_summary,
                "details": serializable_details,
            },
            f,
            indent=2,
        )
    print(f"\nSaved LongMemEval results to {out_file}")


if __name__ == "__main__":
    main()
