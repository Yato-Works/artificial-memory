"""AM Apex Phase CHRONOS: Temporal Compiler Benchmark (37Q on phi4-mini:3.8b).

Evaluates the 37 Category 2 (Temporal) questions of LoCoMo-10 Conv 0 with phi4-mini:3.8b
using the Temporal Compiler (時間認識器官):
  - Relative-to-Absolute Calendar Anchoring
  - Provenance-grounded [TEMPORAL STATE]
  - TemporalAnchorNormalizer in AnswerVerifier

Target Gates:
  - Baseline: 10.8% ( 4/37) | 411.7 tok
  - Gate 1:   50.0% (19/37) | +39.2 pt
  - Gate 2:   67.6% (25/37) | +56.8 pt
  - Gate 3:   78.4% (29/37) | +67.6 pt (Oracle Recall Summit)

Saves results to benchmark_results/temporal/temporal_compiler_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/temporal")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 85)
    print("      AM APEX PHASE CHRONOS: TEMPORAL COMPILER BENCHMARK (37Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    temporal_questions = [q for q in all_questions if q.category == 2]
    print(f"Targeting {len(temporal_questions)} Category 2 (Temporal) questions.")

    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )
    adapter.compiler = compiler

    results = []
    t0 = time.perf_counter()

    print("\n" + f"{'Q#':<4} | {'Status':<6} | {'Ora':<4} | {'Tokens':<8} | Question")
    print("-" * 85)

    for i, q in enumerate(temporal_questions):
        t_states = compiler.temporal_compiler.compile_temporal_states(q.question, ir_records)
        state_str = t_states[0].format_state() if t_states else "NONE"

        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results.append({
            "question_id": q.question_id,
            "question": q.question,
            "ground_truth": q.ground_truth,
            "predicted_answer": res.predicted_answer,
            "is_correct": res.is_correct,
            "oracle_recall": res.oracle_recall,
            "tokens_used": res.tokens_used,
            "compiled_temporal_state": state_str,
            "has_temporal_state": bool(t_states),
        })

        status = "PASS" if res.is_correct else "FAIL"
        ora = "YES" if res.oracle_recall else "NO"
        print(f"[{i+1:02d}]  | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | {q.question[:45]}")
        if res.is_correct:
            print(f"       >>> PRED: {res.predicted_answer}")
            if t_states:
                print(f"       >>> STATE: {state_str}")
        else:
            print(f"       --- FAIL | GT: {q.ground_truth} | PRED: {res.predicted_answer}")
            if t_states:
                print(f"       --- STATE: {state_str}")

    elapsed = time.perf_counter() - t0
    num_correct = sum(1 for r in results if r["is_correct"])
    num_ora = sum(1 for r in results if r["oracle_recall"])
    acc = num_correct / len(results) * 100
    ora_pct = num_ora / len(results) * 100
    mean_tok = sum(r["tokens_used"] for r in results) / len(results)
    state_cov = sum(1 for r in results if r["has_temporal_state"]) / len(results) * 100

    print("\n" + "=" * 85)
    print("      PHASE CHRONOS: TEMPORAL COMPILER EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Questions Evaluated:      {len(results)}")
    print(f"Temporal State Coverage:  {sum(1 for r in results if r['has_temporal_state'])}/37 ({state_cov:.1f}%)")
    print(f"Accuracy:                 {acc:.1f}% ({num_correct}/37)")
    print(f"Oracle Recall:            {ora_pct:.1f}% ({num_ora}/37)")
    print(f"Mean Tokens / Question:   {mean_tok:.1f} tokens")
    print(f"Elapsed Time:             {elapsed:.1f}s")
    print("=" * 85)

    # Evolution Ladder
    print("\n" + "=" * 85)
    print("      AM APEX TEMPORAL EVOLUTION LADDER")
    print("=" * 85)
    print(f"1. Baseline P4 (No State):                  10.8% ( 4/37) | 412 tok")
    print(f"2. Phase CHRONOS (Current Run):             {acc:4.1f}% ({num_correct:2d}/37) | {mean_tok:3.0f} tok")
    print(f"   Delta vs Baseline:                      {acc - 10.8:+5.1f} pt ({num_correct - 4:+2d} Qs)")
    print(f"   Oracle Recall Ceiling:                   {ora_pct:4.1f}% ({num_ora:2d}/37)")
    print("=" * 85)

    out_file = RESULTS_DIR / "temporal_compiler_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(results),
                "accuracy": acc,
                "num_correct": num_correct,
                "oracle_recall": ora_pct,
                "num_oracle": num_ora,
                "mean_tokens": mean_tok,
                "state_coverage_pct": state_cov,
                "elapsed_s": elapsed,
                "baseline_accuracy": 10.8,
                "delta_vs_baseline": acc - 10.8,
                "details": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
