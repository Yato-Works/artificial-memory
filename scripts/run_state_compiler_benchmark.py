"""AM Apex Phase C5 & C6: Deterministic State Compiler Benchmark (32Q Multi-Hop).

Evaluates the 32 Multi-Hop questions (Category 1) of LoCoMo-10 Conv 0 with phi4-mini:3.8b
using the new Deterministic State Compiler (Phase C5 & C6).

Compares:
  1. Baseline P4 (No State):                  18.8% ( 6/32) | 361 tok
  2. Old StateSynthesizer (Retrieved State):  21.9% ( 7/32) | 363 tok
  3. New StateCompiler (Deterministic State):  ???% ( ?/32) | ??? tok
  4. Gold State (Upper Bound):                68.8% (22/32) | 370 tok

Saves complete telemetry to benchmark_results/protein/state_compiler_benchmark.json
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

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 85)
    print("   AM APEX PHASE C5 & C6: DETERMINISTIC STATE COMPILER BENCHMARK (32Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    # Compiler with New StateCompiler activated
    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )
    adapter.compiler = compiler

    results = []
    t0 = time.perf_counter()

    print("\n" + f"{'Q#':<4} | {'Status':<6} | {'Tokens':<8} | {'State Compiled':<14} | Question")
    print("-" * 85)

    for i, q in enumerate(multihop_questions):
        st_list = compiler.state_compiler.compile_states(q.question, ir_records)
        state_str = st_list[0].format_state() if st_list else "NONE"
        has_state = "YES" if st_list else "NO"

        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results.append({
            "question_id": q.question_id,
            "question": q.question,
            "ground_truth": q.ground_truth,
            "predicted_answer": res.predicted_answer,
            "is_correct": res.is_correct,
            "tokens_used": res.tokens_used,
            "compiled_state": state_str,
            "has_state": bool(st_list),
        })

        status = "PASS" if res.is_correct else "FAIL"
        print(f"[{i+1:02d}]  | {status:<6} | {res.tokens_used:3d} tok | {has_state:<14} | {q.question[:40]}")
        if res.is_correct and st_list:
            print(f"       >>> STATE: {state_str[:70]}...")
            print(f"       >>> PRED:  {res.predicted_answer}")

    elapsed = time.perf_counter() - t0
    num_correct = sum(1 for r in results if r["is_correct"])
    acc = num_correct / len(results) * 100
    mean_tok = sum(r["tokens_used"] for r in results) / len(results)
    state_cov = sum(1 for r in results if r["has_state"]) / len(results) * 100

    print("\n" + "=" * 85)
    print("      PHASE C5 & C6: STATE COMPILER EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Questions Evaluated:      {len(results)}")
    print(f"State Coverage:           {sum(1 for r in results if r['has_state'])}/32 ({state_cov:.1f}%)")
    print(f"Accuracy:                 {acc:.1f}% ({num_correct}/32)")
    print(f"Mean Tokens / Question:   {mean_tok:.1f} tokens")
    print(f"Elapsed Time:             {elapsed:.1f}s")
    print("=" * 85)

    # Comparative Ladder
    print("\n" + "=" * 85)
    print("      AM APEX MULTI-HOP EVOLUTION LADDER")
    print("=" * 85)
    print(f"1. Baseline P4 (No State):                  18.8% ( 6/32) | 361 tok")
    print(f"2. Old StateSynthesizer (Retrieved State):  21.9% ( 7/32) | 363 tok")
    print(f"3. New StateCompiler (Deterministic State):  {acc:4.1f}% ({num_correct:2d}/32) | {mean_tok:3.0f} tok")
    print(f"4. Gold State (Upper Bound):                68.8% (22/32) | 370 tok")
    print("=" * 85)

    out_file = RESULTS_DIR / "state_compiler_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(results),
                "accuracy": acc,
                "num_correct": num_correct,
                "mean_tokens": mean_tok,
                "state_coverage_pct": state_cov,
                "elapsed_s": elapsed,
                "ladder": {
                    "baseline_no_state": {"accuracy": 18.8, "correct": 6, "tokens": 361.2},
                    "old_retrieved_state": {"accuracy": 21.9, "correct": 7, "tokens": 362.4},
                    "new_state_compiler": {"accuracy": acc, "correct": num_correct, "tokens": mean_tok},
                    "gold_state_upper_bound": {"accuracy": 68.8, "correct": 22, "tokens": 369.8},
                },
                "details": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
