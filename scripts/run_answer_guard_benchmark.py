"""AM Apex Phase IMMUNE: Answer Guard Benchmark (32Q Multi-Hop on phi4-mini:3.8b).

Evaluates the 32 Multi-Hop questions (Category 1) of LoCoMo-10 Conv 0 with phi4-mini:3.8b
using the full Phase IMMUNE suite:
  - StateRefusalGuard: Recovers answers from directly supported [STATE] upon false abstention / contradiction
  - NumericNormalizer: Normalizes surface word forms ("twice" -> "2")
  - Full StateCompiler & VITAMIN suite

Target Gates:
  - Gate 1: 22/32 (68.8% — Gold State Upper Bound Level)
  - Gate 2: 23/32 (71.9% — Surpassing Gold State Reference)
  - Gate 3: 24/32 (75.0% — Beyond-Gold Summit)

Saves complete results to benchmark_results/protein/answer_guard_benchmark.json
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
    print("        AM APEX PHASE IMMUNE: ANSWER GUARD SUMMIT BENCHMARK (32Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    compiler = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )
    adapter.compiler = compiler

    results = []
    t0 = time.perf_counter()

    print("\n" + f"{'Q#':<4} | {'Status':<6} | {'Tokens':<8} | {'State':<5} | Question")
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
        print(f"[{i+1:02d}]  | {status:<6} | {res.tokens_used:3d} tok | {has_state:<5} | {q.question[:40]}")
        if res.is_correct:
            print(f"       >>> PRED: {res.predicted_answer}")
            if st_list:
                print(f"       >>> STATE: {state_str[:70]}...")
        else:
            print(f"       --- FAIL | GT: {q.ground_truth} | PRED: {res.predicted_answer}")

    elapsed = time.perf_counter() - t0
    num_correct = sum(1 for r in results if r["is_correct"])
    acc = num_correct / len(results) * 100
    mean_tok = sum(r["tokens_used"] for r in results) / len(results)
    state_cov = sum(1 for r in results if r["has_state"]) / len(results) * 100

    print("\n" + "=" * 85)
    print("      PHASE IMMUNE: ANSWER GUARD EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Questions Evaluated:      {len(results)}")
    print(f"State Coverage:           {sum(1 for r in results if r['has_state'])}/32 ({state_cov:.1f}%)")
    print(f"Accuracy:                 {acc:.1f}% ({num_correct}/32)")
    print(f"Mean Tokens / Question:   {mean_tok:.1f} tokens")
    print(f"Elapsed Time:             {elapsed:.1f}s")
    print("=" * 85)

    # Evolution Ladder
    print("\n" + "=" * 85)
    print("      AM APEX MULTI-HOP GRAND EVOLUTION LADDER")
    print("=" * 85)
    print(f"1. Baseline P4 (No State):                  18.8% ( 6/32) | 361 tok")
    print(f"2. Old StateSynthesizer (Retrieved State):  21.9% ( 7/32) | 362 tok")
    print(f"3. StateCompiler C5/C6:                     43.8% (14/32) | 376 tok")
    print(f"4. Phase VITAMIN:                           59.4% (19/32) | 375 tok")
    print(f"5. Gold State (Upper Bound Reference):      68.8% (22/32) | 368 tok")
    print(f"6. Phase IMMUNE (Current Run):              {acc:4.1f}% ({num_correct:2d}/32) | {mean_tok:3.0f} tok")
    print("=" * 85)

    out_file = RESULTS_DIR / "answer_guard_benchmark.json"
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
                    "state_compiler_c5_c6": {"accuracy": 43.8, "correct": 14, "tokens": 375.7},
                    "phase_vitamin": {"accuracy": 59.4, "correct": 19, "tokens": 375.4},
                    "gold_state_upper_bound": {"accuracy": 68.8, "correct": 22, "tokens": 368.3},
                    "phase_immune": {"accuracy": acc, "correct": num_correct, "tokens": mean_tok},
                },
                "details": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
