"""Chain Retention Study on LoCoMo-10 Multi-Hop Questions (Phase C1).

Compares:
- Baseline P4 (Top-K = 10, Chain Retention OFF)
- Chain-Aware P4 (Top-K = 10 + Chain Retention ON, Reserve = 3)

Evaluates on the 32 Multi-Hop questions (Category 1) of LoCoMo-10 Conversation 0.
Saves detailed results to benchmark_results/protein/chain_retention_multihop_study.json
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
    print("      AM APEX PHASE C1: GRAPH CHAIN RETENTION STUDY (LOCOMO MULTI-HOP 32Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Loaded {len(all_questions)} total questions from Conv 0.")
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions for ablation.")
    print("-" * 85)

    # Compiler A: Baseline P4 (Chain Retention OFF)
    compiler_baseline = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
    )

    # Compiler B: Chain-Aware P4 (Chain Retention ON, Reserve = 3)
    compiler_chain = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=True,
        max_chain_reserve=3,
    )

    results_baseline = []
    results_chain = []
    comparison_table = []

    print(f"\n>>> Running Condition A: Baseline P4 (K=10, Chain OFF)...")
    t0_base = time.perf_counter()
    adapter.compiler = compiler_baseline
    for i, q in enumerate(multihop_questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results_baseline.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"
        print(f"  [Baseline {i+1:02d}/32] {status} (Ora:{ora}) | {res.tokens_used:3d} tok | Q: {q.question[:40]}")
    elapsed_base = time.perf_counter() - t0_base

    print(f"\n>>> Running Condition B: Chain-Aware P4 (K=10 + Chain Reserve 3)...")
    t0_chain = time.perf_counter()
    adapter.compiler = compiler_chain
    for i, q in enumerate(multihop_questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results_chain.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"
        print(f"  [Chain-Aware {i+1:02d}/32] {status} (Ora:{ora}) | {res.tokens_used:3d} tok | Q: {q.question[:40]}")
    elapsed_chain = time.perf_counter() - t0_chain

    # Analysis & Comparison
    acc_base = sum(1 for r in results_baseline if r.is_correct) / len(results_baseline) * 100
    ora_base = sum(1 for r in results_baseline if r.oracle_recall) / len(results_baseline) * 100
    tok_base = sum(r.tokens_used for r in results_baseline) / len(results_baseline)

    acc_chain = sum(1 for r in results_chain if r.is_correct) / len(results_chain) * 100
    ora_chain = sum(1 for r in results_chain if r.oracle_recall) / len(results_chain) * 100
    tok_chain = sum(r.tokens_used for r in results_chain) / len(results_chain)

    print("\n" + "=" * 85)
    print("      MULTI-HOP ABLATION STUDY RESULTS (32 QUESTIONS)")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Baseline (Chain OFF)':<22} | {'Chain-Aware (Reserve=3)':<24} | {'Delta':<10}")
    print("-" * 85)
    print(f"{'Multi-Hop Accuracy':<25} | {acc_base:5.1f}% ({sum(1 for r in results_baseline if r.is_correct):02d}/32)            | {acc_chain:5.1f}% ({sum(1 for r in results_chain if r.is_correct):02d}/32)             | {acc_chain - acc_base:+5.1f} pt")
    print(f"{'Oracle Recall':<25} | {ora_base:5.1f}% ({sum(1 for r in results_baseline if r.oracle_recall):02d}/32)            | {ora_chain:5.1f}% ({sum(1 for r in results_chain if r.oracle_recall):02d}/32)             | {ora_chain - ora_base:+5.1f} pt")
    print(f"{'Mean Tokens/Q':<25} | {tok_base:5.1f} tok                  | {tok_chain:5.1f} tok                   | {tok_chain - tok_base:+5.1f} tok")
    print(f"{'Elapsed Time':<25} | {elapsed_base:5.1f} s                    | {elapsed_chain:5.1f} s                     | {elapsed_chain - elapsed_base:+5.1f} s")
    print("=" * 85)

    # Detailed Question-by-Question Deltas
    print("\n--- DETAILED QUESTION-BY-QUESTION TRANSITIONS ---")
    transitions = []
    for q, rb, rc in zip(multihop_questions, results_baseline, results_chain):
        trans = ""
        if not rb.is_correct and rc.is_correct:
            trans = "IMPROVED (FAIL -> PASS) *"
        elif rb.is_correct and not rc.is_correct:
            trans = "REGRESSED (PASS -> FAIL) !"
        elif rb.is_correct and rc.is_correct:
            trans = "SUSTAINED (PASS -> PASS)"
        else:
            trans = "UNRESOLVED (FAIL -> FAIL)"

        print(f"[{trans:<25}] Q: {q.question[:45]}")
        if "IMPROVED" in trans:
            print(f"    GT:   {q.ground_truth}")
            print(f"    Base: {rb.predicted_answer}")
            print(f"    Chn:  {rc.predicted_answer}")

        transitions.append(
            {
                "question_id": q.question_id,
                "question": q.question,
                "ground_truth": q.ground_truth,
                "transition": trans,
                "baseline_correct": rb.is_correct,
                "chain_correct": rc.is_correct,
                "baseline_oracle": rb.oracle_recall,
                "chain_oracle": rc.oracle_recall,
                "baseline_tokens": rb.tokens_used,
                "chain_tokens": rc.tokens_used,
                "baseline_pred": rb.predicted_answer,
                "chain_pred": rc.predicted_answer,
            }
        )

    out_file = RESULTS_DIR / "chain_retention_multihop_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_questions": len(multihop_questions),
                "baseline": {
                    "accuracy": acc_base,
                    "oracle_recall": ora_base,
                    "mean_tokens": tok_base,
                    "elapsed_s": elapsed_base,
                },
                "chain_aware": {
                    "accuracy": acc_chain,
                    "oracle_recall": ora_chain,
                    "mean_tokens": tok_chain,
                    "elapsed_s": elapsed_chain,
                },
                "delta": {
                    "accuracy": acc_chain - acc_base,
                    "oracle_recall": ora_chain - ora_base,
                    "tokens": tok_chain - tok_base,
                },
                "transitions": transitions,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
