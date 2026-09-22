"""State Synthesis & Chain Retention 4-Condition Factorial Ablation (Phase C2).

Evaluates the 32 Multi-Hop questions (Category 1) of LoCoMo-10 Conv 0 under 4 conditions:
  Condition A: Chain ❌ | STATE ❌ (Current Frozen P4 Baseline)
  Condition B: Chain ✅ | STATE ❌ (Chain Retention alone)
  Condition C: Chain ❌ | STATE ✅ (State Synthesis alone on Top-K)
  Condition D: Chain ✅ | STATE ✅ (Full State-Augmented Chain)

Computes:
- Multi-Hop Accuracy (%)
- Oracle Recall (%)
- Mean Tokens/Q
- Attribution Funnel:
    Retrieval -> Chain Completion -> State Synthesis -> Final Answer Correct
Saves full telemetry to benchmark_results/protein/state_synthesis_ablation_study.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_condition(
    name: str,
    compiler: ProteinContextCompiler,
    adapter: LoCoMoAdapter,
    questions: list,
    turns: list,
    ir_records: list,
    answerer: OllamaAnswerer,
) -> dict:
    print(f"\n{'=' * 85}")
    print(f"  >>> RUNNING {name}")
    print(f"{'=' * 85}")

    adapter.compiler = compiler
    results = []
    t0 = time.perf_counter()

    for i, q in enumerate(questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"
        print(f"  [{name[:12]:<12} {i+1:02d}/32] {status} (Ora:{ora}) | {res.tokens_used:3d} tok | Q: {q.question[:40]}")

    elapsed = time.perf_counter() - t0
    acc = sum(1 for r in results if r.is_correct) / len(results) * 100
    ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
    tok = sum(r.tokens_used for r in results) / len(results)

    return {
        "name": name,
        "accuracy": acc,
        "oracle_recall": ora,
        "mean_tokens": tok,
        "elapsed_s": elapsed,
        "results": results,
    }


def main():
    print("=" * 85)
    print("      AM APEX PHASE C2: STATE SYNTHESIS 4-CONDITION ABLATION (32Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    # 1. Condition A: Chain ❌, STATE ❌ (Current P4 Baseline)
    comp_a = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=False,
    )

    # 2. Condition B: Chain ✅, STATE ❌ (Chain Retention alone)
    comp_b = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=True,
        max_chain_reserve=3,
        enable_state_synthesis=False,
    )

    # 3. Condition C: Chain ❌, STATE ✅ (State Synthesis alone on Top-K)
    comp_c = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )

    # 4. Condition D: Chain ✅, STATE ✅ (Full State-Augmented Chain)
    comp_d = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=True,
        max_chain_reserve=3,
        enable_state_synthesis=True,
    )

    res_a = run_condition("Condition A (Chain ❌, STATE ❌)", comp_a, adapter, multihop_questions, turns, ir_records, answerer)
    res_b = run_condition("Condition B (Chain ✅, STATE ❌)", comp_b, adapter, multihop_questions, turns, ir_records, answerer)
    res_c = run_condition("Condition C (Chain ❌, STATE ✅)", comp_c, adapter, multihop_questions, turns, ir_records, answerer)
    res_d = run_condition("Condition D (Chain ✅, STATE ✅)", comp_d, adapter, multihop_questions, turns, ir_records, answerer)

    # Summary Table
    print("\n" + "=" * 85)
    print("      PHASE C2: 4-CONDITION FACTORIAL ABLATION SUMMARY")
    print("=" * 85)
    print(f"{'Condition':<30} | {'Chain':<6} | {'STATE':<6} | {'Accuracy':<14} | {'Oracle':<14} | {'Tokens/Q':<10}")
    print("-" * 85)
    print(f"{'Condition A (Baseline)':<30} | {'OFF':<6} | {'OFF':<6} | {res_a['accuracy']:5.1f}% ({sum(1 for r in res_a['results'] if r.is_correct):02d}/32) | {res_a['oracle_recall']:5.1f}% ({sum(1 for r in res_a['results'] if r.oracle_recall):02d}/32) | {res_a['mean_tokens']:5.1f} tok")
    print(f"{'Condition B (Chain Alone)':<30} | {'ON':<6} | {'OFF':<6} | {res_b['accuracy']:5.1f}% ({sum(1 for r in res_b['results'] if r.is_correct):02d}/32) | {res_b['oracle_recall']:5.1f}% ({sum(1 for r in res_b['results'] if r.oracle_recall):02d}/32) | {res_b['mean_tokens']:5.1f} tok")
    print(f"{'Condition C (State Alone)':<30} | {'OFF':<6} | {'ON':<6} | {res_c['accuracy']:5.1f}% ({sum(1 for r in res_c['results'] if r.is_correct):02d}/32) | {res_c['oracle_recall']:5.1f}% ({sum(1 for r in res_c['results'] if r.oracle_recall):02d}/32) | {res_c['mean_tokens']:5.1f} tok")
    print(f"{'Condition D (Chain + State)':<30} | {'ON':<6} | {'ON':<6} | {res_d['accuracy']:5.1f}% ({sum(1 for r in res_d['results'] if r.is_correct):02d}/32) | {res_d['oracle_recall']:5.1f}% ({sum(1 for r in res_d['results'] if r.oracle_recall):02d}/32) | {res_d['mean_tokens']:5.1f} tok")
    print("=" * 85)

    # Detailed Funnel & Transition Breakdown
    print("\n--- ATTRIBUTION & FUNNEL ANALYSIS (CONDITION D vs A) ---")
    question_details = []
    for i, q in enumerate(multihop_questions):
        ra = res_a["results"][i]
        rb = res_b["results"][i]
        rc = res_c["results"][i]
        rd = res_d["results"][i]

        state_generated = bool(comp_d.state_synthesizer.synthesize(q.question, ir_records))

        status_str = f"A:{'P' if ra.is_correct else 'F'} | B:{'P' if rb.is_correct else 'F'} | C:{'P' if rc.is_correct else 'F'} | D:{'P' if rd.is_correct else 'F'}"
        print(f"[{status_str}] State:{'Y' if state_generated else 'N'} | Q: {q.question[:45]}")
        if not ra.is_correct and rd.is_correct:
            print(f"    >>> SOLVED BY STATE-AUGMENTED CHAIN! GT: {q.ground_truth} | Pred: {rd.predicted_answer}")

        question_details.append({
            "question_id": q.question_id,
            "question": q.question,
            "ground_truth": q.ground_truth,
            "cond_a": {"correct": ra.is_correct, "oracle": ra.oracle_recall, "pred": ra.predicted_answer, "tokens": ra.tokens_used},
            "cond_b": {"correct": rb.is_correct, "oracle": rb.oracle_recall, "pred": rb.predicted_answer, "tokens": rb.tokens_used},
            "cond_c": {"correct": rc.is_correct, "oracle": rc.oracle_recall, "pred": rc.predicted_answer, "tokens": rc.tokens_used},
            "cond_d": {"correct": rd.is_correct, "oracle": rd.oracle_recall, "pred": rd.predicted_answer, "tokens": rd.tokens_used},
        })

    out_file = RESULTS_DIR / "state_synthesis_ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(multihop_questions),
                "conditions": {
                    "A": {"accuracy": res_a["accuracy"], "oracle": res_a["oracle_recall"], "tokens": res_a["mean_tokens"], "elapsed": res_a["elapsed_s"]},
                    "B": {"accuracy": res_b["accuracy"], "oracle": res_b["oracle_recall"], "tokens": res_b["mean_tokens"], "elapsed": res_b["elapsed_s"]},
                    "C": {"accuracy": res_c["accuracy"], "oracle": res_c["oracle_recall"], "tokens": res_c["mean_tokens"], "elapsed": res_c["elapsed_s"]},
                    "D": {"accuracy": res_d["accuracy"], "oracle": res_d["oracle_recall"], "tokens": res_d["mean_tokens"], "elapsed": res_d["elapsed_s"]},
                },
                "questions": question_details,
            },
            f,
            indent=2,
        )
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
