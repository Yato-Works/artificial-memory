"""Steroid Dual Benchmark Runner (Phase S5).

Evaluates the AM Apex Steroid Engine:
- Wide Slicing (6-channel candidate union)
- Adaptive Graph Expansion (1-3 hops based on completeness)
- Fused with Overdrive Core (TemporalResolver, PersonaStore, IntegrityGate, AnswerVerifier)

Evaluates on:
1. LoCoMo-10 (Conv 0-9)
2. LongMemEval (500 Questions)

Saves results to benchmark_results/steroid/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.steroid.steroid_compiler import SteroidContextCompiler

RESULTS_DIR = Path("benchmark_results/steroid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_steroid_locomo(adapter: LoCoMoAdapter, answerer: OllamaAnswerer, conv_idx: int = 0) -> dict:
    print("\n" + "=" * 80)
    print(f"--- STEROID EVALUATION: LOCOMO CONVERSATION {conv_idx} ---")
    print("=" * 80)

    turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
    print(f"Loaded {len(questions)} questions for Conv {conv_idx}.")

    results = []
    t0 = time.perf_counter()

    for i, q in enumerate(questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results.append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"  [{i+1:03d}/{len(questions):03d}] Acc: {curr_acc:5.1f}% | Ora: {curr_ora:5.1f}% | Tok/Q: {curr_tok:5.1f} | Q: {q.question[:32]}")

    elapsed = time.perf_counter() - t0
    acc = sum(1 for r in results if r.is_correct) / len(results) * 100
    ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
    mean_tok = sum(r.tokens_used for r in results) / len(results)

    # Multi-hop specific
    mh_qs = [r for r in results if r.category == 1]
    mh_acc = sum(1 for r in mh_qs if r.is_correct) / len(mh_qs) * 100 if mh_qs else 0.0

    print("\n" + "=" * 80)
    print(f"LOCOMO CONV {conv_idx} STEROID SUMMARY:")
    print(f"  * Overall Accuracy:     {acc:5.1f}% ({sum(1 for r in results if r.is_correct)}/{len(results)})")
    print(f"  * Oracle Recall:        {ora:5.1f}% ({sum(1 for r in results if r.oracle_recall)}/{len(results)})")
    print(f"  * Multi-Hop Accuracy:   {mh_acc:5.1f}% ({sum(1 for r in mh_qs if r.is_correct)}/{len(mh_qs)})")
    print(f"  * Mean Tokens/Q:        {mean_tok:5.1f} tok")
    print(f"  * Elapsed Time:         {elapsed:.1f} s")
    print("=" * 80)

    out_file = RESULTS_DIR / f"locomo_conv_{conv_idx}_steroid.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "conv_idx": conv_idx,
                "overall_accuracy": acc,
                "oracle_recall": ora,
                "multi_hop_accuracy": mh_acc,
                "mean_tokens": mean_tok,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )
    return {"acc": acc, "ora": ora, "mh_acc": mh_acc, "tok": mean_tok}


def run_steroid_longmemeval(adapter: LongMemEvalAdapter, answerer: OllamaAnswerer, num_items: int = 50) -> dict:
    print("\n" + "=" * 80)
    print(f"--- STEROID EVALUATION: LONGMEMEVAL ({num_items} QUESTIONS) ---")
    print("=" * 80)

    items = adapter.load_dataset()
    target_items = items[:num_items] if num_items > 0 else items

    results = []
    t0 = time.perf_counter()

    for i, it in enumerate(target_items):
        res = adapter.evaluate_item(it, answerer)
        results.append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"

        if (i + 1) % 10 == 0 or (i + 1) == len(target_items):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"  [{i+1:03d}/{len(target_items):03d}] Acc: {curr_acc:5.1f}% | Ora: {curr_ora:5.1f}% | Tok/Q: {curr_tok:5.1f} | Type: {it.question_type[:16]:<16} | Q: {it.question[:28]}")

    elapsed = time.perf_counter() - t0
    acc = sum(1 for r in results if r.is_correct) / len(results) * 100
    ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
    mean_tok = sum(r.tokens_used for r in results) / len(results)

    print("\n" + "=" * 80)
    print(f"LONGMEMEVAL STEROID SUMMARY ({len(target_items)} Questions):")
    print(f"  * Overall Accuracy:     {acc:5.1f}% ({sum(1 for r in results if r.is_correct)}/{len(results)})")
    print(f"  * Oracle Recall:        {ora:5.1f}% ({sum(1 for r in results if r.oracle_recall)}/{len(results)})")
    print(f"  * Mean Tokens/Q:        {mean_tok:5.1f} tok")
    print(f"  * Elapsed Time:         {elapsed:.1f} s")
    print("=" * 80)

    out_file = RESULTS_DIR / f"longmemeval_steroid_{len(target_items)}q.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(target_items),
                "overall_accuracy": acc,
                "oracle_recall": ora,
                "mean_tokens": mean_tok,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )
    return {"acc": acc, "ora": ora, "tok": mean_tok}


def main():
    parser = argparse.ArgumentParser(description="Run Steroid Dual Benchmark")
    parser.add_argument("--mode", choices=["locomo", "longmemeval", "both"], default="both")
    parser.add_argument("--conv-idx", type=int, default=0)
    parser.add_argument("--longmem-num", type=int, default=50)
    args = parser.parse_args()

    print("=" * 80)
    print("        AM APEX STEROID PHASE — PHASE S5: DUAL BENCHMARK RUN")
    print("=" * 80)

    answerer = OllamaAnswerer()
    compiler = SteroidContextCompiler()

    locomo_adapter = LoCoMoAdapter()
    locomo_adapter.compiler = compiler

    longmem_adapter = LongMemEvalAdapter()
    longmem_adapter.compiler = compiler

    if args.mode in ["locomo", "both"]:
        run_steroid_locomo(locomo_adapter, answerer, conv_idx=args.conv_idx)

    if args.mode in ["longmemeval", "both"]:
        run_steroid_longmemeval(longmem_adapter, answerer, num_items=args.longmem_num)


if __name__ == "__main__":
    main()
