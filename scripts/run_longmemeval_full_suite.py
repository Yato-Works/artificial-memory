"""LongMemEval Full Benchmark Suite Runner (Phase 3).

Evaluates AM Apex Memory Runtime across all 500 questions of LongMemEval
with a STRICTLY FROZEN configuration (Phase X.4/X.5: No in-flight tuning).

Tests 6 core long-term memory capabilities:
1. Information extraction (single-session-user / single-session-assistant)
2. Multi-session reasoning
3. Temporal reasoning
4. Knowledge update
5. User preference
6. Abstention (safe refusal on unanswerable questions)

Saves incremental checkpoints and final report to benchmark_results/longmemeval/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from artificial_memory.research.benchmarks.external.longmemeval_adapter import (
    LongMemEvalAdapter,
    LongMemEvalResult,
)
from artificial_memory.research.benchmarks.failure_taxonomy_v2 import FailureClassifierV2
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("benchmark_results/longmemeval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval Full Suite (500 Questions)")
    parser.add_argument("--num-questions", type=int, default=0, help="Number of questions to evaluate (0 = all 500)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Reader model override (A/B only; frozen default is {FROZEN_MODEL})")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Ollama context window override. Large tags default to a 32K "
                             "KV cache and return HTTP 500 on an 8 GB GPU; pass 8192 for "
                             "7B-class readers.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for the report/checkpoint files. Use this for every "
                             "A/B arm so the frozen baseline report is never overwritten.")
    args = parser.parse_args()

    print("=" * 85)
    print("        AM APEX: LONGMEMEVAL FROZEN BENCHMARK SUITE (500 QUESTIONS)")
    print("=" * 85)
    print("Rule: STRICTLY FROZEN CONFIGURATION. Zero in-flight tuning.")
    print("=" * 85)

    adapter = LongMemEvalAdapter()
    t0_load = time.perf_counter()
    print("Loading LongMemEval dataset (277 MB)...")
    items = adapter.load_dataset()
    print(f"Loaded {len(items)} evaluation items in {time.perf_counter() - t0_load:.2f}s.\n")

    target_items = items[:args.num_questions] if args.num_questions > 0 else items
    print(f"Targeting {len(target_items)} questions.")

    answerer = OllamaAnswerer(model=args.model or FROZEN_MODEL, num_ctx=args.num_ctx)
    if args.model or args.num_ctx:
        print(f"READER: model={answerer.model} num_ctx={answerer.num_ctx} (A/B override)")

    results: list[LongMemEvalResult] = []
    checkpoint_file = RESULTS_DIR / (f"checkpoint_{args.tag}.json" if args.tag else "checkpoint.json")
    start_idx = 0

    if args.resume and checkpoint_file.exists():
        print(f"Resuming from checkpoint {checkpoint_file}...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
            for r in ckpt_data.get("results", []):
                results.append(
                    LongMemEvalResult(
                        question_id=r["question_id"],
                        question_type=r["question_type"],
                        oracle_recall=r["oracle_recall"],
                        predicted_answer=r["predicted_answer"],
                        ground_truth=r["ground_truth"],
                        tokens_used=r["tokens_used"],
                        latency_ms=r["latency_ms"],
                        is_correct=r["is_correct"],
                    )
                )
            start_idx = len(results)
            print(f"Resumed {start_idx} completed questions. Current Acc: {sum(1 for r in results if r.is_correct)/len(results)*100:.1f}%\n")

    t_eval_start = time.perf_counter()

    for i in range(start_idx, len(target_items)):
        item = target_items[i]
        res = adapter.evaluate_item(item, answerer)
        results.append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora_status = "O" if res.oracle_recall else "X"

        if (i + 1) % 5 == 0 or (i + 1) == len(target_items):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"  [{i+1:03d}/{len(target_items):03d}] [{status}] Acc: {curr_acc:5.1f}% | Ora: {curr_ora:5.1f}% | Tok: {curr_tok:4.0f} | Type: {item.question_type[:16]:<16} | Q: {item.question[:28]}")

        # Save checkpoint every 25 questions
        if (i + 1) % 25 == 0 or (i + 1) == len(target_items):
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed": len(results),
                        "total": len(target_items),
                        "results": [
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
                        ],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    total_time = time.perf_counter() - t_eval_start

    # Compute Final Report Metrics
    total_qs = len(results)
    total_correct = sum(1 for r in results if r.is_correct)
    total_oracle = sum(1 for r in results if r.oracle_recall)

    overall_acc = total_correct / total_qs if total_qs else 0.0
    overall_ora = total_oracle / total_qs if total_qs else 0.0
    mean_tokens = sum(r.tokens_used for r in results) / total_qs if total_qs else 0.0
    mean_lat = sum(r.latency_ms for r in results) / total_qs if total_qs else 0.0

    # Question Type Breakdown
    type_totals = defaultdict(int)
    type_corrects = defaultdict(int)
    type_oracles = defaultdict(int)
    for r in results:
        type_totals[r.question_type] += 1
        if r.is_correct:
            type_corrects[r.question_type] += 1
        if r.oracle_recall:
            type_oracles[r.question_type] += 1

    print("\n" + "=" * 85)
    print("        AM APEX: LONGMEMEVAL FULL BENCHMARK FINAL REPORT")
    print("=" * 85)
    print(f"Total Questions Evaluated:      {total_qs}")
    print(f"Overall Accuracy:               {overall_acc * 100:.1f}% ({total_correct}/{total_qs})")
    print(f"Memory Oracle Recall:           {overall_ora * 100:.1f}% ({total_oracle}/{total_qs})")
    print(f"Mean Context Tokens/Q:          {mean_tokens:.1f} tokens/Q")
    print(f"Mean Latency:                   {mean_lat:.1f} ms")
    print(f"Write LLM Calls:                0 calls (100% Free Deterministic Ingestion)")
    print(f"Total Benchmark Suite Time:     {total_time:.1f} s ({total_time / 60:.1f} min)")
    print("-" * 85)
    print("QUESTION TYPE BREAKDOWN:")
    for t_name in sorted(type_totals.keys()):
        cnt = type_totals[t_name]
        c_acc = (type_corrects[t_name] / cnt) * 100 if cnt else 0.0
        c_ora = (type_oracles[t_name] / cnt) * 100 if cnt else 0.0
        conv_rate = (type_corrects[t_name] / type_oracles[t_name]) * 100 if type_oracles[t_name] else 0.0
        print(f"  * {t_name:<26}: Acc {c_acc:5.1f}% | Oracle {c_ora:5.1f}% | ConvRate {conv_rate:5.1f}% ({type_corrects[t_name]}/{cnt})")
    print("=" * 85)

    # Failure Taxonomy v2 Classification
    classifier = FailureClassifierV2()
    failure_counts = defaultdict(int)
    failure_diagnoses = []
    for r, item in zip(results, target_items):
        if not r.is_correct:
            diag = classifier.classify(
                question=item.question,
                ground_truth=item.answer,
                predicted_answer=r.predicted_answer,
                context="",
                oracle_recall=r.oracle_recall,
                is_correct=r.is_correct,
                question_type=item.question_type,
            )
            if diag:
                failure_counts[diag.category.value] += 1
                failure_diagnoses.append({
                    "question_id": r.question_id,
                    "category": diag.category.value,
                    "explanation": diag.explanation,
                })

    print("\n" + "=" * 85)
    print("        FAILURE TAXONOMY v2 BREAKDOWN (8 Causal Categories)")
    print("=" * 85)
    total_failures = sum(failure_counts.values())
    for cat, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = count / total_failures * 100 if total_failures else 0.0
        print(f"  * {cat:<24}: {count:3d} ({pct:5.1f}%)")
    print("=" * 85)

    # Save Grand Final Report JSON
    final_report_file = RESULTS_DIR / (
        f"grand_longmemeval_report_{args.tag}.json" if args.tag
        else "grand_longmemeval_report.json"
    )
    with open(final_report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "LongMemEval (500 Questions)",
                "reader_model": answerer.model,
                "reader_num_ctx": answerer.num_ctx,
                "total_questions": total_qs,
                "overall_accuracy": overall_acc,
                "oracle_recall": overall_ora,
                "mean_tokens_per_q": mean_tokens,
                "mean_latency_ms": mean_lat,
                "total_elapsed_seconds": total_time,
                "write_llm_calls": 0,
                "type_breakdown": {
                    t: {
                        "total": type_totals[t],
                        "correct": type_corrects[t],
                        "oracle": type_oracles[t],
                        "conversion_rate": (type_corrects[t] / type_oracles[t] * 100) if type_oracles[t] else 0.0,
                    }
                    for t in type_totals
                },
                "failure_taxonomy_v2": dict(failure_counts),
                "failure_diagnoses": failure_diagnoses,
                "results": [
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
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved full comprehensive report to {final_report_file}]")


if __name__ == "__main__":
    main()
