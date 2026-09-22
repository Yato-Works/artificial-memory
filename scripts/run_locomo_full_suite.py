"""LoCoMo-10 Full Benchmark Suite Runner (Phase 1: Frozen Generalization Test).

Evaluates AM Apex Memory Runtime across all 10 conversations of LoCoMo-10 (1,986 questions)
with a STRICTLY FROZEN configuration (Phase X.4/X.5: No in-flight tuning).

Measures:
1. Micro & Macro Answer Accuracy
2. Micro & Macro Memory Oracle Recall
3. Category-by-category breakdown (Temporal, Multi-Hop, Single-Hop, Open-Domain, Adversarial)
4. Resource Pareto Frontier (Tokens/Q, Latency, Write LLM Calls: 0)

Saves each conversation's results to benchmark_results/locomo10/conv_{idx}_results.json
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

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import (
    LoCoMoAdapter,
    LoCoMoEvalResult,
)
from artificial_memory.research.benchmarks.failure_taxonomy_v2 import FailureClassifierV2
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("benchmark_results/locomo10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_conversation(
    adapter: LoCoMoAdapter,
    answerer: OllamaAnswerer,
    weights: EvidenceScoreWeights,
    conv_idx: int,
) -> dict:
    """Run evaluation for a single conversation and save JSON checkpoint."""
    print("\n" + "=" * 75)
    print(f"--- STARTING CONVERSATION {conv_idx} (Frozen Config) ---")
    print("=" * 75)

    t0_load = time.perf_counter()
    turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
    load_time = time.perf_counter() - t0_load

    sample_id = questions[0].conv_id if questions else f"conv-{conv_idx}"
    speakers = sorted(list({(t.speaker or "").strip() for t in turns if t.speaker}))

    print(f"Sample ID: {sample_id} | Speakers: {speakers}")
    print(f"Ingested {len(turns)} turns into {len(ir_records)} StructuredIR records in {load_time:.2f}s (Write LLM: 0).")
    print(f"Evaluating {len(questions)} questions...")

    results: list[LoCoMoEvalResult] = []
    t0_eval = time.perf_counter()

    for i, q in enumerate(questions):
        res = adapter.evaluate_question(
            q,
            turns,
            ir_records,
            answerer,
            weights=weights,
        )
        results.append(res)

        status = "PASS" if res.is_correct else "FAIL"
        ora_status = "O" if res.oracle_recall else "X"
        cat_name = adapter.CATEGORY_NAMES.get(res.category, str(res.category))[:8]
        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            curr_tok = sum(r.tokens_used for r in results) / len(results)
            print(f"  [{i+1:03d}/{len(questions):03d}] Acc: {curr_acc:5.1f}% | Ora: {curr_ora:5.1f}% | Tok/Q: {curr_tok:5.1f} | Q: {q.question[:32]}")

    eval_time = time.perf_counter() - t0_eval

    tot_acc = sum(1 for r in results if r.is_correct) / len(results) if results else 0.0
    tot_ora = sum(1 for r in results if r.oracle_recall) / len(results) if results else 0.0
    mean_tok = sum(r.tokens_used for r in results) / len(results) if results else 0.0
    mean_lat = sum(r.latency_ms for r in results) / len(results) if results else 0.0

    # Category breakdown
    cat_breakdown = {}
    for r in results:
        c_name = adapter.CATEGORY_NAMES.get(r.category, f"cat-{r.category}")
        if c_name not in cat_breakdown:
            cat_breakdown[c_name] = {"total": 0, "correct": 0, "oracle": 0}
        cat_breakdown[c_name]["total"] += 1
        if r.is_correct:
            cat_breakdown[c_name]["correct"] += 1
        if r.oracle_recall:
            cat_breakdown[c_name]["oracle"] += 1

    # Factual (excluding adversarial)
    factual = [r for r in results if r.category != 5]
    factual_acc = sum(1 for r in factual if r.is_correct) / len(factual) if factual else 0.0
    factual_ora = sum(1 for r in factual if r.oracle_recall) / len(factual) if factual else 0.0

    # Diagnose failures with Failure Taxonomy v2
    classifier = FailureClassifierV2()
    failure_counts = defaultdict(int)
    diagnoses = []
    for q, r in zip(questions, results):
        if not r.is_correct:
            diag = classifier.classify(
                question=q.question,
                ground_truth=q.ground_truth,
                predicted_answer=r.predicted_answer,
                context="",
                oracle_recall=r.oracle_recall,
                is_correct=r.is_correct,
                question_type=adapter.CATEGORY_NAMES.get(q.category, "general"),
            )
            if diag:
                failure_counts[diag.category.value] += 1
                diagnoses.append({
                    "question_id": q.question_id,
                    "category": diag.category.value,
                    "explanation": diag.explanation,
                })

    summary = {
        "conv_idx": conv_idx,
        "sample_id": sample_id,
        "speakers": speakers,
        "total_questions": len(questions),
        "total_turns": len(turns),
        "total_ir_records": len(ir_records),
        "overall_accuracy": tot_acc,
        "overall_oracle_recall": tot_ora,
        "factual_accuracy": factual_acc,
        "factual_oracle_recall": factual_ora,
        "mean_tokens": mean_tok,
        "mean_latency_ms": mean_lat,
        "elapsed_seconds": eval_time,
        "categories": cat_breakdown,
        "failure_taxonomy_v2": dict(failure_counts),
        "diagnoses": diagnoses,
    }

    # Save checkpoint
    out_file = RESULTS_DIR / f"conv_{conv_idx}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "results": [
                    {
                        "question_id": r.question_id,
                        "category": r.category,
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
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n>> CONV {conv_idx} COMPLETED in {eval_time:.1f}s:")
    print(f"   * Accuracy:       {tot_acc * 100:.1f}% ({sum(1 for r in results if r.is_correct)}/{len(results)})")
    print(f"   * Oracle Recall:  {tot_ora * 100:.1f}% ({sum(1 for r in results if r.oracle_recall)}/{len(results)})")
    print(f"   * Factual Acc:    {factual_acc * 100:.1f}% ({sum(1 for r in factual if r.is_correct)}/{len(factual)})")
    print(f"   * Factual Recall: {factual_ora * 100:.1f}% ({sum(1 for r in factual if r.oracle_recall)}/{len(factual)})")
    print(f"   * Mean Tokens/Q:  {mean_tok:.1f} tok")
    print(f"   * Saved to:       {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo-10 Full Suite (Conv 0-9)")
    parser.add_argument("--start-conv", type=int, default=0, help="Start conversation index (default: 0)")
    parser.add_argument("--end-conv", type=int, default=9, help="End conversation index (default: 9)")
    parser.add_argument("--resume", action="store_true", help="Skip conversations with existing checkpoints")
    args = parser.parse_args()

    print("=" * 80)
    print("      AM APEX: LOCOMO-10 FROZEN GENERALIZATION BENCHMARK (CONV 0 - 9)")
    print("=" * 80)
    print(f"Target Range: Conversation {args.start_conv} to {args.end_conv}")
    print("Rule: STRICTLY FROZEN CONFIGURATION. Zero in-flight tuning.")
    print("=" * 80)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()
    weights = EvidenceScoreWeights()

    all_summaries = []
    t_global_start = time.perf_counter()

    for c_idx in range(args.start_conv, args.end_conv + 1):
        out_file = RESULTS_DIR / f"conv_{c_idx}_results.json"
        if args.resume and out_file.exists():
            print(f"\n[Resuming: Found existing checkpoint for Conv {c_idx}, loading from {out_file}...]")
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data["summary"]
                all_summaries.append(summary)
                print(f"   * Conv {c_idx} Acc: {summary['overall_accuracy']*100:.1f}% | Ora: {summary['overall_oracle_recall']*100:.1f}%")
                continue

        summary = run_conversation(adapter, answerer, weights, c_idx)
        all_summaries.append(summary)

    total_global_time = time.perf_counter() - t_global_start

    # Global Aggregates (Micro & Macro)
    total_qs = sum(s["total_questions"] for s in all_summaries)
    total_correct = sum(int(round(s["overall_accuracy"] * s["total_questions"])) for s in all_summaries)
    total_oracle = sum(int(round(s["overall_oracle_recall"] * s["total_questions"])) for s in all_summaries)

    micro_acc = total_correct / total_qs if total_qs else 0.0
    micro_ora = total_oracle / total_qs if total_qs else 0.0
    macro_acc = sum(s["overall_accuracy"] for s in all_summaries) / len(all_summaries) if all_summaries else 0.0
    macro_ora = sum(s["overall_oracle_recall"] for s in all_summaries) / len(all_summaries) if all_summaries else 0.0

    mean_tokens = sum(s["mean_tokens"] * s["total_questions"] for s in all_summaries) / total_qs if total_qs else 0.0
    mean_latency = sum(s["mean_latency_ms"] * s["total_questions"] for s in all_summaries) / total_qs if total_qs else 0.0

    # Category aggregates
    cat_totals = defaultdict(int)
    cat_corrects = defaultdict(int)
    cat_oracles = defaultdict(int)
    for s in all_summaries:
        for c_name, c_data in s["categories"].items():
            cat_totals[c_name] += c_data["total"]
            cat_corrects[c_name] += c_data["correct"]
            cat_oracles[c_name] += c_data["oracle"]

    print("\n" + "=" * 85)
    print("        AM APEX: LOCOMO-10 FULL BENCHMARK FINAL SUMMARY REPORT")
    print("=" * 85)
    print(f"Total Conversations Evaluated:  {len(all_summaries)}")
    print(f"Total Questions Evaluated:      {total_qs}")
    print(f"Micro Overall Accuracy:         {micro_acc * 100:.1f}% ({total_correct}/{total_qs})")
    print(f"Macro Overall Accuracy:         {macro_acc * 100:.1f}%")
    print(f"Micro Oracle Recall:            {micro_ora * 100:.1f}% ({total_oracle}/{total_qs})")
    print(f"Macro Oracle Recall:            {macro_ora * 100:.1f}%")
    print(f"Mean Context Tokens/Q:          {mean_tokens:.1f} tokens/Q (vs Mem0 ~7,000)")
    print(f"Mean Latency:                   {mean_latency:.1f} ms (vs Mem0 ~1,400 ms)")
    print(f"Write LLM Calls:                0 calls (100% Free Deterministic Ingestion)")
    print(f"Total Benchmark Suite Time:     {total_global_time:.1f} s ({total_global_time / 60:.1f} min)")
    print("-" * 85)
    print("GLOBAL CATEGORY-BY-CATEGORY BREAKDOWN:")
    for c_name in sorted(cat_totals.keys()):
        cnt = cat_totals[c_name]
        c_acc = (cat_corrects[c_name] / cnt) * 100 if cnt else 0.0
        c_ora = (cat_oracles[c_name] / cnt) * 100 if cnt else 0.0
        conv_rate = (cat_corrects[c_name] / cat_oracles[c_name]) * 100 if cat_oracles[c_name] else 0.0
        print(f"  * {c_name:<18}: Acc {c_acc:5.1f}% | Oracle {c_ora:5.1f}% | ConvRate {conv_rate:5.1f}% ({cat_corrects[c_name]}/{cnt})")
    print("=" * 85)

    # Aggregate failure taxonomy
    global_failures = defaultdict(int)
    for s in all_summaries:
        for cat, cnt in s.get("failure_taxonomy_v2", {}).items():
            global_failures[cat] += cnt

    print("\n" + "=" * 85)
    print("        FAILURE TAXONOMY v2 BREAKDOWN (8 Causal Categories across 1,986 Qs)")
    print("=" * 85)
    tot_failures = sum(global_failures.values())
    for cat, cnt in sorted(global_failures.items(), key=lambda x: -x[1]):
        pct = cnt / tot_failures * 100 if tot_failures else 0.0
        print(f"  * {cat:<24}: {cnt:4d} ({pct:5.1f}%)")
    print("=" * 85)

    # Save Full Final Report JSON
    full_report_path = RESULTS_DIR / "full_locomo10_report.json"
    with open(full_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_conversations": len(all_summaries),
                "total_questions": total_qs,
                "micro_accuracy": micro_acc,
                "macro_accuracy": macro_acc,
                "micro_oracle_recall": micro_ora,
                "macro_oracle_recall": macro_ora,
                "mean_tokens": mean_tokens,
                "mean_latency_ms": mean_latency,
                "total_time_seconds": total_global_time,
                "category_breakdown": {
                    c: {
                        "total": cat_totals[c],
                        "correct": cat_corrects[c],
                        "oracle": cat_oracles[c],
                    }
                    for c in cat_totals
                },
                "failure_taxonomy_v2": dict(global_failures),
                "conversations": all_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[Saved full comprehensive report to {full_report_path}]")


if __name__ == "__main__":
    main()
