"""LoCoMo Smoke / Experiment Runner (Phase 0: Measurement Harness).

Runs a configurable subset of LoCoMo conversations and saves per-question
results in the SAME schema as ``run_locomo_full_suite.py`` so that
``compare_locomo_runs.py`` can diff any two runs at question_id level.

Modes:
  * LLM mode (default): full answer generation + scoring (frozen phi4-mini).
  * Retrieval-only mode (--retrieval-only): compiles MSC context and measures
    Oracle Recall ONLY.  No LLM calls -> fast iteration on retrieval changes.
  * Failed-only mode (--failed): re-run only failed questions from a previous run.
  * Specific questions mode (--questions/--question-ids): run only specified questions.

Frozen protocol: the scorer, dataset, LLM config, and prompts are untouched.
Only system-side retrieval changes are allowed between runs.
"""

from __future__ import annotations

import argparse
import json
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
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer


def run_conversation(
    adapter: LoCoMoAdapter,
    answerer: OllamaAnswerer | None,
    weights: EvidenceScoreWeights,
    conv_idx: int,
    out_dir: Path,
    retrieval_only: bool = False,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> dict:
    """Evaluate one conversation and save a checkpoint into ``out_dir``."""
    print("\n" + "=" * 75)
    mode = "RETRIEVAL-ONLY (no LLM)" if retrieval_only else "FULL (frozen LLM)"
    if question_ids is not None:
        mode += f" | QUESTIONS: {len(question_ids)}"
    print(f"--- STARTING CONVERSATION {conv_idx} ({mode}) ---")
    print("=" * 75)

    t0_load = time.perf_counter()
    turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
    load_time = time.perf_counter() - t0_load

    sample_id = questions[0].conv_id if questions else f"conv-{conv_idx}"
    
    # Filter questions by question_ids if provided
    if question_ids is not None:
        questions = [q for q in questions if q.question_id in question_ids]
        if not questions:
            print(f"No matching questions for conv {conv_idx}")
            return _empty_summary(conv_idx, sample_id, retrieval_only)
    
    if limit is not None:
        questions = questions[:limit]

    print(f"Sample ID: {sample_id}")
    print(f"Ingested {len(turns)} turns into {len(ir_records)} StructuredIR records in {load_time:.2f}s.")
    print(f"Evaluating {len(questions)} questions...")

    results: list[LoCoMoEvalResult] = []
    t0_eval = time.perf_counter()

    for i, q in enumerate(questions):
        if retrieval_only:
            pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
            if q.evidence_ids:
                oracle_recall = any(ev_id in pcc.context_text for ev_id in q.evidence_ids)
            else:
                oracle_recall = True
            res = LoCoMoEvalResult(
                question_id=q.question_id,
                category=q.category,
                oracle_recall=oracle_recall,
                predicted_answer="",
                ground_truth=q.ground_truth,
                tokens_used=pcc.token_cost,
                latency_ms=0.0,
                is_correct=False,
            )
        else:
            res = adapter.evaluate_question(q, turns, ir_records, answerer, weights=weights)
        results.append(res)

        if (i + 1) % 25 == 0 or (i + 1) == len(questions):
            curr_ora = sum(1 for r in results if r.oracle_recall) / len(results) * 100
            if retrieval_only:
                print(f"  [{i+1:03d}/{len(questions):03d}] Ora: {curr_ora:5.1f}% | Q: {q.question[:40]}")
            else:
                curr_acc = sum(1 for r in results if r.is_correct) / len(results) * 100
                curr_tok = sum(r.tokens_used for r in results) / len(results)
                print(
                    f"  [{i+1:03d}/{len(questions):03d}] Acc: {curr_acc:5.1f}% | "
                    f"Ora: {curr_ora:5.1f}% | Tok/Q: {curr_tok:5.1f} | Q: {q.question[:32]}"
                )

    eval_time = time.perf_counter() - t0_eval
    reader_info: dict | None = None
    if answerer is not None:
        reader_info = {
            "reader_model": answerer.model,
            "num_ctx": answerer.num_ctx,
            "hints_enabled": bool(getattr(adapter, "allow_question_hints", False)),
        }
    return _summarize_and_save(adapter, questions, results, conv_idx, sample_id,
                               turns, ir_records, retrieval_only, eval_time, out_dir,
                               reader_info=reader_info)


def _empty_summary(conv_idx: int, sample_id: str, retrieval_only: bool) -> dict:
    """Return empty summary for conversations with no matching questions."""
    return {
        "conv_idx": conv_idx,
        "sample_id": sample_id,
        "mode": "retrieval_only" if retrieval_only else "full",
        "total_questions": 0,
        "total_turns": 0,
        "total_ir_records": 0,
        "overall_accuracy": 0.0,
        "overall_oracle_recall": 0.0,
        "factual_accuracy": 0.0,
        "factual_oracle_recall": 0.0,
        "mean_tokens": 0.0,
        "mean_latency_ms": 0.0,
        "elapsed_seconds": 0.0,
        "categories": {},
        "failure_taxonomy_v2": {},
        "reader": {
            "reader_model": FROZEN_MODEL,
            "num_ctx": None,
            "hints_enabled": False,
        },
    }


def load_failed_question_ids(results_path: Path) -> dict[int, set[str]]:
    """Load failed question IDs from a previous run, grouped by conversation index."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    failed_by_conv: dict[int, set[str]] = defaultdict(set)
    for r in data.get("results", []):
        if not r.get("is_correct", True):
            # Extract conv_idx from question_id like "conv-26-qa-001"
            qid = r["question_id"]
            if qid.startswith("conv-"):
                try:
                    conv_idx = int(qid.split("-")[1])
                    failed_by_conv[conv_idx].add(qid)
                except (IndexError, ValueError):
                    pass
    return failed_by_conv


def load_question_ids(questions_path: Path) -> set[str]:
    """Load question IDs from a JSON file (list of strings or objects with question_id)."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return {item["question_id"] for item in data if "question_id" in item}
        return set(data)
    return set()


def _summarize_and_save(
    adapter: LoCoMoAdapter,
    questions: list,
    results: list[LoCoMoEvalResult],
    conv_idx: int,
    sample_id: str,
    turns: list,
    ir_records: list,
    retrieval_only: bool,
    eval_time: float,
    out_dir: Path,
    reader_info: dict | None = None,
) -> dict:
    tot_ora = sum(1 for r in results if r.oracle_recall) / len(results) if results else 0.0
    mean_tok = sum(r.tokens_used for r in results) / len(results) if results else 0.0
    if retrieval_only:
        tot_acc = 0.0
        mean_lat = 0.0
    else:
        tot_acc = sum(1 for r in results if r.is_correct) / len(results) if results else 0.0
        mean_lat = sum(r.latency_ms for r in results) / len(results) if results else 0.0

    # Category breakdown
    cat_breakdown: dict[str, dict] = {}
    for r in results:
        c_name = adapter.CATEGORY_NAMES.get(r.category, f"cat-{r.category}")
        cat_breakdown.setdefault(c_name, {"total": 0, "correct": 0, "oracle": 0})
        cat_breakdown[c_name]["total"] += 1
        if r.is_correct:
            cat_breakdown[c_name]["correct"] += 1
        if r.oracle_recall:
            cat_breakdown[c_name]["oracle"] += 1

    factual = [r for r in results if r.category != 5]
    factual_ora = sum(1 for r in factual if r.oracle_recall) / len(factual) if factual else 0.0

    # Failure taxonomy (LLM mode only — needs predicted answers)
    failure_counts: dict[str, int] = {}
    if not retrieval_only:
        classifier = FailureClassifierV2()
        counts: defaultdict = defaultdict(int)
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
                    counts[diag.category.value] += 1
        failure_counts = dict(counts)

    summary = {
        "conv_idx": conv_idx,
        "sample_id": sample_id,
        "mode": "retrieval_only" if retrieval_only else "full",
        "total_questions": len(questions),
        "total_turns": len(turns),
        "total_ir_records": len(ir_records),
        "overall_accuracy": tot_acc,
        "overall_oracle_recall": tot_ora,
        "factual_accuracy": tot_acc,
        "factual_oracle_recall": factual_ora,
        "mean_tokens": mean_tok,
        "mean_latency_ms": mean_lat,
        "elapsed_seconds": eval_time,
        "categories": cat_breakdown,
        "failure_taxonomy_v2": failure_counts,
        "reader": reader_info or {
            "reader_model": FROZEN_MODEL,
            "num_ctx": None,
            "hints_enabled": bool(getattr(adapter, "allow_question_hints", False)),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"conv_{conv_idx}_results.json"
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

    n_ora = sum(1 for r in results if r.oracle_recall)
    print(f"\n>> CONV {conv_idx} COMPLETED in {eval_time:.1f}s:")
    print(f"   * Oracle Recall:  {tot_ora * 100:.1f}% ({n_ora}/{len(results)})")
    if not retrieval_only:
        n_cor = sum(1 for r in results if r.is_correct)
        print(f"   * Accuracy:       {tot_acc * 100:.1f}% ({n_cor}/{len(results)})")
        print(f"   * Mean Latency:   {mean_lat:.1f} ms")
    print(f"   * Mean Tokens/Q:  {mean_tok:.1f} tok")
    print(f"   * Saved to:       {out_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo Smoke / Experiment Runner")
    parser.add_argument("--convs", type=str, default="0,3,5",
                        help="Comma-separated conversation indices (default smoke set: 0,3,5)")
    parser.add_argument("--tag", type=str, default="smoke",
                        help="Run tag; results go to benchmark_results/locomo10_runs/<tag>/")
    parser.add_argument("--out", type=str, default=None,
                        help="Explicit output dir (overrides --tag)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Oracle Recall only; no LLM calls (fast retrieval iteration)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions per conversation")
    parser.add_argument("--window-cap", type=int, default=None,
                        help="Override the MSC selection window cap (A/B only; "
                             "production default is untouched when omitted)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the reader model tag (A/B only; frozen default when omitted)")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Override the Ollama context window (A/B only). Needed for large model "
                             "tags on small-VRAM GPUs; omitted = Ollama default (frozen).")

    parser.add_argument("--hints", action='store_true',
                        help="DIAGNOSTIC ONLY: enable per-question hint tables (answer injection); production stays clean.")
    parser.add_argument("--token-bonus", type=int, default=None,
                        help="Override the MSC rescue token bonus per promoted unit "
                             "(A/B only; production default is untouched when omitted)")
    
    # New arguments for failure-driven development
    parser.add_argument("--failed", type=str, default=None,
                        help="Path to previous run results.json; re-run only failed questions")
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to JSON file with question IDs to run (list of strings or objects with question_id)")
    parser.add_argument("--question-ids", type=str, default=None,
                        help="Comma-separated question IDs to run directly")
    parser.add_argument("--exclude-conv", type=str, default=None,
                        help="Comma-separated conversation indices to exclude (e.g., '3,7' for holdout)")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output file path for aggregated results (overrides --out/--tag)")
    args = parser.parse_args()

    # Determine question_ids to run
    question_ids: set[str] | None = None
    if args.question_ids:
        question_ids = {qid.strip() for qid in args.question_ids.split(",") if qid.strip()}
    elif args.questions:
        question_ids = load_question_ids(Path(args.questions))
    elif args.failed:
        failed_by_conv = load_failed_question_ids(Path(args.failed))
        # Flatten all failed question IDs
        question_ids = set()
        for ids in failed_by_conv.values():
            question_ids.update(ids)
        print(f"Loaded {len(question_ids)} failed questions from {args.failed}")

    # Determine convs to run
    convs = [int(c) for c in args.convs.split(",") if c.strip() != ""]
    
    # Apply exclude-conv filter
    if args.exclude_conv:
        exclude = {int(c) for c in args.exclude_conv.split(",") if c.strip() != ""}
        convs = [c for c in convs if c not in exclude]
        print(f"Excluding holdout conversations: {sorted(exclude)}")
    
    # If question_ids specified, we only need convs that have those questions
    # but we still iterate all convs and let run_conversation filter
    
    out_dir = Path(args.out) if args.out else Path("benchmark_results/locomo10_runs") / args.tag

    mode = "RETRIEVAL-ONLY" if args.retrieval_only else "FULL (frozen LLM)"
    if args.failed:
        mode += " | FAILED-ONLY"
    if args.questions or args.question_ids:
        mode += " | SPECIFIC-QUESTIONS"
    print("=" * 80)
    print("      LOCOMO EXPERIMENT RUNNER (Phase 0 Measurement Harness)")
    print("=" * 80)
    print(f"Mode: {mode} | Convs: {convs} | Out: {out_dir}")
    if question_ids:
        print(f"Target questions: {len(question_ids)}")
    print("Frozen protocol: scorer / dataset / LLM config untouched.")
    print("=" * 80)

    adapter = LoCoMoAdapter()
    if args.window_cap is not None:
        adapter.compiler.selection_window_cap = args.window_cap
    if args.token_bonus is not None:
        adapter.compiler.rescue_token_bonus_per_unit = args.token_bonus
    if args.hints:
        adapter.allow_question_hints = True
        print('WARNING: --hints ENABLED (diagnostic answer injection; not for publication)')
    if args.window_cap is not None or args.token_bonus is not None:
        print(f"A/B OVERRIDE: window_cap={adapter.compiler.selection_window_cap} "
              f"token_bonus={adapter.compiler.rescue_token_bonus_per_unit}")
    answerer = None if args.retrieval_only else OllamaAnswerer(
        model=args.model if args.model else FROZEN_MODEL,
        num_ctx=args.num_ctx)
    if args.model:
        print(f"A/B READER OVERRIDE: model={args.model} num_ctx={args.num_ctx}")
    weights = EvidenceScoreWeights()

    summaries = []
    t0 = time.perf_counter()
    for c_idx in convs:
        summaries.append(
            run_conversation(
                adapter, answerer, weights, c_idx,
                out_dir=out_dir,
                retrieval_only=args.retrieval_only,
                limit=args.limit,
                question_ids=question_ids,
            )
        )

    # Aggregate
    total_qs = sum(s["total_questions"] for s in summaries)
    total_ora = sum(int(round(s["overall_oracle_recall"] * s["total_questions"])) for s in summaries)
    micro_ora = total_ora / total_qs if total_qs else 0.0
    print("\n" + "=" * 80)
    print(f"RUN TAG: {out_dir.name} | Questions: {total_qs}")
    print(f"Micro Oracle Recall: {micro_ora * 100:.2f}% ({total_ora}/{total_qs})")
    if not args.retrieval_only:
        total_cor = sum(int(round(s["overall_accuracy"] * s["total_questions"])) for s in summaries)
        print(f"Micro Accuracy:      {total_cor / total_qs * 100:.2f}% ({total_cor}/{total_qs})")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    print("=" * 80)

    # Also save aggregated results if --output specified
    if args.output and not args.retrieval_only:
        all_results = []
        for c_idx in convs:
            conv_file = out_dir / f"conv_{c_idx}_results.json"
            if conv_file.exists():
                with open(conv_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_results.extend(data.get("results", []))
        
        agg_data = {
            "summary": {
                "total_questions": total_qs,
                "overall_accuracy": total_cor / total_qs if total_qs else 0.0,
                "overall_oracle_recall": micro_ora,
                "mean_tokens": sum(s["mean_tokens"] * s["total_questions"] for s in summaries) / total_qs if total_qs else 0.0,
                "mean_latency_ms": sum(s["mean_latency_ms"] * s["total_questions"] for s in summaries) / total_qs if total_qs else 0.0,
            },
            "results": all_results
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(agg_data, f, ensure_ascii=False, indent=2)
        print(f"Aggregated results saved to: {output_path}")


if __name__ == "__main__":
    main()



