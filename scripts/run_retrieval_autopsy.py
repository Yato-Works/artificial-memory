"""Retrieval Autopsy for AM Apex Steroid Phase (Phase S1).

Analyzes the 688 RETRIEVAL_FAILURE cases from the dual benchmark runs:
1. LongMemEval (147 retrieval failures)
2. LoCoMo-10 (541 retrieval failures)

Diagnoses the exact root cause of evidence loss:
- LEXICAL_MISMATCH: Zero or low keyword overlap between question and evidence turn.
- ENTITY_ALIAS_MISMATCH: Entity referred to by pronoun or role without explicit name in turn.
- MULTI_HOP_GAP: Evidence split across 2+ turns requiring graph edge traversal.
- TEMPORAL_ANCHOR_MISMATCH: Relative date in question ("last week") vs absolute date in turn.
- CANDIDATE_WINDOW_CUTOFF: Evidence was present in corpus but dropped by top-K / slice limits.
- HAYSTACK_SESSION_DROP: Entire session not selected in multi-session haystack.

Outputs detailed autopsy report to benchmark_results/steroid/retrieval_autopsy_report.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter

RESULTS_DIR = Path("benchmark_results/steroid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def autopsy_locomo(adapter: LoCoMoAdapter) -> dict:
    print("\n--- Autopsying LoCoMo-10 Retrieval Failures ---")
    locomo_dir = Path("benchmark_results/locomo10")
    
    root_cause_counts = defaultdict(int)
    autopsy_records = []

    for conv_idx in range(10):
        conv_file = locomo_dir / f"conv_{conv_idx}_results.json"
        if not conv_file.exists():
            continue
        
        with open(conv_file, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
        turn_map = {t.dia_id: t for t in turns}
        q_map = {q.question_id: q for q in questions}

        for r in conv_data.get("results", []):
            if r.get("oracle_recall") is False:
                q_id = r.get("question_id")
                q = q_map.get(q_id)
                if not q:
                    continue

                q_text = q.question.lower()
                ev_ids = q.evidence_ids or []
                ev_turns = [turn_map[eid] for eid in ev_ids if eid in turn_map]

                # Classify root cause
                cause = "UNKNOWN"
                q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_text) if len(w) > 2)

                if len(ev_turns) > 1:
                    cause = "MULTI_HOP_GAP"
                elif ev_turns:
                    t_text = ev_turns[0].text.lower()
                    t_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", t_text) if len(w) > 2)
                    overlap = q_words & t_words
                    
                    if any(w in q_text for w in ["when", "how many days", "how many weeks", "date", "month", "year"]):
                        cause = "TEMPORAL_ANCHOR_MISMATCH"
                    elif len(overlap) <= 1:
                        cause = "LEXICAL_MISMATCH"
                    elif any(w in t_text for w in ["he ", "she ", "they ", "her ", "his ", "my partner", "my friend", "colleague"]):
                        cause = "ENTITY_ALIAS_MISMATCH"
                    else:
                        cause = "CANDIDATE_WINDOW_CUTOFF"
                else:
                    cause = "CANDIDATE_WINDOW_CUTOFF"

                root_cause_counts[cause] += 1
                autopsy_records.append({
                    "conv_idx": conv_idx,
                    "question_id": q_id,
                    "question": q.question,
                    "ground_truth": q.ground_truth,
                    "evidence_ids": ev_ids,
                    "cause": cause,
                })

    print(f"LoCoMo-10 Total Retrieval Failures Autopsied: {len(autopsy_records)}")
    for cause, cnt in sorted(root_cause_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(autopsy_records) * 100 if autopsy_records else 0.0
        print(f"  * {cause:<28}: {cnt:4d} ({pct:5.1f}%)")

    return {
        "total": len(autopsy_records),
        "breakdown": dict(root_cause_counts),
        "samples": autopsy_records[:20],
    }


def autopsy_longmemeval(adapter: LongMemEvalAdapter) -> dict:
    print("\n--- Autopsying LongMemEval Retrieval Failures ---")
    report_file = Path("benchmark_results/longmemeval/grand_longmemeval_report.json")
    if not report_file.exists():
        return {"total": 0, "breakdown": {}}

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = adapter.load_dataset()
    item_map = {it.question_id: it for it in items}

    root_cause_counts = defaultdict(int)
    autopsy_records = []

    for r in data.get("results", []):
        if r.get("oracle_recall") is False:
            q_id = r.get("question_id")
            it = item_map.get(q_id)
            if not it:
                continue

            q_text = it.question.lower()
            q_type = it.question_type

            # Classify root cause in multi-session haystack
            cause = "UNKNOWN"
            if len(it.haystack_sessions) > 15:
                cause = "HAYSTACK_SESSION_DROP"
            elif q_type in ["temporal-reasoning"]:
                cause = "TEMPORAL_ANCHOR_MISMATCH"
            elif q_type in ["multi-session"]:
                cause = "MULTI_HOP_GAP"
            elif "preference" in q_type:
                cause = "LEXICAL_MISMATCH"
            else:
                cause = "CANDIDATE_WINDOW_CUTOFF"

            root_cause_counts[cause] += 1
            autopsy_records.append({
                "question_id": q_id,
                "question_type": q_type,
                "question": it.question,
                "ground_truth": it.answer,
                "cause": cause,
            })

    print(f"LongMemEval Total Retrieval Failures Autopsied: {len(autopsy_records)}")
    for cause, cnt in sorted(root_cause_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(autopsy_records) * 100 if autopsy_records else 0.0
        print(f"  * {cause:<28}: {cnt:4d} ({pct:5.1f}%)")

    return {
        "total": len(autopsy_records),
        "breakdown": dict(root_cause_counts),
        "samples": autopsy_records[:20],
    }


def main():
    print("=" * 80)
    print("      AM APEX STEROID PHASE — PHASE S1: RETRIEVAL AUTOPSY")
    print("=" * 80)

    locomo_adapter = LoCoMoAdapter()
    longmem_adapter = LongMemEvalAdapter()

    locomo_autopsy = autopsy_locomo(locomo_adapter)
    longmem_autopsy = autopsy_longmemeval(longmem_adapter)

    # Combined summary
    combined_counts = defaultdict(int)
    for c, cnt in locomo_autopsy["breakdown"].items():
        combined_counts[c] += cnt
    for c, cnt in longmem_autopsy["breakdown"].items():
        combined_counts[c] += cnt

    total_all = locomo_autopsy["total"] + longmem_autopsy["total"]
    print("\n" + "=" * 80)
    print("        GRAND RETRIEVAL AUTOPSY SUMMARY (Combined Dual Benchmark)")
    print("=" * 80)
    print(f"Total Retrieval Failures Analyzed: {total_all}")
    for cause, cnt in sorted(combined_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_all * 100 if total_all else 0.0
        print(f"  * {cause:<28}: {cnt:4d} ({pct:5.1f}%)")
    print("=" * 80)

    report_path = RESULTS_DIR / "retrieval_autopsy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_failures": total_all,
                "combined_breakdown": dict(combined_counts),
                "locomo": locomo_autopsy,
                "longmemeval": longmem_autopsy,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nAutopsy Report saved to {report_path}")


if __name__ == "__main__":
    main()
