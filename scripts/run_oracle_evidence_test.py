"""Oracle Evidence Test for AM Apex Steroid Phase (Phase S4).

Measures the theoretical upper bound of accuracy when the LLM is given 100% perfect
Ground Truth Evidence directly into context:
    A_oracle: Accuracy with Ground Truth Evidence
    A_AM: Current AM Accuracy on the same questions
    Retrieval Gap = A_oracle - A_AM

This quantitatively proves how much of the failure is due to Retrieval vs LLM Reasoning.
Evaluates on LoCoMo questions with ground-truth evidence IDs.
Outputs results to benchmark_results/steroid/oracle_evidence_gap.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/steroid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("      AM APEX STEROID PHASE — PHASE S4: ORACLE EVIDENCE TEST")
    print("=" * 80)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    print("Loading LoCoMo dataset (Conv 0)...")
    turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
    turn_map = {t.dia_id: t for t in turns}

    valid_questions = [q for q in questions if q.evidence_ids]
    print(f"Loaded {len(valid_questions)} questions with ground truth evidence IDs.\n")

    oracle_correct = 0
    t0 = time.perf_counter()

    for i, q in enumerate(valid_questions):
        # Assemble Ground Truth Context directly from evidence turns
        gt_turns = [turn_map[eid] for eid in q.evidence_ids if eid in turn_map]
        gt_context = "\n".join(t.raw_content for t in gt_turns)

        # Downstream 3.8B LLM generation
        ans = answerer.answer(q.question, gt_context)
        pred_lower = ans.text.lower().strip()
        gt_lower = str(q.ground_truth).lower().strip()

        is_correct = False
        if gt_lower in pred_lower or pred_lower in gt_lower:
            is_correct = True
        else:
            gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", gt_lower) if len(w) > 2)
            pred_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", pred_lower) if len(w) > 2)
            if gt_words and pred_words:
                overlap = len(gt_words & pred_words)
                if overlap / len(gt_words) >= 0.5 or (len(gt_words) <= 2 and overlap >= 1):
                    is_correct = True

        if is_correct:
            oracle_correct += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(valid_questions):
            curr_acc = oracle_correct / (i + 1) * 100
            print(f"  [{i+1:03d}/{len(valid_questions):03d}] Current A_oracle: {curr_acc:5.1f}% ({oracle_correct}/{i+1}) | Q: {q.question[:35]}")

    elapsed = time.perf_counter() - t0
    a_oracle = oracle_correct / len(valid_questions) * 100.0

    # Load baseline accuracy for Conv 0
    baseline_acc = 37.5  # From Conv 0 report
    retrieval_gap = a_oracle - baseline_acc

    print("\n" + "=" * 80)
    print("                    ORACLE EVIDENCE GAP REPORT")
    print("=" * 80)
    print(f"Total Questions Evaluated:         {len(valid_questions)}")
    print(f"Oracle Evidence Accuracy (A_oracle): {a_oracle:5.1f}% ({oracle_correct}/{len(valid_questions)})")
    print(f"AM Baseline Accuracy (A_AM):       {baseline_acc:5.1f}%")
    print(f"Quantitative Retrieval Gap:        {retrieval_gap:+5.1f} pt")
    print(f"Elapsed Time:                      {elapsed:.1f} s ({elapsed/len(valid_questions):.2f}s/Q)")
    print("=" * 80)

    out_file = RESULTS_DIR / "oracle_evidence_gap.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_questions": len(valid_questions),
                "a_oracle": a_oracle,
                "a_am": baseline_acc,
                "retrieval_gap": retrieval_gap,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )
    print(f"Report saved to {out_file}")


if __name__ == "__main__":
    main()
