"""Measure Evidence Recall Curve (Phase S3).

Quantifies evidence recall progression across expansion hops:
    R_0 (Wide Slicing) -> R_1 (1-hop) -> R_2 (2-hop) -> R_3 (3-hop)

Compared directly against Frozen Baseline Candidate Gate Recall.
Evaluates on all LoCoMo conversations with ground-truth evidence IDs.
Outputs quantitative curve to benchmark_results/steroid/evidence_recall_curve.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.steroid.evidence_evaluator import EvidenceEvaluator
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler

RESULTS_DIR = Path("benchmark_results/steroid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("      AM APEX STEROID PHASE — PHASE S3: EVIDENCE RECALL CURVE")
    print("=" * 80)

    adapter = LoCoMoAdapter()
    evaluator = EvidenceEvaluator()
    frozen_compiler = MinimumSufficientContextCompiler()

    print("Loading LoCoMo dataset for Evidence Recall analysis...")
    all_turns = []
    all_questions = []
    all_ir = []

    # Evaluate across Conv 0-2 (balanced sample of ~500 questions)
    for c_idx in range(3):
        turns, questions, ir_records = adapter.load_conversation(conv_idx=c_idx)
        all_turns.extend(turns)
        all_questions.extend(questions)
        all_ir.extend(ir_records)

    valid_questions = [q for q in all_questions if q.evidence_ids]
    print(f"Loaded {len(valid_questions)} questions with ground-truth evidence IDs.\n")

    # 1. Baseline Frozen Gate Recall
    print("Measuring Frozen Baseline Candidate Gate Recall...")
    baseline_hits = 0
    t0_base = time.perf_counter()
    for q in valid_questions:
        # Check what the frozen compiler's reconstructor gets
        _, candidate_units = frozen_compiler.reconstructor.reconstruct_world(q.question, all_ir)
        cand_text = " ".join(u.ir.raw_content for u in candidate_units[:8])
        if any(eid in cand_text for eid in q.evidence_ids):
            baseline_hits += 1
    base_recall = baseline_hits / len(valid_questions) * 100.0
    print(f"Frozen Baseline Candidate Recall: {base_recall:.1f}% ({baseline_hits}/{len(valid_questions)}) in {time.perf_counter() - t0_base:.1f}s\n")

    # 2. Steroid Multi-Hop Curve (R_0 to R_3)
    print("Measuring Steroid Evidence Recall Curve (R_0 -> R_3)...")
    t0_curve = time.perf_counter()
    curve = evaluator.evaluate_locomo_curve(valid_questions, all_ir, max_hops=3)
    curve_time = time.perf_counter() - t0_curve

    print("\n" + "=" * 80)
    print("                    EVIDENCE RECALL CURVE TABLE")
    print("=" * 80)
    print(f"{'Expansion Stage':<28} | {'Recall':<10} | {'Found / Total':<15} | {'Delta vs Baseline':<18}")
    print("-" * 80)
    print(f"{'Frozen Baseline (Top-8 Slice)':<28} | {base_recall:5.1f}%    | {baseline_hits:4d} / {len(valid_questions):<6d} | (baseline)")
    
    stage_names = [
        "R_0: Wide Slicing (Union)",
        "R_1: 1-Hop Graph Expansion",
        "R_2: 2-Hop Graph Expansion",
        "R_3: 3-Hop Graph Expansion",
    ]
    for pt in curve.points:
        delta = pt.recall - base_recall
        name = stage_names[pt.hop] if pt.hop < len(stage_names) else f"R_{pt.hop}"
        print(f"{name:<28} | {pt.recall:5.1f}%    | {pt.found_count:4d} / {pt.total_evaluated:<6d} | {delta:+5.1f}pt")
    print("=" * 80)

    # ASCII Graph
    print("\n                    ASCII EVIDENCE RECALL CURVE")
    print("   Recall %")
    max_val = max(pt.recall for pt in curve.points)
    for y in range(80, 20, -10):
        line = f"   {y:3d}% | "
        for pt in curve.points:
            line += "  *   " if abs(pt.recall - y) < 5 else "      "
        print(line)
    print("        +----------------------------------------")
    print("            R_0 (Wide)  R_1 (1-hop) R_2 (2-hop) R_3 (3-hop)\n")

    out_file = RESULTS_DIR / "evidence_recall_curve.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_questions_evaluated": len(valid_questions),
                "frozen_baseline_recall": base_recall,
                "curve_points": [
                    {
                        "hop": pt.hop,
                        "stage": stage_names[pt.hop],
                        "recall": pt.recall,
                        "found": pt.found_count,
                        "total": pt.total_evaluated,
                        "delta_vs_baseline": pt.recall - base_recall,
                    }
                    for pt in curve.points
                ],
                "delta_wide_to_hop3": curve.delta_recall,
                "elapsed_seconds": curve_time,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
