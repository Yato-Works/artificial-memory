"""Temporal (cat-2) failure anatomy: relative prediction vs absolute ground truth.

The strict calendar scorer treats ``GT="The week before 9 June 2023"`` vs
``pred="Last week"`` as wrong by design (frozen protocol).  The fixable part is
system-side: emit absolute calendar expressions derived from the session date
anchors that already travel inside the compiled context.

This script measures that recoverable bucket without any LLM calls.

Usage: python scripts/locomo_temporal_failure_anatomy.py <run_dir>
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REL = re.compile(
    r"\b(?:yesterday|today|last\s+(?:week|month|year|night|summer|spring|winter|fall)|"
    r"next\s+(?:week|month|year)|this\s+(?:week|month|year|summer)|"
    r"a\s+(?:week|month|year|couple of (?:weeks|months|years))|"
    r"\d+\s+(?:days|weeks|months|years)\s+ago|recently|lately|the\s+other\s+day|"
    r"a\s+few\s+(?:days|weeks|months|years))\b",
    re.IGNORECASE,
)
ABS = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                "benchmark_results/locomo10_runs/subject_binding_v1")
    buckets = Counter()
    examples: dict[str, list] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            if r["category"] != 2:
                continue
            ok = score(r["category"], r["ground_truth"], r["predicted_answer"])
            gt, pred = str(r["ground_truth"]), str(r["predicted_answer"])
            if ok:
                buckets["correct"] += 1
                if REL.search(pred) and not ABS.search(pred):
                    buckets["correct_with_relative_only"] += 1
                continue
            if REL.search(pred) and not ABS.search(pred) and ABS.search(gt):
                key = "RECOVERABLE_rel_pred_abs_gt"
            elif REL.search(pred):
                key = "rel_pred_other"
            elif not pred.strip():
                key = "empty_pred"
            elif ABS.search(pred) and ABS.search(gt):
                key = "abs_vs_abs_mismatch"
            elif REL.search(gt) or not ABS.search(gt):
                key = "gt_not_absolute"
            else:
                key = "other"
            buckets[key] += 1
            examples.setdefault(key, []).append((r["question_id"], gt[:44], pred[:60]))

    total = sum(buckets.values())
    print(f"temporal questions: {total} | dir={root}")
    for k, v in buckets.most_common():
        print(f"  {k:<30}{v:>5}  {v / total * 100:>5.1f}%")
    print("\nsamples of RECOVERABLE bucket:")
    for qid, gt, pred in examples.get("RECOVERABLE_rel_pred_abs_gt", [])[:12]:
        print(f"  {qid}  gt={gt!r}\n      pred={pred!r}")
    print("\nsamples of abs_vs_abs_mismatch:")
    for qid, gt, pred in examples.get("abs_vs_abs_mismatch", [])[:8]:
        print(f"  {qid}  gt={gt!r}  pred={pred!r}")


if __name__ == "__main__":
    main()
