"""Dump temporal (category 2) failure examples with their evidence lines.

Reads the cached compiled contexts plus a run's results so we can see, for each
wrong answer, what the model said, what the ground truth is, and which context
line (with its session-date marker) carries the ground-truth date words.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 18

cache: dict[str, str] = {}
for line in open("benchmark_results/locomo_context_cache.jsonl", encoding="utf-8"):
    row = json.loads(line)
    cache[row["question_id"]] = row.get("context", "")

rows = []
for p in sorted(RUN.glob("conv_*_results.json")):
    for r in json.load(open(p, encoding="utf-8"))["results"]:
        if r["category"] == 2 and not r["is_correct"]:
            rows.append(r)

print(f"temporal failures in {RUN.name}: {len(rows)}")
shown = 0
for r in rows:
    context = cache.get(r["question_id"], "")
    gt = str(r["ground_truth"])
    pred = str(r["predicted_answer"])
    # month/year tokens of the ground truth
    gt_words = re.findall(r"[A-Za-z]{3,}|\d{4}|\d{1,2}", gt)
    hits = [ln for ln in context.split("\n")
            if sum(1 for w in gt_words if w.lower() in ln.lower()) >= max(1, len(gt_words) - 1)]
    print("\n" + "=" * 100)
    print(f"{r['question_id']}  oracle={r['oracle_recall']}")
    print(f"  GT   : {gt!r}")
    print(f"  PRED : {pred[:150]!r}")
    print(f"  gt-bearing context lines: {len(hits)}")
    for ln in hits[:2]:
        print(f"    {ln[:190]}")
    shown += 1
    if shown >= LIMIT:
        break
