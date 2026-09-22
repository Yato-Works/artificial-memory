"""Resolve the refusal-scoring contradiction on adversarial (cat 5) rows.

Reproduces the production scorer branch for empty ground truth and reports rows
where the predicted answer contains a refusal marker yet ``is_correct`` is False.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
MARKERS = ["i don't know", "not mentioned", "unknown", "unclear", "no information", "none", "no"]


def prod_scorer(gt: str, ans: str) -> bool:
    gt_lower = str(gt).lower().strip()
    ans_lower = str(ans).lower().strip()
    if not gt_lower:
        return any(w in ans_lower for w in MARKERS)
    return gt_lower in ans_lower or ans_lower in gt_lower


bad = []
ok = 0
mismatch = []
for p in sorted(RUN.glob("conv_*_results.json")):
    d = json.load(open(p, encoding="utf-8"))
    for r in d["results"]:
        if r["category"] != 5:
            continue
        ans = r["predicted_answer"] or ""
        recomputed = prod_scorer(r["ground_truth"], ans)
        if recomputed != r["is_correct"]:
            mismatch.append((r["question_id"], r["is_correct"], recomputed, ans[:70]))
        if recomputed:
            ok += 1
        else:
            bad.append((r["question_id"], r["ground_truth"], ans[:70]))

print(f"run: {RUN}")
print(f"adversarial rows: {ok + len(bad)} | refusal-scored correct: {ok} | wrong: {len(bad)}")
print(f"stored-vs-recomputed mismatches: {len(mismatch)}")
for m in mismatch[:10]:
    print(f"  MISMATCH {m[0]} stored={m[1]} recomputed={m[2]} ans={m[3]!r}")
print("\nfirst wrong rows:")
for qid, gt, ans in bad[:15]:
    print(f"  {qid} gt={gt!r} ans={ans!r}")
