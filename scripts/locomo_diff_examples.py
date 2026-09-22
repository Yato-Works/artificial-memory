"""Show representative regressions/fixes from a run diff (pure JSON, no recompute)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10")
EXP = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10_runs/selection_v3_llm")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


base, exp = load(BASE), load(EXP)
reg = defaultdict(list)
fix = defaultdict(list)
for qid, e in exp.items():
    b = base.get(qid)
    if b is None:
        continue
    cat = CAT.get(int(b["category"]), str(b["category"]))
    if b["is_correct"] and not e["is_correct"]:
        reg[cat].append((qid, b, e))
    elif e["is_correct"] and not b["is_correct"]:
        fix[cat].append((qid, b, e))

for cat in sorted(set(list(reg) + list(fix))):
    print(f"\n=== {cat}: regressions={len(reg[cat])} fixes={len(fix[cat])} ===")
    for qid, b, e in reg[cat][:5]:
        print(f"  REG {qid} gt={b['ground_truth'][:60]!r}\n"
              f"      base_ans={b['predicted_answer'][:70]!r}\n"
              f"      new_ans ={e['predicted_answer'][:70]!r} tok={b['tokens_used']}->{e['tokens_used']}")
    for qid, b, e in fix[cat][:3]:
        print(f"  FIX {qid} gt={b['ground_truth'][:60]!r}\n"
              f"      base_ans={b['predicted_answer'][:70]!r}\n"
              f"      new_ans ={e['predicted_answer'][:70]!r}")
