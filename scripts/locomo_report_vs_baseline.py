"""LoCoMo run vs frozen baseline markdown report (accuracy + oracle + categories)."""
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
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        d = json.load(open(p, encoding="utf-8"))
        for r in d["results"]:
            out[r["question_id"]] = r
    return out


base, exp = load(BASE), load(EXP)
common = sorted(set(base) & set(exp))
print(f"common questions: {len(common)} (baseline {len(base)}, experiment {len(exp)})")

agg = defaultdict(lambda: {"n": 0, "b_acc": 0, "e_acc": 0, "b_ora": 0, "e_ora": 0,
                           "fix": 0, "reg": 0})
tok_b = tok_e = 0
for qid in common:
    b, e = base[qid], exp[qid]
    cat = CAT.get(b["category"], str(b["category"]))
    a = agg[cat]
    a["n"] += 1
    a["b_acc"] += bool(b["is_correct"])
    a["e_acc"] += bool(e["is_correct"])
    a["b_ora"] += bool(b["oracle_recall"])
    a["e_ora"] += bool(e["oracle_recall"])
    if e["is_correct"] and not b["is_correct"]:
        a["fix"] += 1
    if b["is_correct"] and not e["is_correct"]:
        a["reg"] += 1
    tok_b += b["tokens_used"]
    tok_e += e["tokens_used"]

n = len(common)
tb = sum(a["b_acc"] for a in agg.values())
te = sum(a["e_acc"] for a in agg.values())
ob = sum(a["b_ora"] for a in agg.values())
oe = sum(a["e_ora"] for a in agg.values())
print(f"\nOVERALL accuracy {tb / n * 100:.2f}% -> {te / n * 100:.2f}% "
      f"({(te - tb) / n * 100:+.2f}pp)")
print(f"OVERALL oracle   {ob / n * 100:.2f}% -> {oe / n * 100:.2f}% "
      f"({(oe - ob) / n * 100:+.2f}pp)")
print(f"tokens/Q         {tok_b / n:.1f} -> {tok_e / n:.1f}")

print(f"\n| category | n | baseline acc | new acc | delta | baseline oracle | new oracle | delta | fix | reg |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for cat in sorted(agg, key=lambda c: -agg[c]["n"]):
    a = agg[cat]
    c = a["n"]
    print(f"| {cat} | {c} | {a['b_acc'] / c * 100:.1f}% | {a['e_acc'] / c * 100:.1f}% | "
          f"{(a['e_acc'] - a['b_acc']) / c * 100:+.1f}pp | {a['b_ora'] / c * 100:.1f}% | "
          f"{a['e_ora'] / c * 100:.1f}% | {(a['e_ora'] - a['b_ora']) / c * 100:+.1f}pp | "
          f"{a['fix']} | {a['reg']} |")
print(f"| **ALL** | {n} | {tb / n * 100:.1f}% | {te / n * 100:.1f}% | "
      f"{(te - tb) / n * 100:+.1f}pp | {ob / n * 100:.1f}% | {oe / n * 100:.1f}% | "
      f"{(oe - ob) / n * 100:+.1f}pp | "
      f"{sum(a['fix'] for a in agg.values())} | {sum(a['reg'] for a in agg.values())} |")
