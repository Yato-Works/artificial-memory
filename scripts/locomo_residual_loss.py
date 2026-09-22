"""Residual oracle-loss analysis from saved LoCoMo run checkpoints (no recompute)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/selection_v3")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

miss = defaultdict(int)
tot = defaultdict(int)
by_cat_miss = defaultdict(int)
by_cat_tot = defaultdict(int)
cat5_gt = defaultdict(int)

for conv in range(10):
    p = RUN / f"conv_{conv}_results.json"
    if not p.exists():
        continue
    d = json.load(open(p, encoding="utf-8"))
    for r in d["results"]:
        name = CAT.get(r["category"], str(r["category"]))
        tot[name] += 1
        if not r["oracle_recall"]:
            miss[name] += 1
            by_cat_miss[name] += 1
            if name == "adversarial":
                gt = (r["ground_truth"] or "")[:40]
                cat5_gt[gt] += 1
        else:
            by_cat_tot[name] += 1

print(f"RUN: {RUN}")
print(f"{'category':<14}{'n':>6}{'oracle':>9}{'miss':>7}{'miss%':>8}")
g_n = g_m = 0
for k in sorted(tot, key=lambda x: -tot[x]):
    n, m = tot[k], miss[k]
    g_n += n
    g_m += m
    print(f"{k:<14}{n:>6}{(n - m) / n * 100:>8.1f}%{m:>7}{m / n * 100:>7.1f}%")
print(f"{'ALL':<14}{g_n:>6}{(g_n - g_m) / g_n * 100:>8.1f}%{g_m:>7}{g_m / g_n * 100:>7.1f}%")

print(f"\nTOTAL residual oracle misses: {g_m} questions")
print("\nADVERSARIAL residual misses by ground-truth kind (top 15):")
for gt, c in sorted(cat5_gt.items(), key=lambda x: -x[1])[:15]:
    print(f"  {c:>4}  {gt!r}")
