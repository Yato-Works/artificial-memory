"""Check whether the cached contexts match a given run's token profile."""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
cache = Path("benchmark_results/locomo_context_cache.jsonl")
lens = []
cats = {}
with open(cache, encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        n = len((d.get("context") or "").split())
        lens.append(n)
        cats.setdefault(str(d.get("category")), []).append(n)
print(f"cache rows={len(lens)} mean_words={statistics.mean(lens):.0f} "
      f"median={statistics.median(lens):.0f} max={max(lens)}")
for k in sorted(cats):
    print(f"  cat {k}: n={len(cats[k]):>4} mean={statistics.mean(cats[k]):>8.0f}")

run = Path("benchmark_results/locomo10_runs")
for d in ("locomo10", "subject_binding_v1"):
    toks = []
    for p in sorted((run / d if d != "locomo10" else Path("benchmark_results/locomo10"))
                    .glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            toks.append(r["tokens_used"])
    print(f"{d}: n={len(toks)} mean_tokens={statistics.mean(toks):.0f}")
