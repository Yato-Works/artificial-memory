"""Quick LoCoMo run comparison table (retrieval-only runs)."""
import json
from pathlib import Path

TAGS = ["baseline_retrieval", "widening_v2", "selection_v3"]
CONVS = range(10)
ROOT = Path("benchmark_results/locomo10_runs")


def load(tag):
    rows = []
    for c in CONVS:
        p = ROOT / tag / f"conv_{c}_results.json"
        if p.exists():
            rows.append(json.load(open(p, encoding="utf-8")))
    return rows


def summarize(tag):
    rows = load(tag)
    if not rows:
        return None
    tot = ora = 0
    tok = 0.0
    cats = {}
    for d in rows:
        s = d["summary"]
        tot += s["total_questions"]
        ora += sum(v["oracle"] for v in s["categories"].values())
        tok += s["mean_tokens"] * s["total_questions"]
        for k, v in s["categories"].items():
            cats.setdefault(k, [0, 0])
            cats[k][0] += v["oracle"]
            cats[k][1] += v["total"]
    return {"convs": len(rows), "n": tot, "oracle": ora, "tok": tok / tot, "cats": cats}


base = summarize("baseline_retrieval")
print(f"{'run':<22}{'convs':>6}{'n':>7}{'oracle':>9}{'tokens/Q':>10}{'delta':>9}")
for tag in TAGS:
    s = summarize(tag)
    if not s:
        print(f"{tag:<22}{'INCOMPLETE':>6}")
        continue
    delta = ""
    if base:
        delta = f"{(s['oracle'] / s['n'] - base['oracle'] / base['n']) * 100:+.2f}pp"
    print(f"{tag:<22}{s['convs']:>6}{s['n']:>7}{s['oracle'] / s['n'] * 100:>8.2f}%"
          f"{s['tok']:>10.1f}{delta:>9}")

print("\nPER-CATEGORY ORACLE RECALL")
cats = sorted({c for t in TAGS for c in (summarize(t) or {"cats": {}})["cats"]})
hdr = f"{'category':<15}" + "".join(f"{t[:14]:>16}" for t in TAGS)
print(hdr)
for c in cats:
    line = f"{c:<15}"
    for t in TAGS:
        s = summarize(t)
        if not s or c not in s["cats"]:
            line += f"{'--':>16}"
        else:
            o, n = s["cats"][c]
            line += f"{o / n * 100:>15.1f}%"
    print(line)
