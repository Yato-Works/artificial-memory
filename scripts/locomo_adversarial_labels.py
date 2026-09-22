"""Raw LoCoMo category-5 (adversarial) label inspection (ground truth semantics)."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))

keys = Counter()
for s in raw:
    for q in s.get("qa", []):
        keys[tuple(sorted(q.keys()))] += 1
print("QA key sets:", keys)

shown = 0
for s in raw:
    for i, q in enumerate(s.get("qa", [])):
        if q.get("category") != 5:
            continue
        if shown >= 6:
            break
        print(f"\nsample={s.get('sample_id')} idx={i} keys={sorted(q.keys())}")
        print(f"  question          = {q.get('question')!r}")
        print(f"  answer            = {q.get('answer')!r}")
        print(f"  adversarial_answer= {q.get('adversarial_answer')!r}")
        print(f"  evidence          = {q.get('evidence')!r}")
        shown += 1
    if shown >= 6:
        break

# How many category-5 questions have a non-null 'answer'?
tot = null_ans = nonnull = 0
for s in raw:
    for q in s.get("qa", []):
        if q.get("category") != 5:
            continue
        tot += 1
        if q.get("answer") in (None, ""):
            null_ans += 1
        else:
            nonnull += 1
print(f"\ncategory-5 total={tot} null/empty answer={null_ans} non-null answer={nonnull}")

# And what the stored run recorded as ground_truth for category 5.
stored = Counter()
for c in range(10):
    try:
        d = json.load(open(f"benchmark_results/locomo10/conv_{c}_results.json", encoding="utf-8"))
    except FileNotFoundError:
        continue
    for r in d["results"]:
        if r["category"] == 5:
            stored["empty" if not r["ground_truth"] else "nonempty"] += 1
print("stored run ground_truth for category 5:", dict(stored))
