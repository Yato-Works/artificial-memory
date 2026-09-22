"""Inspect raw LoCoMo adversarial (category 5) question structure."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))
keys = Counter()
shown = 0
empty_answer = 0
total = 0
for sample in raw:
    for q in sample.get("qa", []):
        if q.get("category") != 5:
            continue
        total += 1
        keys[tuple(sorted(q.keys()))] += 1
        if not str(q.get("answer", "")).strip():
            empty_answer += 1
        if shown < 6:
            shown += 1
            print(json.dumps(q, ensure_ascii=False, indent=1)[:900])
            print("-" * 70)

print(f"total cat5 questions: {total}")
print(f"empty answer field : {empty_answer}")
print("key signatures:")
for k, c in keys.most_common():
    print(f"  {c:>5}  {k}")
