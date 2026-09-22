"""Direct audit: stored rows for adversarial refusals vs the current scorer."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")

rows = {}
for p in sorted(RUN.glob("conv_*_results.json")):
    for r in json.load(open(p, encoding="utf-8"))["results"]:
        rows[r["question_id"]] = r

for qid in ["conv-26-qa-156", "conv-26-qa-159", "conv-26-qa-167", "conv-42-qa-249"]:
    r = rows.get(qid)
    if not r:
        print(f"{qid}: MISSING")
        continue
    print(f"\n{qid} category={r['category']} is_correct={r['is_correct']}")
    print(f"  gt={r['ground_truth']!r}")
    print(f"  pred={r['predicted_answer']!r}")
    print(f"  rescored={score(r['category'], r['ground_truth'], r['predicted_answer'])}")

# Global: how many category-5 rows have empty GT and a refusal-looking answer yet score False?
n = bad = 0
for r in rows.values():
    if r["category"] != 5:
        continue
    gt = str(r["ground_truth"]).lower().strip()
    ans = str(r["predicted_answer"]).lower()
    if not gt:
        n += 1
        if not score(5, r["ground_truth"], r["predicted_answer"]):
            bad += 1
            if bad <= 5:
                print(f"  UNEXPECTED-FALSE {r['question_id']}: pred={ans[:70]!r}")
print(f"\ncategory5 with EMPTY gt: {n} | scored False: {bad}")
