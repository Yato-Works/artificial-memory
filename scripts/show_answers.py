"""Print full baseline/experiment answers for selected question ids."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path("benchmark_results/locomo10")
EXP = Path("benchmark_results/locomo10_runs/selection_v3_llm")
ids = sys.argv[1:]


def find(root: Path, qid: str):
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            if r["question_id"] == qid:
                return r
    return None


for qid in ids:
    b = find(BASE, qid)
    e = find(EXP, qid)
    print(f"=== {qid}")
    if b:
        print(f"  base correct={b['is_correct']} ora={b['oracle_recall']} cat={b['category']}")
        print(f"  base gt={b['ground_truth']!r}")
        print(f"  base ans={b['predicted_answer']!r}")
    if e:
        print(f"  exp  correct={e['is_correct']} ora={e['oracle_recall']}")
        print(f"  exp  ans={e['predicted_answer']!r}")
