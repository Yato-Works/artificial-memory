"""Show oracle-hit-but-wrong answers for a specific conversation checkpoint."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
path = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6
d = json.load(open(path, encoding="utf-8"))
keys = list(d["results"][0].keys())
if len(sys.argv) > 3:
    print("KEYS:", keys)
rows = [r for r in d["results"] if r["oracle_recall"] and not r["is_correct"]]
print(f"oracle-hit but wrong: {len(rows)} / {len(d['results'])}")
for r in rows[:limit]:
    qtext = r.get("question_text") or r.get("q") or r.get("question") or ""
    print(f"\n[{r['question_id']}] cat={r['category']} {qtext[:70]}")
    print(f"  GT  : {str(r.get('ground_truth'))[:90]}")
    print(f"  PRED: {str(r.get('predicted_answer'))[:90]}")
