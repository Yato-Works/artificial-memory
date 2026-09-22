"""List regressions (baseline correct -> new wrong) with GT/pred for inspection."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
base_root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10")
exp_root = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10_runs/selection_v3_llm")
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 12
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(root):
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


b, e = load(base_root), load(exp_root)
regs = [q for q in sorted(set(b) & set(e)) if b[q]["is_correct"] and not e[q]["is_correct"]]
print(f"regressions: {len(regs)}")
for qid in regs[:limit]:
    r = e[qid]
    print(f"\n[{qid}] {CAT.get(r['category'], r['category'])} oracle={r['oracle_recall']} "
          f"tok={r['tokens_used']}")
    print(f"  GT  : {str(r['ground_truth'])[:80]}")
    print(f"  PRED: {str(r['predicted_answer'])[:80]}")
    print(f"  BASE: {str(b[qid]['predicted_answer'])[:80]}")
