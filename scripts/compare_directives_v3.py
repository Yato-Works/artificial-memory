"""Compare directives_v3 vs subject_binding_v2 per category (conv by conv)."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(run, conv):
    return {r["question_id"]: r for r in json.load(
        open(f"benchmark_results/locomo10_runs/{run}/conv_{conv}_results.json",
             encoding="utf-8"))["results"]}


def compare(conv):
    b, n = load("subject_binding_v2", conv), load("directives_v3", conv)
    if not n:
        return
    stat = Counter()
    ex = []
    for qid in b:
        if qid not in n:
            continue
        cat = CAT.get(b[qid]["category"], str(b[qid]["category"]))
        stat[(cat, "n")] += 1
        stat[(cat, "v2")] += bool(b[qid]["is_correct"])
        stat[(cat, "v3")] += bool(n[qid]["is_correct"])
        if b[qid]["is_correct"] and not n[qid]["is_correct"]:
            stat[(cat, "reg")] += 1
            if len(ex) < 5:
                ex.append((qid, str(b[qid]["predicted_answer"])[:38],
                           str(n[qid]["predicted_answer"])[:56],
                           str(b[qid]["ground_truth"])[:28]))
    print(f"--- conv {conv} ---")
    for cat in ("multi-hop", "adversarial", "temporal", "single-hop", "open-domain"):
        c = stat[(cat, "n")]
        if not c:
            continue
        print(f"{cat:<12} n={c:>4} v2={stat[(cat, 'v2')]:>4} v3={stat[(cat, 'v3')]:>4} "
              f"reg={stat[(cat, 'reg')]}")
    for qid, a, p, g in ex:
        print(f"  {qid} v2={a!r} -> v3={p!r} gt={g!r}")


for conv in range(10):
    compare(conv)
