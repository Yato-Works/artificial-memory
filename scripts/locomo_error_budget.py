"""Per-category residual error budget for a re-scored LoCoMo run.

Combines the current-run re-scored errors with the frozen baseline so every
remaining miss can be attributed to a category and an oracle status:

  oracle=YES, wrong  -> COMPOSITION/ANSWER problem (evidence was retrieved)
  oracle=NO,  wrong  -> RETRIEVAL problem (ground-truth turn never entered context)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import rescore  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
    base = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10")
    cur = rescore(run)
    ref = rescore(base)

    agg = defaultdict(lambda: defaultdict(int))
    for qid, r in cur.items():
        cat = CAT.get(r["category"], str(r["category"]))
        a = agg[cat]
        a["n"] += 1
        a["ok"] += int(r["rescored_correct"])
        if not r["rescored_correct"]:
            a["wrong"] += 1
            a["wrong_oracle"] += int(bool(r["oracle_recall"]))
            a["wrong_nooracle"] += int(not r["oracle_recall"])
        b = ref.get(qid)
        if b is not None:
            a["b_ok"] += int(b["rescored_correct"])
            a["fix"] += int(r["rescored_correct"] and not b["rescored_correct"])
            a["reg"] += int(not r["rescored_correct"] and b["rescored_correct"])

    tot = defaultdict(int)
    print(f"run: {run}\nbaseline ref: {base}\n")
    hdr = (f"{'category':<13}{'n':>5}{'acc':>8}{'base':>8}{'delta':>8}"
           f"{'wrong':>7}{'w/ora':>7}{'w/o-ora':>9}{'fix':>6}{'reg':>5}")
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(agg, key=lambda c: -agg[c]["n"]):
        a = agg[cat]
        for k, v in a.items():
            tot[k] += v
        print(f"{cat:<13}{a['n']:>5}{a['ok'] / a['n'] * 100:>7.1f}%{a['b_ok'] / a['n'] * 100:>7.1f}%"
              f"{(a['ok'] - a['b_ok']) / a['n'] * 100:>+7.1f}p{a['wrong']:>7}{a['wrong_oracle']:>7}"
              f"{a['wrong_nooracle']:>9}{a['fix']:>6}{a['reg']:>5}")
    a = tot
    print("-" * len(hdr))
    print(f"{'ALL':<13}{a['n']:>5}{a['ok'] / a['n'] * 100:>7.1f}%{a['b_ok'] / a['n'] * 100:>7.1f}%"
          f"{(a['ok'] - a['b_ok']) / a['n'] * 100:>+7.1f}p{a['wrong']:>7}{a['wrong_oracle']:>7}"
          f"{a['wrong_nooracle']:>9}{a['fix']:>6}{a['reg']:>5}")

    print(f"\nresidual wrong = {a['wrong']} questions")
    print(f"  * oracle-OK (answer/composition problem): {a['wrong_oracle']} "
          f"({a['wrong_oracle'] / a['wrong'] * 100:.1f}%)")
    print(f"  * oracle-MISS (retrieval problem)       : {a['wrong_nooracle']} "
          f"({a['wrong_nooracle'] / a['wrong'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
