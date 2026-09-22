"""Adversarial (cat-5) regression anatomy: baseline refusal vs new bait answer.

Adversarial questions have NO ground-truth answer (444/446).  The scorer rewards
any answer containing a refusal marker (substring 'no' included).  A regression
therefore means: the baseline answer contained a refusal marker and the new one
does not - i.e. the wider context made the model adopt the bait.

Usage: python scripts/locomo_adversarial_regress_anatomy.py <base_dir> <exp_dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REFUSAL = ["i don't know", "not mentioned", "unknown", "unclear", "no information", "none", "no"]


def is_ok(pred: str) -> bool:
    p = str(pred).lower().strip()
    return any(w in p for w in REFUSAL)


def load(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def main() -> None:
    base = load(Path(sys.argv[1]))
    exp = load(Path(sys.argv[2]))
    regressions = []
    fixes = []
    for qid in sorted(set(base) & set(exp)):
        b, e = base[qid], exp[qid]
        if b["category"] != 5:
            continue
        bo, eo = is_ok(b["predicted_answer"]), is_ok(e["predicted_answer"])
        if bo and not eo:
            regressions.append((qid, e))
        elif eo and not bo:
            fixes.append((qid, e))

    print(f"adversarial regressions: {len(regressions)} | fixes: {len(fixes)}")
    print("\n--- NEW answer markers on regressions (why they are scored wrong) ---")
    pat = Counter()
    for _, e in regressions:
        p = str(e["predicted_answer"]).lower()
        pat["empty_answer"] += (not p.strip())
        if p.strip():
            pat[f"len_{min(4, len(p.split()))}w"] += 1
    for k, v in pat.most_common():
        print(f"  {k:<24}{v:>5}")

    print("\n--- sample regressions (baseline refusal -> new answer) ---")
    for qid, e in regressions[:12]:
        b = base[qid]
        print(f"  {qid}")
        print(f"    base: {str(b['predicted_answer'])[:90]!r}")
        print(f"    new : {str(e['predicted_answer'])[:90]!r}  tokens={e['tokens_used']}")

    print("\n--- sample fixes (baseline bait -> new refusal) ---")
    for qid, e in fixes[:6]:
        b = base[qid]
        print(f"  {qid}")
        print(f"    base: {str(b['predicted_answer'])[:90]!r}")
        print(f"    new : {str(e['predicted_answer'])[:90]!r}")


if __name__ == "__main__":
    main()
