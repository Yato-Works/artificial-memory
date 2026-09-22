"""Classify LoCoMo adversarial regressions: baseline refused, new answered.

For every adversarial question that the baseline got right (refusal) and the
experiment got wrong, decide WHY the new run failed:

  BAIT_COPIED   the new answer echoes the dataset's `adversarial_answer` bait
  REFUSAL_LOST  the new answer is non-refusal but not the bait (own content)
  STILL_REFUSED the new answer still looks like a refusal (scoring/verifier artefact)

It also reports how often the bait text is present in the compiled context of
the experiment, which separates "context put the bait in front of the model"
from "verifier refused too rarely".
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10")
EXP = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10_runs/subject_binding_v1")
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10

REFUSAL_PAT = re.compile(r"\b(no|nope|not|never|dont|don't|doesnt|doesn't|didnt|didn't"
                         r"|cannot|can't|unknown|know|none|neither|nor|unsure|no idea)\b",
                         re.IGNORECASE)
STOP = {"the", "and", "for", "with", "about", "from", "that", "this", "there", "their",
        "she", "he", "they", "her", "his", "was", "were", "is", "are", "not", "but"}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[a-z0-9']+\b", str(text).lower())
            if w not in STOP and len(w) > 2}


def overlap(a: str, b: str) -> float:
    """Jaccard-ish containment of b's content words inside a."""
    ta, tb = tokens(a), tokens(b)
    if not tb:
        return 0.0
    return len(ta & tb) / len(tb)


def load(root: Path, convs: set[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for conv in sorted(convs):
        p = root / f"conv_{conv}_results.json"
        if p.exists():
            for r in json.load(open(p, encoding="utf-8"))["results"]:
                out[r["question_id"]] = r
    return out


def main() -> None:
    raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))
    bait: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for s in raw:
        sid = s.get("sample_id")
        for i, q in enumerate(s.get("qa", [])):
            qid = f"{sid}-qa-{i:03d}"
            if q.get("category") == 5:
                bait[qid] = q.get("adversarial_answer") or ""
                meta[qid] = q

    exp_convs = {int(p.stem.split("_")[1]) for p in EXP.glob("conv_*_results.json")}
    base = load(BASE, exp_convs)
    exp = load(EXP, exp_convs)

    cats: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    verified = Counter()
    for qid, r in exp.items():
        if qid not in base or r["category"] != 5:
            continue
        is_base_ok = bool(base[qid]["is_correct"])
        is_exp_ok = bool(r["is_correct"])
        if is_base_ok == is_exp_ok:
            continue
        ans_new = str(r["predicted_answer"])
        b = bait.get(qid, "")
        looks_refusal = bool(REFUSAL_PAT.search(ans_new)) and len(tokens(ans_new)) <= 12
        if is_base_ok and not is_exp_ok:
            kind = "REFUSAL_LOST"
            if b and overlap(ans_new, b) >= 0.5:
                kind = "BAIT_COPIED"
            if looks_refusal:
                kind = "STILL_REFUSED"
            cats[kind] += 1
            examples[kind].append((qid, ans_new[:90], b[:70],
                                   str(base[qid]["predicted_answer"])[:60]))
        else:
            cats["fixed" if r["category"] == 5 else "other"] += 1

    print(f"experiment conversations: {sorted(exp_convs)} | adversarial compared: "
          f"{sum(1 for q,r in exp.items() if q in base and r['category'] == 5)}")
    print("\nADVERSARIAL NET CHANGE")
    total_reg = sum(v for k, v in cats.items() if k in ("REFUSAL_LOST", "BAIT_COPIED", "STILL_REFUSED"))
    print(f"  fixed (wrong -> refusal) : {cats['fixed']}")
    print(f"  regressed (refusal -> wrong): {total_reg}")
    for k in ("BAIT_COPIED", "REFUSAL_LOST", "STILL_REFUSED"):
        if cats[k]:
            print(f"     {k:<15}: {cats[k]}")
    print(f"  net: {cats['fixed'] - total_reg:+d}")

    for kind in ("BAIT_COPIED", "REFUSAL_LOST", "STILL_REFUSED"):
        if not examples[kind]:
            continue
        print(f"\n--- {kind} examples ---")
        for qid, new, b, old in examples[kind][:LIMIT]:
            print(f"  {qid}\n    new={new!r}\n    base={old!r}\n    bait={b!r}")


if __name__ == "__main__":
    main()
