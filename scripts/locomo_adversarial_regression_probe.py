"""Adversarial (category 5) regression attribution.

Compares the frozen baseline against a current run using ONE identical scorer
(``rescore_locomo_run.score``) and classifies every adversarial regression
(base correct -> new wrong) by the *shape* of the new answer:

  REFUSAL        new answer refuses while the ground truth is a real content
                 answer  -> over-abstention (gate / verifier too eager)
  WRONG_CONTENT  new answer asserts content that misses the ground truth
  EMPTY          empty answer

It also reports, for each bucket, whether the ground-truth content words are
present in the compiled context, which separates "retrieval lost the evidence"
from "generation/gating failed".
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import CAT, score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REFUSAL_MARKERS = (
    "i don't know", "i dont know", "not mentioned", "unknown", "unclear",
    "no information", "none", "not specified", "cannot determine",
    "does not mention", "doesn't mention", "not in the conversation",
    "no record", "cannot be determined", "insufficient information",
)


def is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(m in t for m in REFUSAL_MARKERS)


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            r = dict(r)
            r["ok"] = score(r["category"], r["ground_truth"], r["predicted_answer"])
            out[r["question_id"]] = r
    return out


def main() -> None:
    base = load(Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10"))
    exp = load(Path(sys.argv[2] if len(sys.argv) > 2
                    else "benchmark_results/locomo10_runs/subject_binding_v1"))
    common = sorted(set(base) & set(exp))
    cats = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    cat5 = [q for q in common if base[q]["category"] == 5]

    for qid in cat5:
        b, e = base[qid], exp[qid]
        cats["n"] += 1
        if b["ok"] and e["ok"]:
            cats["both_ok"] += 1
            kind = "both_ok"
        elif e["ok"] and not b["ok"]:
            cats["fixed"] += 1
            kind = "fixed"
        elif b["ok"] and not e["ok"]:
            cats["regressed"] += 1
            kind = "regressed"
        else:
            cats["both_wrong"] += 1
            kind = "both_wrong"

        ans = str(e["predicted_answer"]).strip()
        gt = str(e["ground_truth"]).strip()
        if not ans:
            shape = "EMPTY"
        elif is_refusal(ans):
            shape = "REFUSAL"
        else:
            shape = "WRONG_CONTENT"
        buckets[kind][shape] += 1
        buckets[kind]["gt_empty" if not gt else "gt_content"] += 1
        buckets[kind]["had_oracle" if e["oracle_recall"] else "no_oracle"] += 1
        if kind == "regressed" and len(examples[shape]) < 6:
            examples[shape].append((qid, gt, ans[:70]))

    print(f"adversarial questions compared: {cats['n']} (of {len(common)} common)")
    print(f"  both correct : {cats['both_ok']}")
    print(f"  FIXED        : {cats['fixed']}")
    print(f"  REGRESSED    : {cats['regressed']}")
    print(f"  both wrong   : {cats['both_wrong']}")
    for kind in ("regressed", "fixed", "both_wrong"):
        if not buckets[kind]:
            continue
        print(f"\n[{kind}]")
        for k, v in buckets[kind].most_common():
            print(f"    {k:<14}{v:>5}")
    for shape, rows in examples.items():
        print(f"\nregression examples ({shape}):")
        for qid, gt, ans in rows:
            print(f"  {qid}\n    gt  = {gt[:70]!r}\n    pred= {ans!r}")


if __name__ == "__main__":
    main()
