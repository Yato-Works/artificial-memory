"""Anatomise LongMemEval failures by *mechanism*, not just by category.

The failure-targeted loop needs to know, for every wrong question, which
mechanism could possibly recover it:

    forced_refusal_temporal   the temporal branch answered without the LLM
                              ("The information provided is not enough to
                              answer this question.") - a resolver verdict
                              that can be overridden instead of trusted
    forced_refusal_pcc        the MSC compiler flagged abstention and the
                              adapter answered without the LLM
    model_refused             the reader itself said "I don't know" although
                              the evidence was in the context (oracle=OK):
                              a prompt-structure hole, fixable per question type
    model_answered_wrong      the reader answered but with the wrong value:
                              reader capability / composition
    oracle_miss               the evidence never reached the context:
                              retrieval work

Because LongMemEval guarantees that every non-abstention question has its answer
somewhere in the haystack, every non-abstention refusal is by definition a
recoverable failure - that is why refusals are counted separately.

Usage:
  uv run python scripts/lme_failure_anatomy.py benchmark_results/longmemeval/grand_longmemeval_report_qwen7b.json
  uv run python scripts/lme_failure_anatomy.py <report> --samples 12
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

T_FORCED = "not enough to answer this question"
P_FORCED = "you did not mention this information"
REFUSAL = (
    "i don't know", "i dont know", "not enough", "no information", "not mentioned",
    "unknown", "unclear", "cannot be determined", "cannot determine", "unable to",
)


def classify(row: dict, forced_abstention_types: set[str]) -> str:
    if not row.get("oracle_recall"):
        return "oracle_miss"
    ans = str(row.get("predicted_answer", "")).lower()
    if T_FORCED in ans:
        return "forced_refusal_temporal"
    if P_FORCED in ans:
        return "forced_refusal_pcc"
    if any(m in ans for m in REFUSAL):
        return "model_refused"
    return "model_answered_wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default="benchmark_results/longmemeval/grand_longmemeval_report_qwen7b.json")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["results"]
    n = len(rows)
    wrong = [r for r in rows if not r["is_correct"]]

    forced_types = {r["question_type"] for r in rows if str(r.get("question_id", "")).endswith("_abs")}
    shapes = collections.Counter()
    by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in wrong:
        k = classify(r, forced_types)
        shapes[k] += 1
        by_type[r["question_type"]][k] += 1

    print(f"report: {args.report}")
    print(f"  accuracy      : {d.get('overall_accuracy', 0) * 100:.1f}%  ({n - len(wrong)}/{n})")
    print(f"  oracle recall : {d.get('oracle_recall', 0) * 100:.1f}%")
    print(f"  wrong         : {len(wrong)}\n")
    print("recovery mechanism")
    print("-" * 62)
    for k, v in shapes.most_common():
        print(f"  {k:<28}{v:>5}   ({v / n * 100:+.1f}pp if fully recovered)")
    print()
    print("by question type")
    print("-" * 62)
    header = f"  {'type':<28}" + "".join(f"{k[:11]:>12}" for k, _ in shapes.most_common())
    print(header)
    for t, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        line = f"  {t:<28}" + "".join(f"{c.get(k, 0):>12}" for k, _ in shapes.most_common())
        print(line)

    for target in ("forced_refusal_temporal", "forced_refusal_pcc", "model_refused"):
        samples = [r for r in wrong if classify(r, forced_types) == target][: args.samples]
        if not samples:
            continue
        print(f"\n--- {target} samples ---")
        for r in samples:
            print(f"  [{r['question_type']}] {r['question_id']}")
            print(f"      Q   = {str(r.get('question', ''))[:100]!r}")
            print(f"      GT  = {str(r['ground_truth'])[:80]!r}")
            print(f"      PRED= {str(r['predicted_answer'])[:80]!r}")


if __name__ == "__main__":
    main()
