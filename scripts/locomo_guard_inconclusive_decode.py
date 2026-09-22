"""Decode the ``3_attribution_inconclusive`` bucket of the subject-binding guard.

For every adversarial row that the guard currently lets through, we replay the
guard's own helpers and report *why* ``bound`` became True:

  A  an attribute+answer co-occurrence sentence exists in a subject turn
     (genuinely grounded -> guard is right to stay silent)
  B  the answer stems matched a subject turn, but the subject's turn never
     mentions the question's attribute (generic word overlap -> guard is fooled)
  C  the attribute only ever appears in the *other* speaker's turns
     (misattribution -> a stricter rule could refuse)

The output drives the design of a stricter ownership rule.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path("benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
ver = AnswerVerifier()


def load_rows() -> dict[str, dict]:
    rows = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            rows[r["question_id"]] = r
    return rows


def load_ctx(wanted: set[str]) -> dict[str, dict]:
    out = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["qid"] in wanted:
                out[d["qid"]] = d
    return out


def main() -> None:
    rows = load_rows()
    adv = {q: r for q, r in rows.items() if r["category"] == 5
           and not any(m in str(r["predicted_answer"]).lower() for m in ver._REFUSAL_MARKERS)}
    ctx = load_ctx(set(adv))
    print(f"adversarial asserted rows: {len(adv)} | cached contexts: {len(ctx)}")

    buckets = Counter()
    examples: dict[str, list] = {}
    for qid, row in adv.items():
        cached = ctx.get(qid)
        if not cached:
            buckets["NO_CONTEXT"] += 1
            continue
        context = cached["context"]
        q = cached["question"]
        ans = str(row["predicted_answer"])
        turns = ver._parse_turns(context)
        speakers = {s.lower() for s, _ in turns}
        if len(turns) < 2:
            buckets["TOO_FEW_TURNS"] += 1
            continue

        # subject resolution
        subject = None
        for s in sorted(speakers, key=len, reverse=True):
            if re.search(rf"\b{re.escape(s)}'s\b", q.lower()):
                subject = s
                break
        if subject is None:
            mentioned = [s for s in sorted(speakers, key=len, reverse=True)
                         if re.search(rf"\b{re.escape(s)}\b", q.lower())]
            if len(mentioned) != 1:
                buckets["NO_SUBJECT"] += 1
                continue
            subject = mentioned[0]

        stems = ver._content_stems(ans)
        attr = ver._content_stems(q)
        attr.discard(ver._stem(subject))
        if not stems or not attr:
            buckets["EMPTY_STEMS"] += 1
            continue

        # does the attribute ever appear in the subject's own turns?
        own_texts = [t for s, t in turns if s.lower() == subject]
        other_texts = [t for s, t in turns if s.lower() != subject]
        attr_in_own = any(st in ver._content_stems(t) for st in attr for t in own_texts)
        attr_in_other = any(st in ver._content_stems(t) for st in attr for t in other_texts)
        # attribute + answer co-occurrence in a single sentence
        co_own = False
        for t in own_texts:
            for m in re.finditer(r"[^\n.!?]+", t):
                cs = ver._content_stems(m.group(0))
                if any(st in cs for st in attr) and any(st in cs for st in stems):
                    co_own = True
        if co_own:
            buckets["A_grounded_in_own_turn"] += 1
            key = "A"
        elif not attr_in_own and attr_in_other:
            buckets["B_attr_only_other_speaker"] += 1
            key = "B"
        elif not attr_in_own and not attr_in_other:
            buckets["C_attr_absent"] += 1
            key = "C"
        else:
            buckets["D_attr_in_own_no_cooccurrence"] += 1
            key = "D"
        if len(examples.setdefault(key, [])) < 6:
            examples[key].append((qid, subject, q[:52], ans[:42]))

    total = sum(buckets.values())
    for k, c in buckets.most_common():
        print(f"  {k:<30}{c:>5}  {c / total * 100:>5.1f}%")
    for key in sorted(examples):
        print(f"\n--- {key} examples ---")
        for qid, subj, q, ans in examples[key]:
            print(f"  {qid} subject={subj} q={q!r} ans={ans!r}")


if __name__ == "__main__":
    main()
