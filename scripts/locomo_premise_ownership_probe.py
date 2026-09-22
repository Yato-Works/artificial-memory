"""Offline A/B of a candidate *premise-ownership* refusal rule (no production edits).

LoCoMo adversarial questions presuppose an attribute of the queried subject
("Caroline's hand-painted bowl").  The bait was built by moving that attribute to
the *other* speaker.  A copy-style model answers with content that IS grounded in
Caroline's own turns (so the existing subject-binding guard stays silent), yet the
premise itself is misattributed -> the protocol-correct answer is a refusal.

Candidate rule
--------------
attr_stems = content stems of the question minus the subject's own stem.
premise_turns = turns matching >= THRESH of attr_stems.
If premise_turns is non-empty and NO premise turn is owned by the queried
subject, refuse.

We measure, for each threshold, the adversarial gain (wrong non-refusal rows the
rule would convert into a refusal) and the collateral cost (currently-correct
non-refusal rows it would clobber), per category.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
THRESHOLDS = (0.5, 0.6, 0.75, 0.9)

ver = AnswerVerifier()


def is_refusal(text: str) -> bool:
    tl = (text or "").lower()
    return any(m in tl for m in ver._REFUSAL_MARKERS)


def _subject_of(question: str, speakers: set[str]) -> str | None:
    for s in sorted(speakers, key=len, reverse=True):
        if ver._subject_bound_patterns(s)[0].search(question.lower()):
            return s
    return None


def premise_verdict(question: str, answer: str, context: str, thresh: float) -> bool:
    """True when the question premise is owned exclusively by another speaker."""
    turns = ver._parse_turns(context)
    if len(turns) < 2:
        return False
    speakers = {s.lower() for s, _ in turns}
    subject = _subject_of(question, speakers)
    if subject is None:
        return False
    ql = question.lower()
    if not any(p.search(ql) for p in ver._subject_bound_patterns(subject)):
        return False
    attr_stems = ver._content_stems(question)
    attr_stems.discard(ver._stem(subject))
    if len(attr_stems) < 2:
        return False
    premise, own_premise = 0, 0
    for speaker, text in turns:
        t_stems = ver._content_stems(text)
        ratio = sum(1 for st in attr_stems if st in t_stems) / len(attr_stems)
        if ratio >= thresh:
            premise += 1
            if speaker.lower() == subject:
                own_premise += 1
    return premise > 0 and own_premise == 0


def main() -> None:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    rows = []
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            c = cache.get(r["question_id"])
            if c is None:
                continue
            rows.append((CAT.get(r["category"], str(r["category"])), r, c))
    print(f"rows joined: {len(rows)}")
    print(f"(guarded rows already refused by the production guard: "
          f"{sum(1 for cat, r, c in rows if ver.check_subject_binding(c['question'], r['predicted_answer'] or '', c['context']) is not None)})")

    for thresh in THRESHOLDS:
        gain = defaultdict(int)
        cost = defaultdict(int)
        for cat, r, c in rows:
            ans = r["predicted_answer"] or ""
            if is_refusal(ans):
                continue
            if ver.check_subject_binding(c["question"], ans, c["context"]) is not None:
                continue  # already refused by the production guard
            if not premise_verdict(c["question"], ans, c["context"], thresh):
                continue
            if r["is_correct"]:
                cost[cat] += 1
            else:
                gain[cat] += 1
        n_tot = len(rows)
        g = sum(gain.values())
        c_ = sum(cost.values())
        print(f"\n=== threshold {thresh} ===")
        print(f"  gain (wrong -> refusal): {g:>4}   cost (correct -> refusal): {c_:>4}")
        for cat in sorted(set(gain) | set(cost), key=lambda k: -gain[k]):
            print(f"    {cat:<14} gain={gain[cat]:>4} cost={cost[cat]:>4}")


if __name__ == "__main__":
    main()
