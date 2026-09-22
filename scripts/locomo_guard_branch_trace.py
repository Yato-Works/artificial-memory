"""Trace WHICH early-exit branch lets adversarial bait answers pass the guard.

Uses the cached contexts (no LLM calls) plus the saved answers of a completed
run, and instruments ``AnswerVerifier.check_subject_binding`` with the same
decision points so every unchecked answer is attributed to one branch.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
BASE = Path("benchmark_results/locomo10")
SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def load_cache() -> dict[str, dict]:
    out = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["qid"]] = row
    return out


def load_run(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def main() -> None:
    cache = load_cache()
    run = load_run(RUN)
    base = load_run(BASE)
    verifier = AnswerVerifier()

    reason: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)

    for qid, r in run.items():
        if r["category"] != 5 or not r["is_correct"]:
            continue
        if qid in base and base[qid]["is_correct"]:
            continue  # already correct in baseline: not a regression
        row = cache.get(qid)
        if row is None:
            reason["NO_CACHED_CONTEXT"] += 1
            continue
        ctx, q, ans = row["context"], row["question"], str(r["predicted_answer"])
        verdict = verifier.check_subject_binding(q, ans, ctx)
        if verdict is not None:
            reason["GUARD_FIRES"] += 1
            continue
        # Not fired: attribute to a branch by re-walking the decision points.
        turns = verifier._parse_turns(ctx)
        if len(turns) < 2:
            reason["TOO_FEW_TURNS"] += 1
            continue
        if any(m in ans.lower() for m in verifier._REFUSAL_MARKERS):
            reason["ALREADY_REFUSAL"] += 1
            if len(examples["ALREADY_REFUSAL"]) < SHOW:
                examples["ALREADY_REFUSAL"].append(
                    (qid, ans[:80], row["question"][:60]))
            continue
        speakers = {s.lower() for s, _ in turns}
        ctx_lower = ctx.lower()
        ent = None
        for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", q)[1:]:
            low = tok.lower()
            if low in speakers or low in verifier._STOP_WORDS:
                continue
            if low not in ctx_lower:
                ent = tok
                break
        if ent:
            reason["ENTITY_GUARD_BUG"] += 1
            continue
        subject = None
        for s in sorted(speakers, key=len, reverse=True):
            if re.search(rf"\b{re.escape(s)}'s\b", q.lower()):
                subject = s
                break
        if subject is None:
            mentioned = [s for s in sorted(speakers, key=len, reverse=True)
                         if re.search(rf"\b{re.escape(s)}\b", q.lower())]
            if len(mentioned) >= 2:
                reason["BOTH_SPEAKERS_IN_Q"] += 1
                continue
            subject = mentioned[0] if mentioned else None
        if subject is None:
            reason["NO_SUBJECT_IN_Q"] += 1
            continue
        ql = q.lower()
        if not any(p.search(ql) for p in verifier._subject_bound_patterns(subject)):
            reason["NOT_SUBJECT_BOUND_FORM"] += 1
            if len(examples["NOT_SUBJECT_BOUND_FORM"]) < SHOW:
                examples["NOT_SUBJECT_BOUND_FORM"].append((qid, q, ans[:70]))
            continue
        if verifier._DATE_LIKE.search(ans) or verifier._TEMPORAL_QUESTION.search(q):
            reason["DATE_TEMPORAL_SKIP"] += 1
            continue
        stems = verifier._content_stems(ans)
        if not stems:
            reason["NO_ANSWER_STEMS"] += 1
            continue
        scored = []
        for speaker, text in turns:
            t_stems = verifier._content_stems(text)
            ratio = sum(1 for st in stems if st in t_stems) / len(stems)
            scored.append((ratio, speaker, text.lower()))
        matched = [s for s in scored if s[0] >= 0.5]
        if not matched:
            reason["NO_MATCHED_TURN_0.5"] += 1
            if len(examples["NO_MATCHED_TURN_0.5"]) < SHOW:
                best = max(scored, key=lambda x: x[0])
                examples["NO_MATCHED_TURN_0.5"].append(
                    (qid, q, ans[:60], f"best={best[0]:.2f} speaker={best[1]}"))
            continue
        # Matched turns exist: with the current logic the absence of first-person
        # ownership means no refusal.
        reason["MATCHED_BUT_NOT_MISATTRIBUTED"] += 1
        if len(examples["MATCHED_BUT_NOT_MISATTRIBUTED"]) < SHOW:
            ex = [(round(s[0], 2), s[1], s[2][:110]) for s in matched[:2]]
            examples["MATCHED_BUT_NOT_MISATTRIBUTED"].append((qid, q, ans[:60], ex))

    total = sum(reason.values())
    print(f"run={RUN.name} | unchecked adversarial wrong answers: {total}")
    for k, v in reason.most_common():
        print(f"  {k:<32}{v:>5}  {v / total * 100:>5.1f}%")
    for kind, rows in examples.items():
        print(f"\n--- {kind} ---")
        for row in rows:
            print(f"  {row}")


if __name__ == "__main__":
    main()
