"""Offline attribution of why the subject-binding guard misses adversarial bait.

For each adversarial row whose stored answer asserts content (i.e. is scored
wrong), we replay the production guard helpers stage by stage and count where
the guard bails out.  Also measures the collateral cost on rows that are
currently correct in other categories.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

ver = AnswerVerifier()


def is_refusal(text: str) -> bool:
    tl = (text or "").lower()
    return any(m in tl for m in ver._REFUSAL_MARKERS)


def audit(question: str, answer: str, context: str) -> dict:
    """Replay the guard stages and report the first bail-out point."""
    out = {"guarded": False, "stage": None, "matched": 0, "own": 0,
           "subject": None, "attr_in_context": None}
    if ver.check_subject_binding(question, answer, context) is not None:
        out["guarded"] = True
        out["stage"] = "REFUSE"
        return out

    turns = ver._parse_turns(context)
    if len(turns) < 2:
        out["stage"] = "0_too_few_turns"
        return out
    speakers = {s.lower() for s, _ in turns}
    ans_lower = answer.lower().strip()
    if any(m in ans_lower for m in ver._REFUSAL_MARKERS):
        out["stage"] = "0_already_refusal"
        return out

    context_lower = context.lower()
    for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", question)[1:]:
        low = tok.lower()
        if low in speakers or low in ver._STOP_WORDS:
            continue
        if low not in context_lower:
            out["stage"] = "0_entity_absent"
            return out

    subject = None
    for s in sorted(speakers, key=len, reverse=True):
        if re.search(rf"\b{re.escape(s)}'s\b", question.lower()):
            subject = s
            break
    if subject is None:
        mentioned = [s for s in sorted(speakers, key=len, reverse=True)
                     if re.search(rf"\b{re.escape(s)}\b", question.lower())]
        if len(mentioned) >= 2:
            out["stage"] = "1_two_speakers"
            return out
        subject = mentioned[0] if mentioned else None
    out["subject"] = subject
    if subject is None:
        out["stage"] = "1_no_subject"
        return out

    ql = question.lower()
    if not any(p.search(ql) for p in ver._subject_bound_patterns(subject)):
        out["stage"] = "1b_not_subject_bound"
        return out
    if ver._DATE_LIKE.search(answer) or ver._TEMPORAL_QUESTION.search(question):
        out["stage"] = "1c_temporal"
        return out

    stems = ver._content_stems(answer)
    if not stems:
        out["stage"] = "2_no_stems"
        return out
    attr_stems = ver._content_stems(question)
    attr_stems.discard(ver._stem(subject))
    out["attr_in_context"] = any(
        st in ver._content_stems(text) for st in attr_stems for _, text in turns
    )
    scored = []
    for speaker, text in turns:
        t_stems = ver._content_stems(text)
        scored.append((sum(1 for st in stems if st in t_stems) / len(stems), speaker))
    scored.sort(key=lambda x: -x[0])
    matched = [s for s in scored if s[0] >= 0.5]
    out["matched"] = len(matched)
    if not matched:
        out["stage"] = "2_no_matched_turn"
        return out
    out["own"] = sum(1 for _, speaker in matched if speaker.lower() == subject)
    out["stage"] = "3_attribution_inconclusive"
    return out


def main() -> None:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    stages = defaultdict(int)
    gate_loss = defaultdict(int)
    samples = defaultdict(list)
    n = 0
    for p in sorted(RUN.glob("conv_*_results.json")):
        d = json.load(open(p, encoding="utf-8"))
        for r in d["results"]:
            c = cache.get(r["question_id"])
            if c is None:
                continue
            n += 1
            cat = CAT.get(r["category"], str(r["category"]))
            ans = r["predicted_answer"] or ""
            info = audit(c["question"], ans, c["context"])
            fires = info["guarded"]
            if cat == "adversarial":
                if not r["is_correct"] and not is_refusal(ans):
                    stages[info["stage"]] += 1
                    if len(samples[info["stage"]]) < 3:
                        samples[info["stage"]].append(
                            (r["question_id"], info.get("subject"), info.get("matched"),
                             ans[:50], c["question"][:52]))
            elif r["is_correct"] and not is_refusal(ans) and fires:
                gate_loss[cat] += 1
                if len(samples["LOSS_" + cat]) < 3:
                    samples["LOSS_" + cat].append((r["question_id"], ans[:55]))

    print(f"rows joined: {n}")
    print("\nWHY THE GUARD DOES NOT FIRE on adversarial bait answers:")
    total = sum(stages.values())
    for k, v in sorted(stages.items(), key=lambda x: -x[1]):
        print(f"  {k:<26}{v:>5}  {v / total * 100:>5.1f}%")
        for s in samples[k]:
            print(f"       {s[0]} subject={s[1]} matched={s[2]} ans={s[3]!r}\n"
                  f"         q={s[4]!r}")
    print(f"\n  asserted-and-wrong adversarial rows total: {total}")

    print("\nCOLLATERAL: currently-correct rows the guard would clobber")
    for k, v in sorted(gate_loss.items(), key=lambda x: -x[1]):
        print(f"  {k:<14}{v}")
        for s in samples.get("LOSS_" + k, []):
            print(f"       {s[0]} ans={s[1]!r}")


if __name__ == "__main__":
    main()

