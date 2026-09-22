"""Confirm the adversarial (LoCoMo category 5) ground-truth protocol.

Prints, for a few adversarial questions, the question text, the evidence turn
content, the trap answer, and whether the question subject appears in the
evidence turn.  This settles whether "refuse" or "answer" is the correct policy.
"""
from __future__ import annotations

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))

want = ["conv-26-qa-153", "conv-26-qa-089"]


def qid(sid: str, idx: int) -> str:
    return f"{sid}-qa-{idx:03d}"


for sample in raw:
    sid = sample["sample_id"]
    conv = sample["conversation"]
    # Build dia_id -> (speaker, text)
    turns: dict[str, tuple[str, str]] = {}
    for key, value in conv.items():
        if not key.startswith("session_"):
            continue
        for t in value:
            if isinstance(t, dict) and "dia_id" in t:
                turns[t["dia_id"]] = (t.get("speaker", "?"), t.get("text", ""))
    for i, q in enumerate(sample.get("qa", [])):
        if q.get("category") != 5:
            continue
        if qid(sid, i) not in want:
            continue
        print("=" * 90)
        print(f"{qid(sid, i)}  question = {q.get('question')!r}")
        print(f"  answer field      : {q.get('answer')!r}")
        print(f"  adversarial_answer: {q.get('adversarial_answer')!r}")
        ev = q.get("evidence") or []
        for e in ev:
            spk, txt = turns.get(e, ("?", "<missing>"))
            print(f"  evidence {e}: speaker={spk!r}")
            print(f"      text={txt[:220]!r}")
        # Does the question's subject token appear in the evidence turn?
        subj = [w for w in re.findall(r"\b[A-Z][a-z]+\b", q.get("question", ""))]
        print(f"  question proper nouns: {subj}")
        for e in ev:
            spk, txt = turns.get(e, ("?", ""))
            present = [s for s in subj if s.lower() in txt.lower()]
            print(f"    -> subject tokens present in {e}: {present}")
