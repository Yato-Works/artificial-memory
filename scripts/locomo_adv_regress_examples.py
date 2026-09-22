"""Show concrete adversarial regression examples with their contexts.

For each adversarial question that passed before and fails now, print the
question, the new answer, the matched context turns, and whether the current
guard fires.  Used to design (not guess) the next guard revision.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path("benchmark_results/locomo10_runs/subject_binding_v1")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def load(root: Path) -> dict:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


base, exp = load(Path("benchmark_results/locomo10")), load(RUN)
ctx_by_qid = {}
qtext = {}
for line in open("benchmark_results/locomo_context_cache.jsonl", encoding="utf-8"):
    d = json.loads(line)
    ctx_by_qid[d["qid"]] = d["context"]
    qtext[d["qid"]] = d["question"]

adv = [q for q, r in base.items() if r["category"] == 5]
reg = [q for q in adv if base[q]["is_correct"] and not exp[q]["is_correct"]]
print(f"adversarial n={len(adv)} regressions={len(reg)}\n")

verifier = AnswerVerifier(subject_binding=True)
shown = 0
for qid in sorted(reg):
    ctx = ctx_by_qid.get(qid, "")
    ans = exp[qid]["predicted_answer"]
    q = qtext.get(qid, "")
    fired = verifier.check_subject_binding(q, ans, ctx)
    print("=" * 92)
    print(f"{qid}  guard_fires={fired is not None}")
    print(f"  Q  : {q}")
    print(f"  NEW: {ans[:120]!r}")
    print(f"  OLD: {base[qid]['predicted_answer'][:90]!r}")
    # Show the context turns that share the most content with the answer
    turns = []
    for line in ctx.splitlines():
        clean = re.sub(r"^\[[^\]]*\]\s*", "", line)
        m = re.match(r"\(In reply to [^)]*\)\s*(.*)", clean)
        if m:
            clean = m.group(1)
        turns.append(clean)
    ans_words = {w for w in re.findall(r"\b[a-z']+\b", ans.lower()) if len(w) > 3}
    scored = sorted(
        ((len(ans_words & set(re.findall(r"\b[a-z']+\b", t.lower()))) / max(1, len(ans_words)), t)
         for t in turns), key=lambda x: -x[0])[:2]
    for ratio, t in scored:
        speaker = t.split(":")[0][:24]
        print(f"  ctx[{ratio:.2f}] {speaker}: {t[len(speaker) + 1:][:110]!r}")
    shown += 1
    if shown >= LIMIT:
        break
