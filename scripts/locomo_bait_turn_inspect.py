"""Look at adversarial bait answers and see which turn they were copied from."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ver = AnswerVerifier()

cache = {}
for line in Path("benchmark_results/locomo_context_cache.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        cache[row["qid"]] = row

rows = []
for p in sorted(RUN.glob("conv_*_results.json")):
    for r in json.load(open(p, encoding="utf-8"))["results"]:
        if r["category"] == 5 and not r["is_correct"]:
            rows.append(r)

print(f"adversarial wrong rows: {len(rows)}\n")


def turns_of(ctx: str, q: str) -> list[tuple[str, str]]:
    out = []
    for line in ctx.split("\n"):
        m = re.match(r"\s*\[([^\]]+)\]\s*(.*)", line, re.DOTALL)
        if not m:
            continue
        header, body = m.group(1), m.group(2)
        sp = "?"
        mm = re.search(r"\b([A-Z][a-z]{2,})\s*:", body)
        if mm:
            sp = mm.group(1).lower()
        out.append((sp, body))
    return out


shown = 0
for r in rows:
    if shown >= LIMIT:
        break
    c = cache.get(r["question_id"])
    if not c:
        continue
    q = c["question"]
    ans = (r["predicted_answer"] or "").strip()
    aw = {w for w in re.findall(r"\b[a-z]{4,}\b", ans.lower())}
    if not aw:
        continue
    turns = turns_of(c["context"], q)
    ranked = sorted(turns, key=lambda t: -len(aw & set(re.findall(r"\b[a-z]{4,}\b", t[1].lower()))))
    best = ranked[0] if ranked else ("?", "")
    ov = len(aw & set(re.findall(r"\b[a-z]{4,}\b", best[1].lower())))
    print("=" * 100)
    print(f"Q ({r['question_id']}): {q}")
    print(f"  ANSWER: {ans[:140]!r}   (answer words={len(aw)})")
    print(f"  BEST TURN [speaker={best[0]}, overlap={ov}]: {best[1][:220]!r}")
    shown += 1
