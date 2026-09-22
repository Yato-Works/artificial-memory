"""List adversarial answers the verifier treats as refusals but the scorer rejects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
BASE = Path("benchmark_results/locomo10")
SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 25


def load(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


run, base = load(RUN), load(BASE)
v = AnswerVerifier()
rows = []
for qid, r in run.items():
    if r["category"] != 5 or r["is_correct"]:
        continue
    if qid in base and base[qid]["is_correct"]:
        continue
    ans = str(r["predicted_answer"])
    marker = next((m for m in v._REFUSAL_MARKERS if m in ans.lower()), None)
    if marker:
        rows.append((qid, marker, ans[:80]))

print(f"adversarial regressions whose answer carries a verifier refusal marker: {len(rows)}")
for qid, marker, ans in rows[:SHOW]:
    print(f"  {qid}  marker={marker!r:20} ans={ans!r}")
