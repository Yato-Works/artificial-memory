"""Dump temporal / adversarial regressions (baseline correct -> new wrong)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
WANT = sys.argv[1] if len(sys.argv) > 1 else "temporal"
BASE = Path("benchmark_results/locomo10")
EXP = Path("benchmark_results/locomo10_runs/selection_v3_llm")


def load(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


base, exp = load(BASE), load(EXP)

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402

adapter = LoCoMoAdapter()
qtext: dict[str, str] = {}
for conv in range(10):
    try:
        _, qs, _ = adapter.load_conversation(conv_idx=conv)
    except Exception:
        break
    for q in qs:
        qtext[q.question_id] = q.question
rows = []
for qid, e in exp.items():
    b = base.get(qid)
    if b is None:
        continue
    if CAT.get(b["category"]) != WANT:
        continue
    if not b["is_correct"]:
        continue
    if e["is_correct"]:
        continue
    rows.append((qid, b, e))

print(f"{WANT}: {len(rows)} regressions (baseline correct -> new wrong)\n")
for qid, b, e in rows[:30]:
    print(f"{qid}  oracle {b['oracle_recall']} -> {e['oracle_recall']}  tok {b['tokens_used']} -> {e['tokens_used']}")
    print(f"  Q  : {qtext.get(qid, '?')[:100]}")
    print(f"  GT : {str(b['ground_truth'])[:100]!r}")
    print(f"  was: {str(b['predicted_answer'])[:100]!r}")
    print(f"  now: {str(e['predicted_answer'])[:100]!r}")
print(f"\n... {max(0, len(rows) - 30)} more")
