"""Inspect LoCoMo adversarial questions: evidence turn vs adversarial answer vs model answer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/selection_v3_llm")
CONVS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1").split(",")]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 6

raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))
adv: dict[str, str] = {}
for s in raw:
    sid = s.get("sample_id")
    for i, q in enumerate(s.get("qa", [])):
        if q.get("category") == 5:
            adv[f"{sid}-qa-{i:03d}"] = q.get("adversarial_answer", "")

adapter = LoCoMoAdapter()
shown = 0
for conv in CONVS:
    turns, questions, _ = adapter.load_conversation(conv_idx=conv)
    run = json.load(open(RUN / f"conv_{conv}_results.json", encoding="utf-8"))["results"]
    by_id = {r["question_id"]: r for r in run}
    turn_map = {t.dia_id: t for t in turns}
    for q in questions:
        if q.category != 5 or shown >= LIMIT:
            continue
        r = by_id.get(q.question_id)
        if r is None or r["is_correct"]:
            continue
        src = " | ".join(turn_map[e].text for e in q.evidence_ids if e in turn_map)
        print(f"\n=== {q.question_id} ===")
        print(f"  Q            : {q.question}")
        print(f"  adversarial_a: {adv.get(q.question_id, '?')!r}")
        print(f"  evidence turn: {src[:300]}")
        print(f"  model answer : {str(r['predicted_answer'])[:220]}")
        shown += 1
