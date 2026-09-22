"""Does the integrity gate flag adversarial (no-answer) questions? Probe only."""
import sys
from collections import Counter

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

adapter = LoCoMoAdapter()
stats = Counter()
adv_records: list[dict] = []
notes = []
for conv in (0, 1, 2):
    _, questions, ir_records = adapter.load_conversation(conv_idx=conv)
    weights = EvidenceScoreWeights()
    for q in questions:
        if q.category not in (5, 2):
            continue
        pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
        integrity = getattr(pcc, "certificate", None)
        gate = getattr(adapter.compiler.integrity_gate, "last_result", None)
        flagged = "Proposition Integrity Warning" in pcc.context_text
        stats[("cat", q.category, "flagged" if flagged else "clean")] += 1
        if q.category == 5:
            adv_records.append({
                "question_id": q.question_id,
                "flagged": flagged,
                "question": q.question,
            })
        if q.category == 5 and not flagged and len(notes) < 6:
            notes.append((q.question_id, q.question))

print("Integrity-gate flag rates (context contains 'Proposition Integrity Warning'):")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")

print("\nUnflagged adversarial samples:")
for qid, q in notes:
    print(f"  {qid}: {q}")

# Cross-tab the gate flag against baseline / candidate-run correctness.
import json
from pathlib import Path

for run in ("benchmark_results/locomo10", "benchmark_results/locomo10_runs/selection_v3_llm"):
    root = Path(run)
    hit = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            hit[r["question_id"]] = r
    tab = Counter()
    for rec in adv_records:
        r = hit.get(rec["question_id"])
        if not r:
            continue
        tab[(rec["flagged"], bool(r["is_correct"]))] += 1
    print(f"\n{run}: flagged/clean x correct")
    for k in sorted(tab):
        print(f"  flagged={k[0]} correct={k[1]}: {tab[k]}")

out = Path("benchmark_results/probe_adversarial_flags.json")
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(adv_records, open(out, "w", encoding="utf-8"), indent=2)
print(f"\nSaved: {out}")
