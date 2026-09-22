"""Inspect residual oracle misses: cross-reference with raw dataset evidence/GT."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path("benchmark_results/locomo10_runs/selection_v3")
raw = json.load(open("datasets/external/locomo10.json", encoding="utf-8"))

# Build question_id -> (evidence, ground_truth, category) from the raw dataset.
meta = {}
for sample in raw:
    for key, conv in sample.get("conversation", {}).items() if isinstance(sample.get("conversation"), dict) else []:
        pass
for sample in raw:
    sid = sample.get("sample_id")
    qs = sample.get("qa", [])
    conv = sample.get("conversation", {})
    if isinstance(conv, dict):
        conv = {k: v for k, v in conv.items()}
        n_sess = sum(1 for k in conv if k.startswith("session_"))
    for i, q in enumerate(qs):
        meta[f"{sid}-qa-{i:03d}"] = {
            "evidence": q.get("evidence", []),
            "gt": q.get("answer", q.get("adversarial_answer", "")),
            "category": q.get("category"),
        }

pattern = Counter()
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
samples = {}
for conv in range(10):
    p = RUN / f"conv_{conv}_results.json"
    if not p.exists():
        continue
    d = json.load(open(p, encoding="utf-8"))
    for r in d["results"]:
        if r["oracle_recall"]:
            continue
        m = meta.get(r["question_id"], {})
        ev = m.get("evidence") or []
        cat = CAT.get(r["category"], str(r["category"]))
        if not ev:
            key = f"{cat}:NO_EVIDENCE_FIELD"
        else:
            key = f"{cat}:{len(ev)}_ids"
        pattern[key] += 1
        samples.setdefault(key, (r["question_id"], ev, m.get("gt")))

print(f"{'pattern':<34}{'n':>6}   example")
for k, c in pattern.most_common():
    ex = samples[k]
    print(f"{k:<34}{c:>6}   {ex[0]} ev={ex[1]} gt={str(ex[2])[:40]!r}")
