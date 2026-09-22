"""Dump regression / fix examples between a frozen baseline and an experiment run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10")
EXP = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10_runs/subject_binding_v1")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
WANT = sys.argv[3] if len(sys.argv) > 3 else "temporal"
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 12


def _question_text_map() -> dict[str, str]:
    """question_id -> question text, from the frozen dataset (for taxonomy)."""
    dataset = Path("datasets/external/locomo10.json")
    if not dataset.exists():
        return {}
    with open(dataset, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, str] = {}
    for conv_idx, conv in enumerate(raw):
        sample_id = conv.get("sample_id", f"conv-{conv_idx}")
        for i, qa in enumerate(conv.get("qa", [])):
            out[f"{sample_id}-qa-{i:03d}"] = qa.get("question", "")
    return out


QTEXT = _question_text_map()


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            r = dict(r)
            r.setdefault("question", QTEXT.get(r["question_id"], ""))
            out[r["question_id"]] = r
    return out


base, exp = load(BASE), load(EXP)
regs, fixes = [], []
for qid in sorted(set(base) & set(exp)):
    b, e = base[qid], exp[qid]
    if CAT.get(b["category"], str(b["category"])) != WANT:
        continue
    if b["is_correct"] and not e["is_correct"]:
        regs.append((qid, b, e))
    elif e["is_correct"] and not b["is_correct"]:
        fixes.append((qid, b, e))

print(f"{WANT}: {len(regs)} regressions, {len(fixes)} fixes\n")

print("=" * 100)
print("REGRESSIONS")
print("=" * 100)
for qid, b, e in regs[:LIMIT]:
    print(f"\n{qid}  | Q: {b['question']}")
    print(f"  GT        : {str(b['ground_truth'])[:90]!r}")
    print(f"  base pred : {str(b['predicted_answer'])[:90]!r}")
    print(f"  new  pred : {str(e['predicted_answer'])[:90]!r}")
    print(f"  oracle base={b['oracle_recall']} new={e['oracle_recall']} "
          f"| tok base={b['tokens_used']} new={e['tokens_used']}")

print("\n" + "=" * 100)
print("FIXES")
print("=" * 100)
for qid, b, e in fixes[:LIMIT]:
    print(f"\n{qid}  | Q: {b['question']}")
    print(f"  GT        : {str(b['ground_truth'])[:90]!r}")
    print(f"  base pred : {str(b['predicted_answer'])[:90]!r}")
    print(f"  new  pred : {str(e['predicted_answer'])[:90]!r}")
