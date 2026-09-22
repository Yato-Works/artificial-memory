"""Retrieval-only sweep of the LoCoMo selection window / rescue token budget.

The Phase-1 fix that recovers ground-truth turns ranked 7-19 costs tokens
(155 -> ~1,500 per question) because the selection loop now has to inspect
every promoted rescue unit.  This script measures the oracle-recall / token
trade-off of ``selection_window_cap`` and ``rescue_token_bonus_per_unit``
without any LLM calls, so the window can be frozen on evidence.

Usage:
  python scripts/locomo_window_sweep.py --convs 0,3,5
  python scripts/locomo_window_sweep.py --convs 0 --caps 8,12,16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def evaluate(conv_idx: int, adapter: LoCoMoAdapter, cap: int, bonus: int) -> dict:
    adapter.compiler.selection_window_cap = cap
    adapter.compiler.rescue_token_bonus_per_unit = bonus
    _, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
    weights = EvidenceScoreWeights()

    per_cat: dict[str, list[int]] = {}
    hit = tokens = 0
    for q in questions:
        pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
        ok = True
        if q.evidence_ids:
            ok = any(ev in pcc.context_text for ev in q.evidence_ids)
        cat = CAT.get(q.category, str(q.category))
        row = per_cat.setdefault(cat, [0, 0])
        row[0] += int(ok)
        row[1] += 1
        hit += int(ok)
        tokens += pcc.token_cost
    return {"n": len(questions), "hit": hit, "tokens": tokens,
            "per_cat": {k: tuple(v) for k, v in per_cat.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="0,3,5")
    ap.add_argument("--caps", default="8,12,16,24")
    ap.add_argument("--bonus", type=int, default=130)
    ap.add_argument("--json", default="benchmark_results/locomo_window_sweep.json")
    args = ap.parse_args()

    convs = [int(c) for c in args.convs.split(",") if c.strip()]
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    adapter = LoCoMoAdapter()

    results = []
    base_cap = adapter.compiler.selection_window_cap
    for cap in caps:
        t0 = time.perf_counter()
        agg = {"n": 0, "hit": 0, "tokens": 0}
        per_cat: dict[str, list[int]] = {}
        for conv in convs:
            r = evaluate(conv, adapter, cap, args.bonus)
            agg["n"] += r["n"]
            agg["hit"] += r["hit"]
            agg["tokens"] += r["tokens"]
            for k, (h, n) in r["per_cat"].items():
                row = per_cat.setdefault(k, [0, 0])
                row[0] += h
                row[1] += n
        dt = time.perf_counter() - t0
        row = {
            "cap": cap, "bonus": args.bonus, "n": agg["n"],
            "oracle": agg["hit"] / agg["n"], "tokens": agg["tokens"] / agg["n"],
            "per_cat": {k: v for k, v in per_cat.items()},
        }
        results.append(row)
        print(f"cap={cap:>3} bonus={args.bonus:>4}  oracle={row['oracle'] * 100:>5.2f}%  "
              f"tokens/Q={row['tokens']:>7.1f}  ({dt:.0f}s)", flush=True)

    adapter.compiler.selection_window_cap = base_cap

    print("\n| cap | oracle | tokens/Q | " + " | ".join(
        k for k in ("single-hop", "adversarial", "temporal", "multi-hop", "open-domain")) + " |")
    print("|---|---:|---:|" + "---:|" * 5)
    for r in results:
        cells = []
        for k in ("single-hop", "adversarial", "temporal", "multi-hop", "open-domain"):
            h, n = r["per_cat"].get(k, (0, 0))
            cells.append(f"{h / n * 100:.1f}%" if n else "-")
        print(f"| {r['cap']} | {r['oracle'] * 100:.2f}% | {r['tokens']:.0f} | "
              + " | ".join(cells) + " |")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
