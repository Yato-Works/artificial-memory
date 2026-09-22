"""Authoritative LoCoMo comparison: both runs re-scored with the current scorer.

Usage:
  python scripts/locomo_compare_rescored.py <baseline_dir> <experiment_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import CAT, score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            r = dict(r)
            r["ok"] = score(r["category"], r["ground_truth"], r["predicted_answer"])
            out[r["question_id"]] = r
    return out


def main() -> None:
    base_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10")
    exp_dir = Path(sys.argv[2] if len(sys.argv) > 2
                   else "benchmark_results/locomo10_runs/subject_binding_v1")
    base, exp = load(base_dir), load(exp_dir)
    common = sorted(set(base) & set(exp))
    print(f"Baseline {base_dir} ({len(base)} Q) | Experiment {exp_dir} ({len(exp)} Q)")
    print(f"Common questions: {len(common)}  (both sides re-scored with the current harness)")

    agg = defaultdict(lambda: defaultdict(int))
    tok_b = tok_e = 0
    for qid in common:
        b, e = base[qid], exp[qid]
        cat = CAT.get(b["category"], str(b["category"]))
        a = agg[cat]
        a["n"] += 1
        a["b"] += b["ok"]
        a["e"] += e["ok"]
        a["b_ora"] += bool(b["oracle_recall"])
        a["e_ora"] += bool(e["oracle_recall"])
        a["fix"] += bool(e["ok"] and not b["ok"])
        a["reg"] += bool(b["ok"] and not e["ok"])
        tok_b += b["tokens_used"]
        tok_e += e["tokens_used"]

    n = len(common)
    tb, te = (sum(a["b"] for a in agg.values()), sum(a["e"] for a in agg.values()))
    ob, oe = (sum(a["b_ora"] for a in agg.values()), sum(a["e_ora"] for a in agg.values()))
    print(f"\n| category | n | base acc | new acc | delta | base oracle | new oracle | "
          f"delta | fix | reg |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in sorted(agg, key=lambda c: -agg[c]["n"]):
        a = agg[cat]
        c = a["n"]
        print(f"| {cat} | {c} | {a['b'] / c * 100:.1f}% | {a['e'] / c * 100:.1f}% | "
              f"{(a['e'] - a['b']) / c * 100:+.1f}pp | {a['b_ora'] / c * 100:.1f}% | "
              f"{a['e_ora'] / c * 100:.1f}% | {(a['e_ora'] - a['b_ora']) / c * 100:+.1f}pp | "
              f"{a['fix']} | {a['reg']} |")
    print(f"| **ALL** | {n} | {tb / n * 100:.1f}% | {te / n * 100:.1f}% | "
          f"{(te - tb) / n * 100:+.1f}pp | {ob / n * 100:.1f}% | {oe / n * 100:.1f}% | "
          f"{(oe - ob) / n * 100:+.1f}pp | "
          f"{sum(a['fix'] for a in agg.values())} | {sum(a['reg'] for a in agg.values())} |")
    print(f"\ntokens/Q: {tok_b / n:.1f} -> {tok_e / n:.1f} "
          f"(x{tok_e / max(1, tok_b):.1f})")
    print(f"accuracy delta: {(te - tb) / n * 100:+.2f}pp | oracle delta: "
          f"{(oe - ob) / n * 100:+.2f}pp")


if __name__ == "__main__":
    main()
