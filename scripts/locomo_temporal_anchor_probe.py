"""Date-anchor analysis for LoCoMo temporal questions using the context cache.

For each category-2 (temporal) question we check where the session date that the
ground truth is anchored to appears in the compiled context (which is stored in
selection order).  If the anchor date is nearly always in the *first* unit, the
temporal regressions are pure context dilution and an explicit primary-date
header is the right fix.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo_context_cache.jsonl")
RUN = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo10_runs/subject_binding_v1")

# "[D1:3 on 1:56 pm on 8 May, 2023]" -> session D1, date "8 May, 2023"
HDR = re.compile(r"\[(D\d+):(\d+)\s+on\s+[^,\]]*,\s*([^\]]+?)\]")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")


def gt_anchors(gt: str) -> list[str]:
    """Date-like fragments inside the ground truth, normalised to 'D Month, YYYY'."""
    out: list[str] = []
    for m in re.finditer(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", gt):
        out.append(f"{int(m.group(1))} {m.group(2).lower()}, {m.group(3)}")
    for m in re.finditer(r"([A-Za-z]+)\s+(\d{4})\b", gt):
        out.append(f"{m.group(1).lower()}, {m.group(2)}")
    for m in re.finditer(r"\b(\d{4})\b", gt):
        out.append(m.group(1))
    return out


def match(anchor: str, date_field: str) -> bool:
    d = date_field.lower().strip()
    if anchor.lower() in d:
        return True
    # year-only anchors: require the same year token
    if re.fullmatch(r"\d{4}", anchor):
        return bool(re.search(rf"\b{anchor}\b", d))
    return False


def main() -> None:
    results = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            results[r["question_id"]] = r

    stats = Counter()
    ranks_when_found = Counter()
    dil = {"correct": [], "wrong": []}
    examples = []

    for line in CACHE.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["category"] != 2:
            continue
        row = results.get(rec["qid"])
        if row is None:
            continue
        units = HDR.findall(rec["context"])
        dates = [u[2] for u in units]
        n_distinct = len({d.strip() for d in dates})
        anchors = gt_anchors(str(row["ground_truth"]))
        stats["n"] += 1
        if not anchors:
            stats["no_anchor_in_gt"] += 1
            continue
        if not dates:
            stats["no_dates_in_ctx"] += 1
            continue
        rank = None
        for i, d in enumerate(dates):
            if any(match(a, d) for a in anchors):
                rank = i
                break
        if rank is None:
            stats["anchor_absent"] += 1
        else:
            stats["anchor_present"] += 1
            ranks_when_found["0" if rank == 0 else "1-4" if rank < 5 else "5+"] += 1
        key = "correct" if row["is_correct"] else "wrong"
        dil[key].append(n_distinct)
        if not row["is_correct"] and rank is not None and rank == 0 and len(examples) < 10:
            examples.append((rec["qid"], row["ground_truth"], row["predicted_answer"],
                             n_distinct, dates[:3]))

    print(f"temporal questions with cached context + result: {stats['n']}")
    print(f"  anchor date present in context : {stats['anchor_present']} "
          f"({stats['anchor_present'] / max(1, stats['n']) * 100:.1f}%)")
    print(f"  anchor date absent             : {stats['anchor_absent']}")
    print(f"  GT has no date-like anchor     : {stats['no_anchor_in_gt']}")
    print(f"  context has no dated headers   : {stats['no_dates_in_ctx']}")
    print("\nrank of the unit carrying the GT anchor date (0 = very first unit):")
    for k in ("0", "1-4", "5+"):
        c = ranks_when_found.get(k, 0)
        print(f"  rank {k:<4}{c:>5}  {'#' * int(c / max(1, stats['anchor_present']) * 50)}")

    for k in ("correct", "wrong"):
        v = dil[k]
        if v:
            print(f"\nmean distinct session dates in context ({k}): "
                  f"{sum(v) / len(v):.1f}  (n={len(v)})")

    print("\nwrong answers whose context had the GT anchor in the FIRST unit:")
    for qid, gt, pred, nd, head in examples:
        print(f"  {qid} ndates={nd} gt={str(gt)[:44]!r} pred={str(pred)[:44]!r}")
        print(f"      first dates: {head}")


if __name__ == "__main__":
    main()
