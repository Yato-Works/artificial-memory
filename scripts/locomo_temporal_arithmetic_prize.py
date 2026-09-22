"""Sizing the deterministic calendar-arithmetic prize for LoCoMo temporal (cat-2).

Among temporal questions that are scored wrong while BOTH the prediction and the
ground truth carry absolute dates, two very different causes exist:

  A. ANCHOR_MATCH + DERIVATION_WRONG - the prediction cites the SAME anchor date
     as the ground truth but derives the weekday / week offset incorrectly
     ("The sunday before 25 May 2023" -> "Saturday before 25 May 2023").
     A deterministic calendar engine can repair this from the anchor alone.
  B. ANCHOR_DIFFERENT - the prediction cites a different date entirely, i.e. the
     wrong memory was used.  Calendar arithmetic cannot help.

Usage: python scripts/locomo_temporal_arithmetic_prize.py <run_dir>
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
WEEKDAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}

DAY_RE = re.compile(r"\b(\d{1,2})\s*(" + "|".join(MONTHS) + r")\b(?:\s*,?\s*(\d{4}))?", re.I)
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b(?:\s*,?\s*(\d{4}))?", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
WD_RE = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.I)


def anchors(text: str) -> set[str]:
    """Full date anchors (day+month[+year]) expressed as canonical keys."""
    out = set()
    for d, m, y in DAY_RE.findall(text):
        out.add(f"{int(d):02d}-{MONTHS[m.lower()]:02d}-{y or '????'}")
    return out


def month_anchors(text: str) -> set[str]:
    return {f"{MONTHS[m.lower()]:02d}-{y or '????'}" for m, y in MONTH_RE.findall(text)}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                "benchmark_results/locomo10_runs/subject_binding_v1")
    buckets = Counter()
    fixes: list[tuple[str, str, str, str]] = []
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            if r["category"] != 2:
                continue
            gt, pred = str(r["ground_truth"]), str(r["predicted_answer"])
            if score(2, gt, pred):
                buckets["correct"] += 1
                continue
            a_gt, a_pr = anchors(gt), anchors(pred)
            if a_gt and a_pr and (a_gt & a_pr):
                buckets["A_anchor_match_derivation_wrong"] += 1
                fixes.append((r["question_id"], gt, pred, ""))
                continue
            if a_gt and not a_pr:
                if month_anchors(gt) & month_anchors(pred):
                    buckets["A2_month_match_day_missing"] += 1
                elif YEAR_RE.search(gt) and YEAR_RE.search(pred):
                    buckets["B_year_only_match"] += 1
                else:
                    buckets["B_anchor_different"] += 1
                continue
            if a_pr and not a_gt:
                buckets["C_gt_relative_only"] += 1
                continue
            buckets["D_no_anchor"] += 1

    total = sum(buckets.values())
    print(f"temporal questions: {total} | dir={root}")
    for k, v in buckets.most_common():
        print(f"  {k:<32}{v:>5}  {v / total * 100:>5.1f}%")

    print("\n--- A bucket: same anchor, arithmetic/derivation wrong ---")
    for qid, gt, pred, _ in fixes[:20]:
        wd_gt, wd_pr = WD_RE.findall(gt), WD_RE.findall(pred)
        tag = "weekday_derivation" if wd_gt or wd_pr else "same_anchor_other"
        print(f"  [{tag}] {qid}\n      gt  ={gt!r}\n      pred={pred!r}")


if __name__ == "__main__":
    main()
