"""Re-score a stored LoCoMo run with the CURRENT (frozen) category scorers.

The frozen baseline report was produced before the strict calendar scorer
(``_temporal_answer_matches``) existed, so its ``is_correct`` flags are not
comparable with a current run.  Both runs store ``predicted_answer``,
``ground_truth`` and ``category``, so every run can be re-scored offline with
one identical scorer - no LLM calls, no protocol change.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADAPTER = LoCoMoAdapter.__new__(LoCoMoAdapter)  # scorer methods are classmethods


def score(category: int, gt: str, pred: str) -> bool:
    gt_l = str(gt).lower().strip()
    ans_l = str(pred).lower().strip()
    if category == 2:
        return ADAPTER._temporal_answer_matches(gt_l, ans_l)
    if category == 3:
        return ADAPTER._open_domain_answer_matches(gt_l, ans_l)
    if category == 4:
        return ADAPTER._single_hop_answer_matches(gt_l, ans_l)
    if not gt_l:
        # Mirrors the production empty-ground-truth branch (word-bounded refusal
        # detection; the old substring test credited "Fantasy novels").
        return LoCoMoAdapter.is_refusal_shaped(ans_l)
    if gt_l in ans_l or ans_l in gt_l:
        return True
    clean_gt = re.sub(r"\bde-stress\b", "destress", gt_l).replace("-", " ")
    clean_ans = re.sub(r"\bde-stress\b", "destress", ans_l).replace("-", " ")
    for w, n in LoCoMoAdapter._NUMBER_WORDS.items():
        clean_gt = re.sub(rf"\b{w}\b", n, clean_gt)
        clean_ans = re.sub(rf"\b{w}\b", n, clean_ans)
    if clean_gt in clean_ans or clean_ans in clean_gt:
        return True
    gt_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_gt) if len(w) > 2 or w.isdigit()}
    ans_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_ans) if len(w) > 2 or w.isdigit()}
    if gt_words and ans_words:
        stem = lambda w: w[:4]  # noqa: E731 - WideSlicer._stem is instance-bound
        gt_stems = {stem(w) for w in gt_words}
        ans_stems = {stem(w) for w in ans_words}
        overlap = max(len(gt_words & ans_words), len(gt_stems & ans_stems))
        if overlap / len(gt_words) >= 0.33:
            return True
        if len(gt_words) <= 3 and overlap >= 1:
            return True
    return False


def rescore(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            r = dict(r)
            r["rescored_correct"] = score(r["category"], r["ground_truth"], r["predicted_answer"])
            out[r["question_id"]] = r
    return out


def report(name: str, rows: dict[str, dict]) -> None:
    n = len(rows)
    re_ok = sum(1 for r in rows.values() if r["rescored_correct"])
    old_ok = sum(1 for r in rows.values() if r["is_correct"])
    print(f"\n{name}: n={n}")
    print(f"  stored is_correct   : {old_ok / n * 100:6.2f}% ({old_ok})")
    print(f"  re-scored (current) : {re_ok / n * 100:6.2f}% ({re_ok})")
    print(f"  {'category':<14}{'n':>6}{'stored':>9}{'rescored':>10}")
    cats: dict[str, list] = {}
    for r in rows.values():
        cats.setdefault(CAT.get(r["category"], str(r["category"])), []).append(r)
    for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        so = sum(1 for r in v if r["is_correct"])
        ro = sum(1 for r in v if r["rescored_correct"])
        print(f"  {c:<14}{len(v):>6}{so / len(v) * 100:>8.1f}%{ro / len(v) * 100:>9.1f}%")


def main() -> None:
    for arg in sys.argv[1:]:
        root = Path(arg)
        rows = rescore(root)
        report(str(root), rows)
        if len(sys.argv) == 3:
            out = root / "rescored.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({k: {"rescored_correct": v["rescored_correct"],
                               "category": v["category"],
                               "tokens_used": v["tokens_used"]}
                           for k, v in rows.items()}, fh, indent=1)
            print(f"  saved: {out}")


if __name__ == "__main__":
    main()
