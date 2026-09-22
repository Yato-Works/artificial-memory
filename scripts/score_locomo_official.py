"""Score a stored LoCoMo run with the OFFICIAL LoCoMo metric (token-F1).

Official protocol (``third_party/benchmarks/locomo/task_eval/evaluation.py``)::

    category 1 (multi-hop)   -> f1(pred, gt)        comma-split multi-answer F1
    category 2,3,4           -> f1_score(pred, gt)  Porter-stemmed token F1
    category 5 (adversarial) -> 1 if "no information available" / "not mentioned"
                                appears in the prediction, else 0

Aggregation (``evaluation_stats.analyze_aggr_acc``) is a micro mean over every
question, which is what published LoCoMo numbers (Mem0, LiveMem, past.dev, ...)
report.  This is a *different* protocol from AM's deterministic category
matchers used during development, so a run should always be published with
both numbers side by side.

The official module imports ``bert_score`` (and therefore torch) at import
time; that dependency is stubbed because the deterministic F1 path does not
use it.

Requires ``nltk`` for the Porter stemmer, so run it with a benchmark venv::

    .venv-benchmarks/longmemeval/Scripts/python.exe scripts/score_locomo_official.py <run_dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_EVAL = REPO / "third_party" / "benchmarks" / "locomo" / "task_eval" / "evaluation.py"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load_official_module() -> types.ModuleType:
    """Import the pinned official scorer, stubbing the unused bert_score import."""
    stub = types.ModuleType("bert_score")

    def _unused(*_args, **_kwargs):  # pragma: no cover - never called
        raise NotImplementedError("bert_score is not used by the official F1 path")

    stub.score = _unused
    sys.modules.setdefault("bert_score", stub)
    spec = importlib.util.spec_from_file_location("locomo_official_eval", OFFICIAL_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official scorer: {OFFICIAL_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question_score(official: types.ModuleType, category: int, prediction: str, answer: str) -> float:
    """Official per-question score for one LoCoMo QA pair."""
    pred = "" if prediction is None else str(prediction)
    gt = "" if answer is None else str(answer)
    if category == 1:
        return float(official.f1(pred, gt))
    if category in (2, 3, 4):
        # The official harness strips sub-answers for open-domain (cat 3).
        if category == 3:
            gt = gt.split(";")[0].strip()
        return float(official.f1_score(pred, gt))
    if category == 5:
        low = pred.lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0
    raise ValueError(f"unknown LoCoMo category: {category}")


def iter_questions(root: Path):
    for path in sorted(root.glob("conv_*_results.json")):
        payload = json.load(open(path, encoding="utf-8"))
        for record in payload["results"]:
            yield path.name, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="directory holding conv_*_results.json")
    parser.add_argument("--json", default=None, help="optional path for a JSON report")
    parser.add_argument("--tag", default=None, help="label written into the JSON report")
    parser.add_argument("--convs", default=None,
                        help="Comma-separated conversation indices to restrict the score "
                             "(for apples-to-apples comparison with a partial A/B run)")
    args = parser.parse_args()

    root = Path(args.run_dir)
    official = load_official_module()
    keep = None if args.convs is None else {
        f"conv_{c.strip()}_results.json" for c in args.convs.split(",") if c.strip() != ""
    }

    per_cat: dict[int, list[float]] = defaultdict(list)
    per_conv: dict[str, list[float]] = defaultdict(list)
    frozen_binary: dict[int, list[float]] = defaultdict(list)
    rows: list[dict] = []

    for conv, record in iter_questions(root):
        if keep is not None and conv not in keep:
            continue
        category = int(record["category"])
        score = question_score(official, category, record["predicted_answer"],
                               record["ground_truth"])
        per_cat[category].append(score)
        per_conv[conv].append(score)
        frozen_binary[category].append(1.0 if record.get("is_correct") else 0.0)
        rows.append({"question_id": record["question_id"], "category": category,
                     "official_f1": round(score, 4),
                     "dev_matcher_correct": bool(record.get("is_correct"))})

    total = sum(len(v) for v in per_cat.values())
    if total == 0:
        raise SystemExit(f"no conv_*_results.json found under {root}")

    micro = sum(sum(v) for v in per_cat.values()) / total
    macro = sum(sum(v) / len(v) for v in per_cat.values()) / len(per_cat)
    print(f"run: {root}")
    print(f"official LoCoMo protocol (F1, micro) : {micro * 100:.2f}%  ({total} questions)")
    print(f"official LoCoMo protocol (F1, macro) : {macro * 100:.2f}%")
    print(f"{'category':<14}{'n':>6}{'official F1':>13}{'dev matcher':>13}")
    for category in sorted(per_cat, key=lambda c: -len(per_cat[c])):
        n = len(per_cat[category])
        print(f"{CAT.get(category, str(category)):<14}{n:>6}"
              f"{sum(per_cat[category]) / n * 100:>12.1f}%"
              f"{sum(frozen_binary[category]) / n * 100:>12.1f}%")

    if args.json:
        report = {
            "run_dir": str(root),
            "tag": args.tag or root.name,
            "protocol": "official_locomo_f1_micro",
            "overall_micro_pct": round(micro * 100, 2),
            "overall_macro_pct": round(macro * 100, 2),
            "n_questions": total,
            "per_category": {
                CAT.get(c, str(c)): {
                    "n": len(per_cat[c]),
                    "official_f1_pct": round(sum(per_cat[c]) / len(per_cat[c]) * 100, 2),
                    "dev_matcher_pct": round(sum(frozen_binary[c]) / len(frozen_binary[c]) * 100, 2),
                }
                for c in sorted(per_cat)
            },
            "per_conversation": {
                conv: round(sum(v) / len(v) * 100, 2) for conv, v in sorted(per_conv.items())
            },
            "questions": rows,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
