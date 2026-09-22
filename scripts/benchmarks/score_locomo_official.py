"""Score a stored AM LoCoMo run with the PINNED official harness.

Uses the official repository's own metric functions
(``third_party/benchmarks/locomo/task_eval/evaluation.py``): the per-category
rule of ``eval_question_answering`` (stemmed F1 for categories 1-4, literal
"no information available" / "not mentioned" match for category 5) is imported
and executed verbatim, so no AM code decides correctness.

The metric only needs ``regex``, ``nltk`` and ``numpy``; the module-level
``bert_score`` import is stubbed because ``--metric f1`` never calls it.  Run it
with an interpreter that provides nltk (the benchmark venvs do):

  .venv-benchmarks/beam/Scripts/python.exe scripts/benchmarks/score_locomo_official.py \
      --run benchmark_results/locomo10_runs/directives_v4

Output: benchmark_results/official_locomo_score.json  (+ console table)
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

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = REPO / "third_party" / "benchmarks" / "locomo"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load_official_module():
    """Import the pinned official evaluation module (stubbing unused bert_score)."""
    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])  # never reached with metric='f1'
        sys.modules["bert_score"] = stub
    path = OFFICIAL_DIR / "task_eval" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("official_locomo_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    # evaluation.py imports its siblings by top-level name only
    sys.path.insert(0, str(OFFICIAL_DIR))
    spec.loader.exec_module(module)
    return module


def load_run(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(run_dir.glob("conv_*_results.json")):
        for r in json.load(open(path, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="benchmark_results/locomo10_runs/directives_v4")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--out", default="benchmark_results/official_locomo_score.json")
    ap.add_argument("--tag", default=None, help="Label stored in the output JSON")
    args = ap.parse_args()

    official = load_official_module()
    run = load_run(Path(args.run))
    dataset = json.load(open(REPO / args.dataset, encoding="utf-8"))
    print(f"official scorer: {OFFICIAL_DIR/'task_eval'/'evaluation.py'}")
    print(f"run: {args.run}  ({len(run)} predictions)")

    model_key = "am"
    merged: list[dict] = []
    per_cat_f1: dict[str, list[float]] = defaultdict(list)
    per_cat_n: dict[str, int] = defaultdict(int)
    missing = 0

    for sample in dataset:
        sid = sample["sample_id"]
        qas = []
        for i, qa in enumerate(sample["qa"]):
            qid = f"{sid}-qa-{i:03d}"
            pred = run.get(qid)
            if pred is None:
                missing += 1
                continue
            item = {k: v for k, v in qa.items()}
            item[f"{model_key}_prediction"] = pred["predicted_answer"]
            qas.append(item)
        merged.append({"sample_id": sid, "qa": qas})

    if missing:
        print(f"WARNING: {missing} questions without a stored prediction were skipped")

    total = 0
    for sample in merged:
        ems, _lengths, _recall = official.eval_question_answering(
            sample["qa"], f"{model_key}_prediction", metric="f1"
        )
        for qa, em in zip(sample["qa"], ems):
            per_cat_f1[CAT.get(qa["category"], str(qa["category"]))].append(float(em))
            per_cat_n[qa["category"]] += 1
            total += 1

    all_f1 = [v for vals in per_cat_f1.values() for v in vals]
    overall = sum(all_f1) / len(all_f1) * 100 if all_f1 else 0.0
    header = f"{'category':<14}{'n':>6}{'official F1':>13}"
    print()
    print(header)
    print("-" * len(header))
    for cat in [1, 2, 3, 4, 5]:
        name = CAT[cat]
        vals = per_cat_f1.get(name, [])
        if vals:
            print(f"{name:<14}{len(vals):>6}{sum(vals) / len(vals) * 100:>12.2f}%")
    print("-" * len(header))
    print(f"{'ALL':<14}{len(all_f1):>6}{overall:>12.2f}%")

    no_adv = [v for cat in [1, 2, 3, 4] for v in per_cat_f1.get(CAT[cat], [])]
    no_adv_score = sum(no_adv) / len(no_adv) * 100 if no_adv else 0.0
    print(f"{'ALL (no cat5)':<14}{len(no_adv):>6}{no_adv_score:>12.2f}%")

    out = REPO / args.out
    payload = {
        "tag": args.tag or Path(args.run).name,
        "run_dir": str(args.run),
        "scorer": "official locomo task_eval/evaluation.py (metric=f1)",
        "n": len(all_f1),
        "official_f1": round(overall, 4),
        "official_f1_no_cat5": round(no_adv_score, 4),
        "per_category": {
            CAT[c]: {
                "n": len(per_cat_f1.get(CAT[c], [])),
                "official_f1": round(
                    sum(per_cat_f1.get(CAT[c], [])) / len(per_cat_f1.get(CAT[c], [])) * 100, 4
                ) if per_cat_f1.get(CAT[c]) else None,
            }
            for c in CAT
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
