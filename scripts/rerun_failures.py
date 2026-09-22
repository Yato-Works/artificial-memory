"""Failure-focused iteration loop for the LoCoMo campaign.

Re-answers ONLY the questions that are currently wrong (plus a random control
sample of currently-correct ones as a regression sensor), instead of re-running
the full 1,986-question benchmark every time.  Iteration cost drops from hours
to minutes and shrinks as the failure set shrinks.

Modes
-----
cached : retrieval is FROZEN by replaying benchmark_results/locomo_context_cache.jsonl
         (contexts from the v4 compile).  Valid for reader / prompt / guard
         changes, where the only variable is what happens after retrieval.
full   : fresh retrieval + fresh answer for the selected questions.  Required
         for retrieval-side changes (window cap, rescue, ranking).  NOTE: a
         full-mode delta only measures the retested questions; contexts of
         non-retested questions also changed, so projections are invalid there.

Guardrail (permanent rule, agreed 2026-09-23)
---------------------------------------------
Fixes validated through this loop must be MECHANISMS that generalize across
questions (retrieval, ranking, guards, prompt structure, reader model).
Per-question keyword -> answer tables (e.g. answer-injection directives) are
FORBIDDEN: they turn the benchmark into memorised answers and invalidate the
claim "AM retrieves and compiles evidence".  If a failure cannot be fixed by a
mechanism, the next mechanism (better reader / better retrieval) is the answer,
not a lookup table.

Usage:
  python scripts/rerun_failures.py --base reader7b_cap32 --mode cached --tag round1
  python scripts/rerun_failures.py --base reader7b_cap32 --mode cached --tag round1 \\
      --model qwen3:4b --window-cap 32
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import rescore  # noqa: E402

from artificial_memory.research.benchmarks.external.locomo_adapter import (  # noqa: E402
    CachedContext,
    LoCoMoAdapter,
)
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer  # noqa: E402

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
ROUND_ROOT = Path("benchmark_results/failure_rounds")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def select_questions(base: dict[str, dict], controls: int, seed: int) -> tuple[list[str], list[str]]:
    """Failures first, plus a random sample of correct ones (regression sensor)."""
    failures = [qid for qid, r in base.items() if not r["rescored_correct"]]
    correct = [qid for qid, r in base.items() if r["rescored_correct"]]
    rng = random.Random(seed)
    chosen_controls = rng.sample(correct, min(controls, len(correct)))
    return failures, chosen_controls


def answer_selected(adapter: LoCoMoAdapter, answerer, qids: set[str], mode: str,
                    window_cap: int | None, token_bonus: int | None) -> list[dict]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    if window_cap is not None:
        adapter.compiler.selection_window_cap = window_cap
    if token_bonus is not None:
        adapter.compiler.rescue_token_bonus_per_unit = token_bonus
    for conv_idx in range(10):
        turns, questions, ir_records = adapter.load_conversation(conv_idx)
        todo = [q for q in questions if q.question_id in qids]
        if not todo:
            continue
        for i, q in enumerate(todo):
            res = adapter.evaluate_question(q, turns, ir_records, answerer)
            rows.append({
                "qid": q.question_id,
                "category": q.category,
                "is_correct": bool(res.is_correct),
                "oracle": bool(res.oracle_recall),
                "predicted": res.predicted_answer,
                "ground_truth": q.ground_truth,
                "tokens": res.tokens_used,
                "latency_ms": res.latency_ms,
            })
            done = len(rows)
            if done % 25 == 0 or done == len(qids) or i + 1 == len(todo):
                el = time.perf_counter() - t0
                print(f"  [{done}/{len(qids)}] {done / el:.1f} Q/s  {q.question_id}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Run dir name under locomo10_runs")
    ap.add_argument("--mode", choices=["cached", "full"], default="cached")
    ap.add_argument("--tag", required=True, help="Round tag, e.g. round1")
    ap.add_argument("--model", default=None, help="Reader override (default: frozen phi4-mini)")
    ap.add_argument("--window-cap", type=int, default=None)
    ap.add_argument("--token-bonus", type=int, default=None)
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-failures", type=int, default=None,
                    help="Debug: cap the number of failures retested")
    args = ap.parse_args()

    base = load_base_run(args.base)
    failures, control_qids = select_questions(base, args.controls, args.seed)
    if args.limit_failures is not None:
        failures = failures[: args.limit_failures]
    retest = failures + control_qids
    print(f"base={args.base}  failures={len(failures)}  controls={len(control_qids)}  "
          f"mode={args.mode}  model={args.model or FROZEN_MODEL}")

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer(model=args.model) if args.model else OllamaAnswerer()
    if args.mode == "cached":
        adapter.context_override = load_cache(CACHE)
        print(f"cached contexts loaded: {len(adapter.context_override)} (retrieval frozen)")
    else:
        print("full mode: fresh retrieval per question")

    rows = answer_selected(adapter, answerer, set(retest), args.mode,
                           args.window_cap, args.token_bonus)

    out_dir = ROUND_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fixed = [r["qid"] for r in rows if r["qid"] in set(failures) and r["is_correct"]]
    still = [r["qid"] for r in rows if r["qid"] in set(failures) and not r["is_correct"]]
    regressed = [r["qid"] for r in rows
                 if r["qid"] in set(control_qids) and not r["is_correct"]]
    next_failures = sorted(still + regressed)
    (out_dir / "failures_next.txt").write_text("\n".join(next_failures), encoding="utf-8")

    summary = {
        "base": args.base, "mode": args.mode, "model": args.model or FROZEN_MODEL,
        "n_failures": len(failures), "n_controls": len(control_qids),
        "fixed": len(fixed), "still_wrong": len(still), "regressed": len(regressed),
        "net": len(fixed) - len(regressed),
        "fix_rate": round(len(fixed) / max(len(failures), 1) * 100, 1),
        "regression_rate": round(len(regressed) / max(len(control_qids), 1) * 100, 1),
        "next_failure_count": len(next_failures),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"saved: {rows_path}\nnext failures: {out_dir / 'failures_next.txt'}")


if __name__ == "__main__":
    main()
