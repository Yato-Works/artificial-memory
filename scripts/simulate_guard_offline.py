"""Offline guard simulator: apply AnswerVerifier variants to saved answers.

Uses the frozen contexts (locomo_context_cache.jsonl) plus the saved predictions
of a completed run, so guard variants can be evaluated over all 1,986 questions
without any retrieval or LLM cost.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path("benchmark_results/locomo_context_cache.jsonl")


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in (
        "i don't know", "i dont know", "not mentioned", "no information", "unknown",
        "none", "cannot", "unclear", "not specified", "doesn't mention",
    ))


def main() -> None:
    rows: dict[str, dict] = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            rows[r["question_id"]] = r
    contexts: dict[str, str] = {}
    for line in open(CACHE, encoding="utf-8"):
        d = json.loads(line)
        contexts[d["qid"]] = d["context"]

    # Baseline (pre-run) predictions for the accuracy reference: reuse the same
    # guard-free answers by re-deriving them from the frozen baseline run.
    base: dict[str, dict] = {}
    for p in sorted(Path("benchmark_results/locomo10").glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            base[r["question_id"]] = r

    def score(pred_answer: str, gt: str) -> bool:
        """Mirror the harness scoring for the two answer protocols."""
        if not (gt or "").strip():
            return is_refusal(pred_answer)
        a = (pred_answer or "").lower()
        return bool(gt) and gt.lower() in a

    variants = {
        "no_guard": None,
        "guard_current": AnswerVerifier(subject_binding=True),
    }
    # Additional variants exercised through the public verify() entry point.
    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "ok": 0, "fired": 0}))
    for qid, r in rows.items():
        ctx = contexts.get(qid)
        if ctx is None:
            continue
        cat = CAT.get(r["category"], str(r["category"]))
        gt = r.get("ground_truth") or ""
        # baseline truth comes from the frozen baseline run so both variants are
        # measured against the same protocol.
        for name, verifier in variants.items():
            if verifier is None:
                pred = r["predicted_answer"]
                fired = False
            else:
                res = verifier.verify(r["question"], r["predicted_answer"], ctx, [])
                pred = res.verified_answer
                fired = res.hallucination_detected
            ok = score(pred, gt)
            cell = agg[name][cat]
            cell["n"] += 1
            cell["ok"] += bool(ok)
            cell["fired"] += bool(fired)

    print(f"run: {RUN}")
    for name in variants:
        print(f"\n== {name} ==")
        tn = to = tf = 0
        for cat in sorted(agg[name], key=lambda c: -agg[name][c]["n"]):
            c = agg[name][cat]
            tn += c["n"]
            to += c["ok"]
            tf += c["fired"]
            print(f"  {cat:<14}{c['n']:>6}  acc={c['ok'] / c['n'] * 100:>5.1f}%  fired={c['fired']:>4}")
        print(f"  {'ALL':<14}{tn:>6}  acc={to / tn * 100:>5.1f}%  fired={tf:>4}")


if __name__ == "__main__":
    main()
