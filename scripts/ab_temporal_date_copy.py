"""Cached-context A/B for LoCoMo category 2 (temporal): deterministic date copy.

Root cause fixed in this round: TemporalNormalizer rule 10 only matched full
weekday names, so chat turns saying "Last Fri" kept the bare relative phrase
in the compiled context.  After the abbreviation fix the context now carries
"last Friday (the Friday before 15 July 2023)" — the model only needs to COPY
the parenthesized absolute expression (no arithmetic, unlike the rejected
temporal_v1 directive that measured -1.2pp).

Arms (freshly compiled contexts, deterministic; production compiler settings):
  * stored    : no LLM — directives_v4 predictions re-scored (fidelity anchor)
  * plain_new : new contexts + plain prompt (isolates normalizer-only effect)
  * date_copy : new contexts + copy-the-date directive (combined effect)

Usage: python scripts/ab_temporal_date_copy.py [--convs 0,1,2] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

STORED_RUN = Path("benchmark_results/locomo10_runs/directives_v4")
OUT = Path("benchmark_results/ab_temporal_date_copy.json")

DATE_COPY_DIRECTIVE = (
    "[INSTRUCTION: TEMPORAL DATE EXTRACTION]\n"
    "Answer 'when' questions with the exact date or date expression found in\n"
    "the dialogue context below.\n"
    "- If a turn contains a date expression in parentheses, for example\n"
    "  'last Friday (the Friday before 15 July 2023)', the parenthesized part\n"
    "  is the precise answer. Quote it exactly as written.\n"
    "- If the context states a concrete date (e.g. '2 July 2023'), answer with\n"
    "  that exact date.\n"
    "- Never answer with a relative phrase like 'yesterday', 'last week' or\n"
    "  'a while ago'; always use the absolute date expression from the context.\n"
    "- If the context contains no date for the event, reply exactly: I don't know.\n\n"
    "{context}"
)


def score(adapter: LoCoMoAdapter, cat: int, gt: str, ans: str) -> bool:
    gt_lower = gt.lower().strip()
    ans_lower = ans.lower().strip()
    if cat == 2:
        return adapter._temporal_answer_matches(gt_lower, ans_lower)
    return score_generic(adapter, gt_lower, ans_lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convs", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    convs = [int(c) for c in args.convs.split(",") if c.strip()]

    stored: dict[str, dict] = {}
    for p in sorted(STORED_RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            stored[r["question_id"]] = r

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()
    verifier = adapter.compiler.answer_verifier

    def _verify(q, pred, ctx):
        return verifier.verify(
            question=q, predicted_answer=pred, context=ctx, propositions=[],
            integrity_abstention_recommended="Proposition Integrity Warning" in ctx,
        ).verified_answer

    results = {"stored": {}, "plain_new": {}, "date_copy": {}}
    n_total = 0
    for conv in convs:
        turns, questions, ir_records = adapter.load_conversation(conv_idx=conv)
        q_sub = [q for q in questions if q.category == 2]
        if args.limit:
            q_sub = q_sub[: args.limit]
        for q in q_sub:
            n_total += 1
            # Freshly compiled context with the abbreviation-fixed normalizer.
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text
            s = stored.get(q.question_id, {})
            results["stored"][q.question_id] = {
                "pred": s.get("predicted_answer", ""),
                "correct": score(adapter, 2, q.ground_truth, s.get("predicted_answer", "")),
            }
            # plain_new
            ans = answerer.answer(q.question, ctx)
            pred = _verify(q.question, ans.text, ctx)
            results["plain_new"][q.question_id] = {
                "pred": pred, "correct": score(adapter, 2, q.ground_truth, pred),
            }
            # date_copy
            ans = answerer.answer(q.question, DATE_COPY_DIRECTIVE.format(context=ctx))
            pred = _verify(q.question, ans.text, ctx)
            results["date_copy"][q.question_id] = {
                "pred": pred, "correct": score(adapter, 2, q.ground_truth, pred),
            }
            if n_total % 25 == 0:
                line = f"[{n_total}]"
                for a in ("stored", "plain_new", "date_copy"):
                    res = results[a]
                    line += f" {a}={sum(1 for v in res.values() if v['correct']) / len(res) * 100:.1f}%"
                print(line, flush=True)

    print(f"\n===== FINAL (temporal, n={n_total}) =====")
    for a in ("stored", "plain_new", "date_copy"):
        res = results[a]
        acc = sum(1 for v in res.values() if v["correct"]) / len(res) * 100
        print(f"{a:<10}: {acc:.1f}%")
    base, var = results["plain_new"], results["date_copy"]
    gain = sum(1 for k in var if var[k]["correct"] and not base[k]["correct"])
    loss = sum(1 for k in var if not var[k]["correct"] and base[k]["correct"])
    print(f"date_copy vs plain_new: gain {gain} / loss {loss}")

    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
