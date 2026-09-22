"""Cached-context prompt A/B for LoCoMo categories 1 (multi-hop) and 2 (temporal).

Both categories currently answer with the *plain* frozen prompt (no directive),
unlike cats 3/4 which have dedicated instruction prompts.  This probe replays
the cached production contexts through the LLM with candidate directives and
scores everything with the production scorers + production AnswerVerifier.

Cost: LLM calls only (no recompilation).  Baseline arm must reproduce the
stored per-category accuracy to prove harness fidelity.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CATS = (1, 2)
ARMS = ("baseline", "temporal_v1", "multihop_v1")

TEMPORAL_DIRECTIVE = (
    "[INSTRUCTION: TEMPORAL REASONING]\n"
    "Answer the question using ONLY the dialogue context below.\n"
    "- Each context block starts with a provenance tag like [D12 on May 7, 2022].\n"
    "  Use these session dates to resolve relative time expressions.\n"
    "- Give the most SPECIFIC absolute answer you can (day, month, year, weekday\n"
    "  or duration), never a vague phrase like 'recently' or 'a while ago'.\n"
    "- If the question asks for a relative period (e.g. 'last week', 'the week\n"
    "  before <date>'), resolve it against the nearest session date and state the\n"
    "  resulting absolute period in your answer.\n"
    "- If the context does not contain the answer, reply exactly: I don't know.\n\n"
    "{context}"
)

MULTIHOP_DIRECTIVE = (
    "[INSTRUCTION: MULTI-HOP EVIDENCE SYNTHESIS]\n"
    "Answer the question using ONLY the dialogue context below.\n"
    "- The answer may require combining facts from several sessions or both\n"
    "  speakers. Identify every part of the question first, find the evidence\n"
    "  for each part, then combine them.\n"
    "- Name every item/person/event the question asks about; never answer with\n"
    "  only one part of a multi-part question.\n"
    "- Quote names and facts exactly as they appear in the context.\n"
    "- If the context does not contain the answer, reply exactly: I don't know.\n\n"
    "{context}"
)



def build_prompt(cat: int, context: str, arm: str) -> str:
    if arm == "baseline":
        return context
    if cat == 2 and arm == "temporal_v1":
        return TEMPORAL_DIRECTIVE.format(context=context)
    if cat == 1 and arm == "multihop_v1":
        return MULTIHOP_DIRECTIVE.format(context=context)
    return context


def score_generic(adapter: LoCoMoAdapter, gt_lower: str, ans_lower: str) -> bool:
    """Replicate the production generic scorer (adapter lines 411-444)."""
    if not gt_lower:
        return any(w in ans_lower for w in
                   ["i don't know", "not mentioned", "unknown", "unclear",
                    "no information", "none", "no"])
    if gt_lower in ans_lower or ans_lower in gt_lower:
        return True
    clean_gt = re.sub(r"\bde-stress\b", "destress", gt_lower).replace("-", " ")
    clean_ans = re.sub(r"\bde-stress\b", "destress", ans_lower).replace("-", " ")
    for w, n in adapter._NUMBER_WORDS.items():
        clean_gt = re.sub(rf"\b{w}\b", n, clean_gt)
        clean_ans = re.sub(rf"\b{w}\b", n, clean_ans)
    if clean_gt in clean_ans or clean_ans in clean_gt:
        return True
    gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_gt)
                   if len(w) > 2 or w.isdigit())
    ans_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_ans)
                    if len(w) > 2 or w.isdigit())
    if gt_words and ans_words:
        ws = getattr(adapter.compiler, "wide_slicer", None)
        if ws and hasattr(ws, "_stem"):
            gt_stems = {ws._stem(w) for w in gt_words}
            ans_stems = {ws._stem(w) for w in ans_words}
            overlap = max(len(gt_words & ans_words), len(gt_stems & ans_stems))
        else:
            overlap = len(gt_words & ans_words)
        if overlap / len(gt_words) >= 0.33:
            return True
        if len(gt_words) <= 3 and overlap >= 1:
            return True
    return False


def score(adapter: LoCoMoAdapter, cat: int, gt: str, ans: str) -> bool:
    gt_lower = str(gt).lower().strip()
    ans_lower = ans.lower().strip()
    if cat == 2:
        return adapter._temporal_answer_matches(gt_lower, ans_lower)
    return score_generic(adapter, gt_lower, ans_lower)


def main() -> None:
    rows = [json.loads(line) for line in CACHE.open(encoding="utf-8")]
    rows = [r for r in rows if r["category"] in CATS]
    run_root = Path("benchmark_results/locomo10_runs/subject_binding_v2")
    stored: dict[str, dict] = {}
    for p in sorted(run_root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            stored[r["question_id"]] = r
    joined = []
    for r in rows:
        s = stored.get(r["qid"])
        if s is None:
            continue
        joined.append({**r, "gt": s["ground_truth"], "stored_pred": s["predicted_answer"]})
    print(f"questions: {len(joined)} (cat1={sum(1 for r in joined if r['category'] == 1)}, "
          f"cat2={sum(1 for r in joined if r['category'] == 2)})")

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()
    verifier = adapter.compiler.answer_verifier

    def _verify(q, pred, ctx):
        return verifier.verify(
            question=q, predicted_answer=pred, context=ctx, propositions=[],
            integrity_abstention_recommended="Proposition Integrity Warning" in ctx,
        ).verified_answer

    results: dict[str, dict[str, dict]] = {"baseline": {}, "temporal_v1": {}, "multihop_v1": {}}
    n = len(joined)
    for i, r in enumerate(joined):
        q, ctx, cat = r["question"], r["context"], r["category"]
        results["baseline"][r["qid"]] = {
            "pred": r["stored_pred"],
            "correct": score(adapter, cat, r["gt"], r["stored_pred"]),
        }
        if cat == 2:
            arm = "temporal_v1"
        elif cat == 1:
            arm = "multihop_v1"
        else:
            continue
        prompt = build_prompt(cat, ctx, arm)
        ans = answerer.answer(q, prompt)
        pred = _verify(q, ans.text, ctx)
        results[arm][r["qid"]] = {"pred": pred, "correct": score(adapter, cat, r["gt"], pred)}
        if (i + 1) % 50 == 0:
            line = f"[{i + 1}/{n}]"
            for a in ARMS:
                res = results[a]
                if res:
                    line += f" {a}={sum(x['correct'] for x in res.values()) / len(res) * 100:.1f}%"
            print(line, flush=True)

    print("\n===== FINAL (production scorers) =====")
    for cat, arm in ((2, "temporal_v1"), (1, "multihop_v1")):
        sub = [r for r in joined if r["category"] == cat]
        cn = len(sub)
        b = sum(results["baseline"][r["qid"]]["correct"] for r in sub)
        v = sum(results[arm][r["qid"]]["correct"] for r in sub)
        gain = sum(1 for r in sub if results[arm][r["qid"]]["correct"]
                   and not results["baseline"][r["qid"]]["correct"])
        loss = sum(1 for r in sub if not results[arm][r["qid"]]["correct"]
                   and results["baseline"][r["qid"]]["correct"])
        print(f"cat{cat}: n={cn} baseline {b / cn * 100:.1f}% -> variant {v / cn * 100:.1f}% "
              f"({(v - b) / cn * 100:+.1f}pp, gain {gain} / loss {loss})")

    out = Path("benchmark_results/ab_cat12_prompt.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump({a: {k: {"correct": v["correct"], "pred": v["pred"]}
                       for k, v in res.items()} for a, res in results.items()},
                  fh, indent=1)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

