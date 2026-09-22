"""Simulate an evidence-grounding refusal gate on a completed LoCoMo run.

For every question of a saved run we recompile the context and measure how much
of the saved answer's content is actually grounded in that context.  We then
simulate a deterministic gate (low-grounding assertions -> refusal) and re-score
with the frozen scorer, yielding the exact net effect WITHOUT any LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

STOP = {
    "the", "and", "for", "with", "about", "from", "that", "this", "there", "their",
    "his", "her", "its", "our", "your", "you", "she", "he", "they", "was", "were",
    "is", "are", "not", "but", "yes", "no", "know", "don't", "does", "did", "has",
    "had", "have", "very", "also", "just", "been", "would", "could", "should",
    "mentioned", "mention", "information", "context", "answer", "question",
    "doing", "because", "while", "when", "what", "where", "which", "who",
    "how", "why", "some", "any", "all", "more", "most", "other", "than", "then",
    "them", "these", "those", "being", "into", "onto", "over", "under", "again",
}

REFUSAL_MARKERS = [
    "i don't know", "i dont know", "not mentioned", "unknown", "no information",
    "none", "unclear", "not specified", "cannot determine", "no record",
    "does not mention", "doesn't mention", "not in the conversation",
]

CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
refusals_ok: dict[int, float] = {}


def content_words(text: str) -> list[str]:
    out = []
    for w in text.lower().split():
        w = w.strip(".,!?;:'\"()")
        if w not in STOP and len(w) > 2:
            out.append(w)
    return out



def main() -> None:
    exp_root = Path("benchmark_results/locomo10_runs/selection_v3_llm")
    convs_all = sorted(int(p.stem.split("_")[1]) for p in exp_root.glob("conv_*_results.json"))
    if not convs_all:
        print("no experiment results", flush=True)
        return
    if len(sys.argv) > 1:
        convs = [c for c in convs_all if c in {int(x) for x in sys.argv[1].split(",")}]
    else:
        convs = convs_all

    exp = {}
    for c in convs:
        for r in json.load(open(exp_root / f"conv_{c}_results.json", encoding="utf-8"))["results"]:
            exp[r["question_id"]] = r

    adapter = LoCoMoAdapter()
    weights = EvidenceScoreWeights()
    ws = getattr(adapter.compiler, "wide_slicer", None)
    stem = ws._stem if ws and hasattr(ws, "_stem") else (lambda w: w)

    rows_path = Path("benchmark_results/grounding_gate_rows.json")
    if rows_path.exists():
        rows = json.load(open(rows_path, encoding="utf-8"))
    else:
        rows = []  # (category, was_correct, grounded_ratio|None, is_refusal)
        for conv in convs:
            _, questions, ir_records = adapter.load_conversation(conv_idx=conv)
            for q in questions:
                row = exp.get(q.question_id)
                if row is None:
                    continue
                ans = str(row["predicted_answer"])
                al = ans.lower().strip()
                is_refusal = any(m in al for m in REFUSAL_MARKERS) or len(al) <= 3
                if is_refusal:
                    rows.append((q.category, bool(row["is_correct"]), None, True))
                    continue
                pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
                ctx = pcc.context_text.lower()
                ctx_stems = {stem(w) for w in ctx.split()}
                words = content_words(ans)
                if not words:
                    rows.append((q.category, bool(row["is_correct"]), None, True))
                    continue
                grounded = sum(1 for w in words if w in ctx or stem(w) in ctx_stems)
                rows.append((q.category, bool(row["is_correct"]), grounded / len(words), False))
            print(f"conv {conv} done ({len(rows)} rows)", flush=True)
        if len(convs) == len(convs_all):
            with open(rows_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh)

    for cat in (1, 2, 3, 4, 5):
        ref = [r for r in rows if r[0] == cat and r[3]]
        if ref:
            refusals_ok[cat] = sum(1 for r in ref if r[1]) / len(ref)

    print(f"\nanalysed {len(rows)} answers", flush=True)
    print("\nanswer correctness by grounded-content ratio (non-refusal answers):")
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"{'band':<12}{'n':>6}{'correct':>9}{'wrong':>7}{'acc%':>7}")
    for lo, hi in bands:
        sel = [r for r in rows if r[2] is not None and lo <= r[2] < hi]
        c = sum(1 for r in sel if r[1])
        n = len(sel)
        if n:
            print(f"{f'{lo:.1f}-{hi:.1f}':<12}{n:>6}{c:>9}{n - c:>7}{c / n * 100:>6.1f}%")

    print("\nper-category: refusal answers | low(<0.34) | high(>=0.34) grounding")
    for cat in (1, 2, 3, 4, 5):
        sel = [r for r in rows if r[0] == cat]
        if not sel:
            continue
        ref = [r for r in sel if r[3]]
        low = [r for r in sel if not r[3] and r[2] is not None and r[2] < 0.34]
        high = [r for r in sel if not r[3] and r[2] is not None and r[2] >= 0.34]

        def cc(rs):
            return sum(1 for r in rs if r[1])
        print(f"  {CAT_NAMES[cat]:<12} refusal {len(ref):>4} (ok {cc(ref):>4}) | "
              f"low {len(low):>4} (ok {cc(low):>4}, wrong {len(low) - cc(low):>4}) | "
              f"high {len(high):>4} (ok {cc(high):>4}, wrong {len(high) - cc(high):>4})")

    n = len(rows)
    base_ok = sum(1 for r in rows if r[1])
    print(f"\ncurrent run: {base_ok}/{n} = {base_ok / n * 100:.2f}%")
    for thr in (0.15, 0.25, 0.34, 0.5, 0.67):
        sim = 0
        for r in rows:
            if r[3] or r[2] is None:
                sim += r[1]
            elif r[2] < thr:
                sim += refusals_ok.get(r[0], 0.0)
            else:
                sim += r[1]
        print(f"  gate thr<{thr:.2f}: simulated {sim:.0f}/{n} = {sim / n * 100:.2f}%")


if __name__ == "__main__":
    main()
