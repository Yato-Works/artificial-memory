"""Question-topic coverage as an adversarial (unanswerable) signal.

For every question we measure how much of the *question topic vocabulary*
(content words after stopword removal) actually appears anywhere in the
compiled context.  Hypothesis: unanswerable (adversarial) questions ask about
topics the conversation never touches, so their topic coverage is near zero,
while answerable questions score high.

Cross-tabulated against the frozen baseline's correctness so the gate can be
sized before any production change.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

STOP = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "am", "has", "have", "had", "the", "and",
    "for", "with", "about", "from", "that", "this", "these", "those", "there",
    "their", "his", "her", "its", "our", "your", "my", "me", "him", "them",
    "you", "she", "he", "it", "they", "we", "i", "in", "on", "at", "to", "of",
    "a", "an", "or", "not", "any", "some", "once", "after", "before", "during",
    "would", "could", "should", "can", "will", "might", "may", "must", "been",
    "being", "also", "very", "more", "most", "than", "then", "when", "while",
    "both", "all", "each", "every", "same", "other", "another", "such", "like",
    "go", "going", "get", "got", "make", "made", "take", "took", "say", "said",
    "tell", "told", "ask", "asked", "know", "think", "thought", "want", "need",
}


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[a-z0-9']+\b", text.lower())
            if w not in STOP and len(w) > 2]


def main() -> None:
    baseline = {}
    for p in sorted(Path("benchmark_results/locomo10").glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            baseline[r["question_id"]] = r

    adapter = LoCoMoAdapter()
    rows = []
    for conv in (0, 1, 2):
        _, questions, ir_records = adapter.load_conversation(conv_idx=conv)
        weights = EvidenceScoreWeights()
        for q in questions:
            pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
            ctx = pcc.context_text.lower()
            words = content_words(q.question)
            if not words:
                cov = 1.0
            else:
                hit = sum(1 for w in words if w in ctx)
                cov = hit / len(words)
            base = baseline.get(q.question_id, {})
            rows.append({
                "question_id": q.question_id,
                "category": q.category,
                "coverage": cov,
                "n_words": len(words),
                "base_correct": bool(base.get("is_correct")),
                "flagged": "Proposition Integrity Warning" in pcc.context_text,
                "question": q.question,
            })

    Path("benchmark_results/probe_topic_coverage.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{'cat':>4}{'n':>6}{'mean cov':>10}{'cov=0':>8}{'cov<0.34':>10}{'cov>=0.6':>10}")
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        n = len(rs)
        print(f"{cat:>4}{n:>6}{sum(r['coverage'] for r in rs) / n:>10.2f}"
              f"{sum(1 for r in rs if r['coverage'] == 0):>8}"
              f"{sum(1 for r in rs if r['coverage'] < 0.34):>10}"
              f"{sum(1 for r in rs if r['coverage'] >= 0.6):>10}")

    print("\nBAND ANALYSIS (baseline correctness by coverage band)")
    bands = [(0.0, 0.0), (0.01, 0.33), (0.34, 0.59), (0.6, 1.0)]
    print(f"{'cat':>4}{'band':>14}{'n':>6}{'base correct':>14}")
    for cat in sorted(by_cat):
        for lo, hi in bands:
            rs = [r for r in by_cat[cat] if lo <= r["coverage"] <= hi]
            if not rs:
                continue
            c = sum(1 for r in rs if r["base_correct"])
            print(f"{cat:>4}{f'{lo:.2f}-{hi:.2f}':>14}{len(rs):>6}{c / len(rs) * 100:>13.1f}%")

    print("\nADVERSARIAL (cat 5) flagged vs unflagged coverage")
    cat5 = by_cat.get(5, [])
    for flag in (True, False):
        rs = [r for r in cat5 if r["flagged"] == flag]
        if rs:
            print(f"  flagged={flag}: n={len(rs)} mean_cov={sum(r['coverage'] for r in rs) / len(rs):.2f} "
                  f"cov=0: {sum(1 for r in rs if r['coverage'] == 0)}")

    print("\nSample unflagged adversarial with zero topic coverage:")
    for r in cat5:
        if not r["flagged"] and r["coverage"] == 0:
            print(f"  {r['question_id']} base_correct={r['base_correct']} | {r['question']}")


if __name__ == "__main__":
    main()
