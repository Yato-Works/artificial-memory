"""Premise-verification gate simulation (offline, cached contexts).

Adversarial (cat-5) failures after widening are answers *synthesised* from
related context: the question's presupposed event/entity ("charity race",
"regionals", "Marley flooring") is not asserted anywhere, yet the model builds a
plausible-sounding 3-4 word answer.  A general memory-system guard should refuse
when the question's own premise is absent from memory.

This script measures that guard without any LLM calls:

  premise signal = question content words / adjacent content bigrams that do NOT
  appear anywhere in the compiled context.

  gain = questions whose current answer is scored wrong and would become correct
  cost = questions whose current answer is scored correct and would be broken

Usage:
  python scripts/simulate_premise_gate.py <exp_dir> [cache_path]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import CAT, score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

STOP = {
    "the", "and", "for", "with", "about", "from", "that", "this", "there", "their",
    "his", "her", "its", "our", "your", "you", "she", "he", "they", "was", "were",
    "is", "are", "not", "but", "yes", "know", "does", "did", "has", "had", "have",
    "very", "also", "just", "been", "would", "could", "should", "mentioned",
    "mention", "information", "context", "answer", "question", "doing", "because",
    "while", "when", "what", "where", "which", "who", "how", "why", "some", "any",
    "all", "more", "most", "other", "than", "then", "them", "these", "those",
    "being", "into", "onto", "over", "under", "again", "him", "much", "many",
    "like", "want", "get", "got", "make", "made", "take", "took", "time", "times",
    "day", "days", "year", "years", "week", "weeks", "month", "months", "today",
    "recently", "lately", "currently", "now", "still", "ever", "never", "long",
    "own", "same", "first", "last", "next", "new", "old", "get", "let", "put",
}


def norm_words(text: str) -> list[str]:
    out = []
    for raw in re.findall(r"[A-Za-z0-9']+", text.lower()):
        w = raw.strip("'")
        if len(w) > 2 and w not in STOP:
            out.append(w)
    return out


def stem(w: str) -> str:
    for suf in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w


def premise_features(question: str, context: str) -> tuple[int, int]:
    """Return (missing content stems, missing adjacent content bigrams)."""
    qw = norm_words(question)
    ctx = context.lower()
    ctx_stems = {stem(w) for w in norm_words(context)}
    missing_words = [w for w in set(qw) if stem(w) not in ctx_stems and w not in ctx]
    bigrams = list(zip(qw, qw[1:]))
    missing_bigrams = 0
    for a, b in bigrams:
        if f"{a} {b}" not in ctx and f"{stem(a)} {stem(b)}" not in ctx:
            missing_bigrams += 1
    return len(missing_words), missing_bigrams


def load_run(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def load_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["qid"]] = d
    return out


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
                   "benchmark_results/locomo10_runs/selection_v3_llm")
    cache_path = Path(sys.argv[2] if len(sys.argv) > 2 else
                      "benchmark_results/locomo_context_cache.jsonl")
    rows = load_run(run_dir)
    cache = load_cache(cache_path)
    print(f"run: {run_dir} ({len(rows)} Q) | cache: {cache_path} ({len(cache)} Q)")

    feats: dict[str, tuple[int, int]] = {}
    for qid, row in rows.items():
        c = cache.get(qid)
        if c is None:
            continue
        feats[qid] = premise_features(c["question"], c["context"])
    print(f"analysed: {len(feats)}")

    print("\nthreshold sweep (gate = refuse when premise signal exceeded)")
    print(f"{'rule':<34}{'gain':>6}{'cost':>6}{'net':>6}   by-category net")
    for min_missing_words, min_missing_bigrams in [
        (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (2, 2), (1, 2)
    ]:
        gain = cost = 0
        per_cat: dict[str, int] = defaultdict(int)
        for qid, (mw, mb) in feats.items():
            if not (mw >= min_missing_words and mb >= min_missing_bigrams):
                continue
            row = rows[qid]
            cat = CAT.get(row["category"], str(row["category"]))
            # Gate output is a refusal; score that refusal with the frozen scorer.
            before = score(row["category"], row["ground_truth"], row["predicted_answer"])
            after = score(row["category"], row["ground_truth"], "I don't know.")
            if after and not before:
                gain += 1
                per_cat[cat] += 1
            elif before and not after:
                cost += 1
                per_cat[cat] -= 1
        cats = " ".join(f"{k}:{v:+d}" for k, v in sorted(per_cat.items()) if v)
        print(f"  words>={min_missing_words} bigrams>={min_missing_bigrams:<16}"
              f"{gain:>6}{cost:>6}{gain - cost:>+6}   {cats}")


if __name__ == "__main__":
    main()

