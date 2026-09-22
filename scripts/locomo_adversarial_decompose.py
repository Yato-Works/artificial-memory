"""Adversarial failure decomposition on a completed LoCoMo run.

For every category-5 (hallucination-bait) question we bin the outcome by
*why* the produced answer was accepted/rejected, using only information the
system could legitimately have at answer time (question + compiled context):

  REFUSAL_OK        refusal marker emitted (protocol-correct) -> scored correct
  ANSWERED_WRONG    a concrete answer was emitted -> scored wrong
  ANSWERED_RIGHT    concrete answer that happens to pass the acceptance scorer

For ANSWERED_WRONG we additionally measure whether the accepted proposition can
be traced back to a turn whose subject is the queried person (subject binding)
or is merely *text present somewhere* in the wide context (the bait signature).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REFUSAL = ["i don't know", "not mentioned", "unknown", "unclear", "no information",
           "none", "no"]
STOP = {
    "what", "when", "where", "which", "who", "whom", "how", "why", "did", "does",
    "was", "were", "has", "have", "had", "the", "and", "for", "with", "about",
    "from", "that", "this", "there", "their", "his", "her", "its", "our", "your",
    "you", "she", "he", "they", "them", "was", "are", "is", "not", "but", "yes",
    "know", "does", "did", "has", "had", "very", "also", "just", "been", "would",
    "could", "should", "into", "over", "than", "then", "some", "any", "how",
    "much", "many", "get", "got", "make", "made", "new", "one", "two", "first",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[a-z0-9']+\b", text.lower())
            if w not in STOP and len(w) > 2}


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1
               else "benchmark_results/locomo10_runs/subject_binding_v1")
    rows: dict[str, dict] = {}
    for p in sorted(run.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            rows[r["question_id"]] = r

    adv = {k: v for k, v in rows.items() if v["category"] == 5}
    print(f"run={run.name} adversarial rows: {len(adv)}")

    cache = Path("benchmark_results/locomo_context_cache.jsonl")
    ctx_of: dict[str, str] = {}
    qtext: dict[str, str] = {}
    if cache.exists():
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("qid")
                if qid:
                    ctx_of[qid] = row.get("context", "") or ""
                    qtext[qid] = row.get("question", "") or ""
    print(f"cached contexts available: {len(ctx_of)}")

    bins: Counter = Counter()
    wrong_sample: list = []
    for qid, r in sorted(adv.items()):
        ans = str(r["predicted_answer"]).lower()
        ok = score(5, r["ground_truth"], r["predicted_answer"])
        refused = any(w in ans for w in REFUSAL)
        if refused and ok:
            bins["REFUSAL_OK"] += 1
        elif refused and not ok:
            bins["REFUSAL_REJECTED"] += 1
        elif ok:
            bins["ANSWERED_RIGHT(accepted)"] += 1
        else:
            bins["ANSWERED_WRONG"] += 1
            wrong_sample.append((qid, qtext.get(qid, ""), str(r["predicted_answer"])))

    for k, v in bins.most_common():
        print(f"  {k:<26}{v:>5}  {v / len(adv) * 100:>5.1f}%")

    support = defaultdict(int)
    shown = 0
    for qid, q, a in wrong_sample:
        ctx = ctx_of.get(qid, "")
        if not ctx:
            continue
        qw = content_words(q)
        if not qw:
            continue
        ratio = sum(1 for w in qw if w in ctx.lower()) / len(qw)
        band = "premise_supported" if ratio >= 0.6 else \
               "premise_partial" if ratio >= 0.3 else "premise_absent"
        support[band] += 1
        if shown < 10 and band == "premise_absent":
            print(f"  [{band}] {qid} ratio={ratio:.2f}\n      q={q}\n      pred={a[:70]!r}")
            shown += 1
    tot = sum(support.values())
    print(f"\nANSWERED_WRONG premise support (n={tot} with cached context):")
    for k, v in sorted(support.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20}{v:>5}  {v / max(1, tot) * 100:>5.1f}%")


if __name__ == "__main__":
    main()
