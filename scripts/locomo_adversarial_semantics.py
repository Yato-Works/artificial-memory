"""Simulate the subject-binding guard on a completed LoCoMo run (no LLM calls).

Uses the cached contexts (benchmark_results/locomo_context_cache.jsonl) plus the
saved answers of the last run, applies the production ``AnswerVerifier``
subject-binding gate, and re-scores both variants with the frozen scorer.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
RUN = Path("benchmark_results/locomo10_runs/selection_v3_llm")
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

refusal_words = ["i don't know", "not mentioned", "unknown", "unclear", "no information", "none", "no"]


def score(adapter: LoCoMoAdapter, category: int, gt_lower: str, ans_lower: str) -> bool:
    if category == 2:
        return adapter._temporal_answer_matches(gt_lower, ans_lower)
    if category == 3:
        return adapter._open_domain_answer_matches(gt_lower, ans_lower)
    if category == 4:
        return adapter._single_hop_answer_matches(gt_lower, ans_lower)
    if not gt_lower:
        return any(w in ans_lower for w in refusal_words)
    if gt_lower in ans_lower or ans_lower in gt_lower:
        return True
    clean_gt = re.sub(r"\bde-stress\b", "destress", gt_lower).replace("-", " ")
    clean_ans = re.sub(r"\bde-stress\b", "destress", ans_lower).replace("-", " ")
    for w, n in adapter._NUMBER_WORDS.items():
        clean_gt = re.sub(rf"\b{w}\b", n, clean_gt)
        clean_ans = re.sub(rf"\b{w}\b", n, clean_ans)
    if clean_gt in clean_ans or clean_ans in clean_gt:
        return True
    gt_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_gt) if len(w) > 2 or w.isdigit()}
    ans_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_ans) if len(w) > 2 or w.isdigit()}
    if gt_words and ans_words:
        ws = getattr(adapter.compiler, "wide_slicer", None)
        if ws and hasattr(ws, "_stem"):
            overlap = max(len(gt_words & ans_words),
                          len({ws._stem(w) for w in gt_words} & {ws._stem(w) for w in ans_words}))
        else:
            overlap = len(gt_words & ans_words)
        if overlap / len(gt_words) >= 0.33 or (len(gt_words) <= 3 and overlap >= 1):
            return True
    return False


def main() -> None:
    saved = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            saved[r["question_id"]] = r

    cache = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d

    adapter = LoCoMoAdapter()
    verifier = AnswerVerifier(subject_binding=True)

    per_cat = defaultdict(lambda: {"n": 0, "base": 0, "gate": 0, "fired": 0,
                                   "recovered": 0, "broken": 0})
    examples = []
    for qid, row in saved.items():
        c = cache.get(qid)
        if c is None:
            continue
        cat = int(row["category"])
        a = per_cat[cat]
        a["n"] += 1
        base_ok = bool(row["is_correct"])
        a["base"] += base_ok
        new_ans = verifier.check_subject_binding(
            c["question"], str(row["predicted_answer"]), c["context"]
        )
        if new_ans is None:
            a["gate"] += base_ok
            continue
        a["fired"] += 1
        gt_lower = str(row["ground_truth"]).lower().strip()
        gated_ok = score(adapter, cat, gt_lower, new_ans.verified_answer.lower().strip())
        a["gate"] += gated_ok
        if gated_ok and not base_ok:
            a["recovered"] += 1
        elif base_ok and not gated_ok:
            a["broken"] += 1
            if len(examples) < 12:
                examples.append((qid, CAT_NAMES[cat], row["predicted_answer"][:60],
                                 row["ground_truth"][:40]))

    n = sum(a["n"] for a in per_cat.values())
    base = sum(a["base"] for a in per_cat.values())
    gate = sum(a["gate"] for a in per_cat.values())
    print(f"{'category':<13}{'n':>6}{'base%':>8}{'gate%':>8}{'fired':>7}{'+rec':>6}{'-brk':>6}")
    for cat in (4, 5, 2, 1, 3):
        a = per_cat.get(cat)
        if not a or not a["n"]:
            continue
        print(f"{CAT_NAMES[cat]:<13}{a['n']:>6}{a['base'] / a['n'] * 100:>7.1f}%"
              f"{a['gate'] / a['n'] * 100:>7.1f}%{a['fired']:>7}{a['recovered']:>6}{a['broken']:>6}")
    print(f"{'ALL':<13}{n:>6}{base / n * 100:>7.1f}%{gate / n * 100:>7.1f}%"
          f"{sum(a['fired'] for a in per_cat.values()):>7}"
          f"{sum(a['recovered'] for a in per_cat.values()):>6}"
          f"{sum(a['broken'] for a in per_cat.values()):>6}")

    print("\nBROKEN examples (was correct, gate refused):")
    for qid, cat, ans, gt in examples:
        print(f"  [{cat}] {qid} gt={gt!r} ans={ans!r}")


def debug() -> None:
    """Trace the gate on known bait questions."""
    saved = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            saved[r["question_id"]] = r
    cache = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d
    v = AnswerVerifier(subject_binding=True)
    # adversarial wrongs with high grounding
    shown = 0
    for qid, row in saved.items():
        if int(row["category"]) != 5 or row["is_correct"]:
            continue
        c = cache.get(qid)
        if c is None:
            continue
        res = v.check_subject_binding(c["question"], str(row["predicted_answer"]), c["context"])
        if res is None and shown < 6:
            shown += 1
            turns = v._parse_turns(c["context"])
            speakers = sorted({s.lower() for s, _ in turns})
            stems = v._content_stems(str(row["predicted_answer"]))
            scored = []
            for speaker, text in turns:
                t = v._content_stems(text)
                scored.append((sum(1 for st in stems if st in t) / max(len(stems), 1), speaker))
            scored.sort(key=lambda x: -x[0])
            print(f"\nQID {qid}: {c['question']}")
            print(f"  ans={row['predicted_answer'][:70]!r}")
            print(f"  speakers={speakers} stems={sorted(stems)[:8]}")
            print(f"  top turns: {scored[:3]}")
            for line in c["context"].splitlines()[:14]:
                print(f"    |{line[:110]}")
            break_shown = True

def debug_qids() -> None:
    saved = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            saved[r["question_id"]] = r
    cache = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d
    v = AnswerVerifier(subject_binding=True)
    targets = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "conv-26-qa-024", "conv-26-qa-011", "conv-26-qa-015",
    ]
    for qid in targets:
        row = saved.get(qid)
        c = cache.get(qid)
        if row is None or c is None:
            print(f"{qid}: missing")
            continue
        print(f"\n=== {qid}: {c['question']}")
        print(f"  ans={row['predicted_answer']!r} gt={row['ground_truth']!r} correct={row['is_correct']}")
        turns = v._parse_turns(c["context"])
        print(f"  parsed turns: {len(turns)}")
        stems = v._content_stems(str(row["predicted_answer"]))
        scored = sorted(
            ((sum(1 for st in stems if st in v._content_stems(t)) / len(stems), s, t[:90])
             for s, t in turns),
            key=lambda x: -x[0],
        )
        for ratio, s, t in scored[:6]:
            print(f"    {ratio:.2f} [{s}] {t}")
        print(f"  raw lines (first 6):")
        for line in c["context"].splitlines()[:6]:
            print(f"      |{line[:130]}")


def debug_residual() -> None:
    """Show adversarial wrongs where the gate did NOT fire."""
    saved = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            saved[r["question_id"]] = r
    cache = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d
    v = AnswerVerifier(subject_binding=True)
    shown = 0
    for qid, row in saved.items():
        if int(row["category"]) != 5 or row["is_correct"]:
            continue
        c = cache.get(qid)
        if c is None:
            continue
        res = v.check_subject_binding(c["question"], str(row["predicted_answer"]), c["context"])
        if res is not None:
            continue
        ans = str(row["predicted_answer"]).lower().strip()
        if any(m in ans for m in ["i don't know", "not mentioned", "unknown", "none"]):
            continue
        shown += 1
        if shown > 10:
            break
        turns = v._parse_turns(c["context"])
        speakers = sorted({s for s, _ in turns})
        stems = v._content_stems(str(row["predicted_answer"]))
        scored = sorted(
            ((sum(1 for st in stems if st in v._content_stems(t)) / len(stems), s, t[:80])
             for s, t in turns),
            key=lambda x: -x[0],
        )
        print(f"\n{qid}: {c['question']}")
        print(f"  ans={row['predicted_answer'][:70]!r}")
        print(f"  speakers={speakers} top={[(round(r,2), s) for r, s, _ in scored[:3]]}")
        for ratio, s, t in scored[:2]:
            print(f"    {ratio:.2f} [{s}] {t}")


def debug_prod() -> None:
    """Run the PRODUCTION verify() path on saved bait answers."""
    saved = {}
    for p in sorted(RUN.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            saved[r["question_id"]] = r
    cache = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d
    adapter = LoCoMoAdapter()
    v = adapter.compiler.answer_verifier
    print("subject_binding flag:", v.subject_binding)
    fired = 0
    per_conv = {}
    for qid, row in saved.items():
        c = cache.get(qid)
        if c is None:
            continue
        res = v.verify(
            question=c["question"],
            predicted_answer=str(row["predicted_answer"]),
            context=c["context"],
            propositions=[],
            integrity_abstention_recommended=False,
        )
        if res.verified_answer != str(row["predicted_answer"]).strip():
            fired += 1
            conv = int(qid.split("-")[1])
            per_conv[conv] = per_conv.get(conv, 0) + 1
    print("total verify()-path overrides:", fired)
    print("per conv:", dict(sorted(per_conv.items())))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug()
    elif len(sys.argv) > 1 and sys.argv[1] == "debugq":
        debug_qids()
    elif len(sys.argv) > 1 and sys.argv[1] == "residual":
        debug_residual()
    elif len(sys.argv) > 1 and sys.argv[1] == "prod":
        debug_prod()
    else:
        main()
