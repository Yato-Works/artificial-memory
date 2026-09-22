"""Stage-faithful trace of adversarial regressions through subject binding.

For every adversarial (category 5) question that flipped correct -> incorrect it
re-walks the production guard's decision points in order and reports the first
condition that made it return ``None`` (silent), with concrete examples of the
matched turn/sentence so the missing rule is visible.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
BASE = Path("benchmark_results/locomo10")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "benchmark_results/locomo10_runs/subject_binding_v1")
SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def load_cache() -> dict[str, dict]:
    out = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["qid"]] = d
    return out


def load_run(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            r = dict(r)
            r["ok"] = score(r["category"], r["ground_truth"], r["predicted_answer"])
            out[r["question_id"]] = r
    return out


def trace_one(v: AnswerVerifier, q: str, ctx: str, ans: str, speakers_hint=None):
    """Return (stage, extra, detail) for the first blocking decision point."""
    if v.check_subject_binding(q, ans, ctx) is not None:
        return "FIRES_(stale_run)", "", ""
    turns = v._parse_turns(ctx)
    if len(turns) < 2:
        return "1_TOO_FEW_TURNS", "", ""
    speakers = {s.lower() for s, _ in turns}
    ctx_lower = ctx.lower()
    for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", q)[1:]:
        low = tok.lower()
        if low in speakers or low in v._STOP_WORDS:
            continue
        if low not in ctx_lower:
            return "2_ENTITY_GUARD_ALREADY_REFUSED", "", ""
    subject = None
    for s in sorted(speakers, key=len, reverse=True):
        if re.search(rf"\b{re.escape(s)}'s\b", q.lower()):
            subject = s
            break
    if subject is None:
        mentioned = [s for s in sorted(speakers, key=len, reverse=True)
                     if re.search(rf"\b{re.escape(s)}\b", q.lower())]
        if len(mentioned) >= 2:
            return "3_BOTH_SPEAKERS_IN_QUESTION", "both speakers named", ""
        subject = mentioned[0] if mentioned else None
    if subject is None:
        return "4_NO_SPEAKER_IN_QUESTION", "", ""
    if not any(p.search(q.lower()) for p in v._subject_bound_patterns(subject)):
        return "5_NOT_SUBJECT_BOUND_PATTERN", f"subject={subject}", ""
    if v._DATE_LIKE.search(ans) or v._TEMPORAL_QUESTION.search(q):
        return "6_DATE_OR_TEMPORAL_EXEMPT", f"subject={subject}", ""
    stems = v._content_stems(ans)
    if not stems:
        return "7_NO_ANSWER_STEMS", f"subject={subject}", ""
    scored = sorted(
        ((sum(1 for st in stems if st in v._content_stems(t)) / len(stems), s, t.lower())
         for s, t in turns), key=lambda x: -x[0])
    matched = [(r, s, t) for r, s, t in scored if r >= 0.5]
    if not matched:
        top = scored[0] if scored else (0.0, "", "")
        return "8_NO_MATCHED_TURN", f"subject={subject} top_ratio={top[0]:.2f}", top[2][:110]
    detail, verdict = [], "UNBOUND_UNATTRIBUTED"
    for _, spk, txt in matched:
        own = spk == subject
        for m in re.finditer(r"[^\n.!?]+", txt):
            sent = m.group(0).strip()
            if not any(st in v._content_stems(sent) for st in stems):
                continue
            if sent.rstrip().endswith("?"):
                continue
            if own:
                verdict = "OWN_TURN_ATTR_MISMATCH"
            detail.append(f"{spk}: {sent[:110]}")
            break
    return f"9_{verdict}", f"subject={subject}", " || ".join(detail)


def main() -> None:
    cache = load_cache()
    base = load_run(BASE)
    run = load_run(RUN)
    v = AnswerVerifier(subject_binding=True)

    stages = Counter()
    examples: dict[str, list] = defaultdict(list)
    n_reg = 0
    for qid, r in run.items():
        if r.get("category") != 5:
            continue
        b = base.get(qid)
        if b is None:
            continue
        ans = str(r.get("predicted_answer", ""))
        if r["ok"]:
            continue
        if not b["ok"]:
            continue
        n_reg += 1
        row = cache.get(qid)
        if row is None:
            stages["0_NO_CACHED_CONTEXT"] += 1
            continue
        q, ctx = row["question"], row.get("context", "")
        stage, extra, detail = trace_one(v, q, ctx, ans)
        stages[stage] += 1
        if len(examples[stage]) < SHOW:
            examples[stage].append((qid, extra, detail, ans[:70], q[:80]))

    print(f"adversarial regressions traced: {n_reg} (run={RUN.name})")
    print("\nFIRST BLOCKING STAGE (why the guard stayed silent):")
    for k, c in stages.most_common():
        print(f"  {k:<32}{c:>5}  {c / max(n_reg, 1) * 100:>5.1f}%")

    for k, c in stages.most_common():
        if k.startswith("9_") or k in ("0_NO_CACHED_CONTEXT",):
            continue
        print(f"\n--- examples: {k} ---")
        for qid, extra, detail, ans, q in examples[k]:
            print(f"  {qid} {extra}\n    q  = {q}\n    ans= {ans!r}\n    ev = {detail}")


if __name__ == "__main__":
    main()

