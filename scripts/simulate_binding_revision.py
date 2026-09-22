"""Offline A/B for the subject-binding rule revision ("rule R").

Measurable set
--------------
Stored ``predicted_answer`` values are POST-guard, so a candidate rule can only
be evaluated on rows where the *current* guard does not fire — there the stored
answer is the raw model answer and a revised rule would act on it for the first
time.

Rule R (candidate)
------------------
Today ``check_subject_binding`` marks a non-owner matched sentence as
misattributed only when it carries a first-person possessive ("my/our").  That
misses the dominant adversarial bait form, where the other speaker asserts the
same content without a possessive:

    Q: What was the poetry reading that **Melanie** attended about?
    Caroline: 'It was a transgender poetry reading ...'        <- no "my"

Rule R treats every matched sentence of a NON-owner speaker as misattributed
unless it explicitly binds the content to the queried subject (bind_b:
non-vocative subject mention, or "subject's" in the turn).  Because ``bound``
short-circuits the refusal whenever ANY matched turn is the subject's own, this
only fires when the high-overlap evidence is exclusively someone else's.

Implementation of the variant is a one-attribute patch of the production class
(``_OWNER_PRONOUNS`` -> always true), so the measured rule is exactly the rule
that would ship.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from artificial_memory.recall.answer_verifier import AnswerVerifier

sys.path.insert(0, str(Path(__file__).parent))
from rescore_locomo_run import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


class RevisedVerifier(AnswerVerifier):
    """Rule R: any non-owner match without subject binding is misattributed."""

    _OWNER_PRONOUNS = re.compile(r".")


def load_cache() -> dict[str, dict]:
    out = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["qid"]] = d
    return out


def load_run(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def main() -> None:
    cache = load_cache()
    run = Path(sys.argv[1] if len(sys.argv) > 1
               else "benchmark_results/locomo10_runs/subject_binding_v1")
    rows = load_run(run)
    base_rows = load_run(Path("benchmark_results/locomo10"))
    print(f"cache={len(cache)} run={len(rows)} ({run}) baseline={len(base_rows)}")

    cur, rev = AnswerVerifier(), RevisedVerifier()
    stats = Counter()
    flips: list[tuple] = []

    for qid, r in rows.items():
        c = cache.get(qid)
        if not c:
            stats["no_context"] += 1
            continue
        q, ctx = c["question"], c["context"]
        ans, cat = str(r["predicted_answer"]), int(r["category"])
        cat_name = CAT.get(cat, str(cat))
        before_ok = score(cat, r["ground_truth"], ans)

        g_cur = cur.check_subject_binding(q, ans, ctx)
        g_rev = rev.check_subject_binding(q, ans, ctx)
        if g_cur is not None:
            stats["guard_already_fires_skip"] += 1
            stats[f"skip_{cat_name}"] += 1
            # Already refused by the current rule; rule R cannot add anything.
            continue
        if g_rev is None:
            stats["rule_R_inactive"] += 1
            continue

        new_ans = g_rev.verified_answer
        after_ok = score(cat, r["ground_truth"], new_ans)
        key = (cat_name, before_ok, after_ok)
        stats[str(key)] += 1
        if before_ok != after_ok:
            flips.append((qid, cat_name, before_ok, after_ok, ans[:44], new_ans[:30], q[:58]))

    # ---------- report ----------
    print("\n" + "=" * 92)
    print("RULE R OFFLINE A/B (only rows where the current guard is silent)")
    print("=" * 92)
    print(f"  current guard already fires   : {stats['guard_already_fires_skip']}")
    print(f"  rule R silent                 : {stats['rule_R_inactive']}")
    print("\n  rule-R candidates by category and outcome "
          "(before_ok -> after_ok):")
    gained = lost = 0
    for cat in ["adversarial", "single-hop", "temporal", "multi-hop", "open-domain"]:
        rows_cat = [(k, v) for k, v in stats.items()
                    if k.startswith(f"('{cat}'")]
        if not rows_cat:
            continue
        g = sum(v for k, v in rows_cat if k.endswith("False, True)"))
        l = sum(v for k, v in rows_cat if k.endswith("True, False)"))
        fired = sum(v for _, v in rows_cat)
        gained += g
        lost += l
        print(f"    {cat:<13} fired={fired:<5} gain={g:<5} loss={l:<5} net={g - l:+d}")
    print(f"\n  NET accuracy change: {gained - lost:+d} questions "
          f"(gain {gained} / loss {lost})")

    print("\n  sample flips (first 25):")
    for qid, cat_name, b, a, old_ans, new_ans, q in flips[:25]:
        mark = "+" if a and not b else "-"
        print(f"   [{mark}] {qid:<20} {cat_name:<12} {old_ans!r} -> {new_ans!r}")
        print(f"       q={q}")


if __name__ == "__main__":
    main()
