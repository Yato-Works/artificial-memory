"""Offline A/B for subject-binding rule variants (R1/R2/R3/R4).

Measurable set
--------------
Stored ``predicted_answer`` values are POST-guard, so a candidate rule can only
be evaluated on rows where the *current* guard is silent — there the stored
answer is the raw model answer and a revised rule would act on it for the first
time.  Gains measured here are therefore **additive on top of** the current
guard's effect.

Variants (each is the exact production class with class attributes patched)
---------------------------------------------------------------------------
R1  any non-owner matched sentence  -> misattributed        (broad skip rule)
R3  non-owner sentence containing a first-person *assertion*
    ("I", "I've", "we", ...)        -> misattributed        (self-assertion rule)
R2  R1 + second-person address ("your puppy") counts as subject-bound
R4  R3 + second-person address counts as subject-bound

Run: python scripts/simulate_binding_variants.py [run_dir]
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
ORDER = ["adversarial", "single-hop", "temporal", "multi-hop", "open-domain"]


class R1(AnswerVerifier):
    _OWNER_PRONOUNS = re.compile(r".")


class R3(AnswerVerifier):
    _OWNER_PRONOUNS = AnswerVerifier._SELF_ASSERTION


class R2(R1):
    _SECOND_PERSON_BINDING = True


class R4(R3):
    _SECOND_PERSON_BINDING = True


VARIANTS = {"R1_broad": R1, "R3_self_assertion": R3,
            "R2_broad+2p": R2, "R4_self+2p": R4}


def load_cache() -> dict[str, dict]:
    out = {}
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["qid"]] = d
    return out


def load_run(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def evaluate(name: str, cls, cache: dict, rows: dict, cur: AnswerVerifier) -> dict:
    verifier = cls()
    stats = Counter()
    for qid, r in rows.items():
        c = cache.get(qid)
        if not c:
            continue
        q, ctx = c["question"], c["context"]
        ans, cat = str(r["predicted_answer"]), int(r["category"])
        cat_name = CAT.get(cat, str(cat))
        if cur.check_subject_binding(q, ans, ctx) is not None:
            stats["guard_skip"] += 1
            continue
        g = verifier.check_subject_binding(q, ans, ctx)
        if g is None:
            continue
        before = score(cat, r["ground_truth"], ans)
        after = score(cat, r["ground_truth"], g.verified_answer)
        stats[f"{cat_name}|fired"] += 1
        if after and not before:
            stats[f"{cat_name}|gain"] += 1
        elif before and not after:
            stats[f"{cat_name}|loss"] += 1
    return stats


def main() -> None:
    cache = load_cache()
    run = Path(sys.argv[1] if len(sys.argv) > 1
               else "benchmark_results/locomo10_runs/subject_binding_v1")
    rows = load_run(run)
    print(f"cache={len(cache)} run={len(rows)} ({run})")
    cur = AnswerVerifier()

    print("\n" + "=" * 78)
    print("SUBJECT-BINDING VARIANT A/B (rows where the current guard is silent)")
    print("=" * 78)
    print(f"{'variant':<20}{'fired':>7}{'gain':>7}{'loss':>7}{'net':>7}   per-category net")
    for name, cls in VARIANTS.items():
        st = evaluate(name, cls, cache, rows, cur)
        fired = sum(v for k, v in st.items() if k.endswith("|fired"))
        gain = sum(v for k, v in st.items() if k.endswith("|gain"))
        loss = sum(v for k, v in st.items() if k.endswith("|loss"))
        per = "  ".join(
            f"{c[:5]}={st.get(f'{c}|gain', 0) - st.get(f'{c}|loss', 0):+d}"
            for c in ORDER if st.get(f"{c}|fired")
        )
        print(f"{name:<20}{fired:>7}{gain:>7}{loss:>7}{gain - loss:>+7}   {per}")


if __name__ == "__main__":
    main()
