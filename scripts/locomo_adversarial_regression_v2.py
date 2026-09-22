"""Anatomy of the adversarial accuracy regression (subject_binding_v1).

The widened selection window lifted adversarial Oracle Recall by +40pp but
lowered accuracy by -14pp: with the bait turn now inside the context the frozen
model answers it instead of refusing.  This probe isolates what actually
distinguishes a *refused* adversarial question from an *answered* one, using the
saved contexts (no recompilation).
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SBC = Path("benchmark_results/locomo10_runs/subject_binding_v1")
CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
REFUSAL = ("i don't know", "i dont know", "not mentioned", "no information", "unknown",
           "none", "cannot", "unclear", "not specified", "doesn't mention")


def load(root: Path) -> dict:
    out = {}
    for p in sorted(root.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            out[r["question_id"]] = r
    return out


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in REFUSAL)


base = load(Path("benchmark_results/locomo10"))
exp = load(SBC)

adv = [q for q in base if base[q]["category"] == 5]
reg = [q for q in adv if base[q]["is_correct"] and not exp[q]["is_correct"]]
fix = [q for q in adv if not base[q]["is_correct"] and exp[q]["is_correct"]]
print(f"adversarial n={len(adv)} regressions={len(reg)} fixes={len(fix)}")

state = Counter()
for q in adv:
    b_ref, e_ref = is_refusal(base[q]["predicted_answer"]), is_refusal(exp[q]["predicted_answer"])
    state[(b_ref, e_ref, base[q]["is_correct"], exp[q]["is_correct"])] += 1
print("\n(baseline refusal, new refusal, base correct, new correct) -> count")
for k, v in state.most_common():
    print(f"  base_ref={k[0]!s:<5} new_ref={k[1]!s:<5} base_ok={k[2]!s:<5} new_ok={k[3]!s:<5} {v:>4}")

qshape = Counter()
for q in reg:
    qt = base[q]["question"].lower()
    if qt.startswith("did "):
        k = "did_you/they"
    elif qt.startswith(("has ", "have ", "was ", "were ", "is ", "are ", "does ", "do ")):
        k = "aux_yesno"
    elif "how many" in qt or "how much" in qt:
        k = "quantity"
    elif "when" in qt:
        k = "when"
    elif "what" in qt:
        k = "what"
    elif "who" in qt:
        k = "who"
    else:
        k = "other"
    qshape[k] += 1
print("\nregression question shapes:", dict(qshape))

print("\n--- sample regressions (question | baseline ans | new ans) ---")
for q in reg[:12]:
    print(f"Q: {base[q]['question'][:95]}")
    print(f"   base: {base[q]['predicted_answer'][:80]!r} (correct)")
    print(f"   new : {exp[q]['predicted_answer'][:80]!r}")
