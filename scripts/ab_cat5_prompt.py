"""Cached-context prompt A/B for LoCoMo category 5 (adversarial refusal).

Baseline arm reuses the stored production predictions (subject-binding guards
already applied).  The variant arm adds a premise-verification directive to the
prompt: the LLM itself must check whether the question's premise (event happened,
person attribution, entity existence) is confirmed in the context, else refuse.
Production scorers + production AnswerVerifier are applied identically to both
arms.
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
RUN_ROOT = Path("benchmark_results/locomo10_runs/subject_binding_v2")

ADVERSARIAL_DIRECTIVE = (
    "[INSTRUCTION: PREMISE VERIFICATION]\n"
    "Some questions describe events or facts that NEVER happened in the\n"
    "conversation, or attribute to one person something that actually belongs\n"
    "to a DIFFERENT person.  Before answering:\n"
    "1. Find the evidence for the exact premise in the context.\n"
    "2. Check WHO said or did it. If the person named in the question is not\n"
    "   the person the context talks about, the premise is false.\n"
    "3. If the event, object or person in the question does not appear in the\n"
    "   context at all, the premise is false.\n"
    "If the premise is false, reply exactly: I don't know.\n"
    "Only give a real answer when the context explicitly confirms the premise\n"
    "for the exact person the question asks about.\n\n"
    "{context}"
)



def main() -> None:
    rows = [json.loads(line) for line in CACHE.open(encoding="utf-8")]
    rows = [r for r in rows if r["category"] == 5]
    stored: dict[str, dict] = {}
    for p in sorted(RUN_ROOT.glob("conv_*_results.json")):
        for r in json.load(open(p, encoding="utf-8"))["results"]:
            stored[r["question_id"]] = r
    joined = []
    for r in rows:
        s = stored.get(r["qid"])
        if s is None:
            continue
        joined.append({**r, "gt": s["ground_truth"], "stored_pred": s["predicted_answer"]})
    print(f"cat5 questions: {len(joined)}")

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()
    verifier = adapter.compiler.answer_verifier

    def refusal_ok(ans_lower: str, gt_lower: str) -> bool:
        if gt_lower:
            return False
        return any(w in ans_lower for w in
                   ["i don't know", "not mentioned", "unknown", "unclear",
                    "no information", "none", "no"])

    base: dict[str, bool] = {}
    var: dict[str, bool] = {}
    n = len(joined)
    for i, r in enumerate(joined):
        q, ctx = r["question"], r["context"]
        gt_lower = str(r["gt"]).lower().strip()
        base[r["qid"]] = refusal_ok(r["stored_pred"].lower(), gt_lower)
        ans = answerer.answer(q, ADVERSARIAL_DIRECTIVE.format(context=ctx))
        pred = verifier.verify(
            question=q, predicted_answer=ans.text, context=ctx, propositions=[],
            integrity_abstention_recommended="Proposition Integrity Warning" in ctx,
        ).verified_answer
        var[r["qid"]] = refusal_ok(pred.lower(), gt_lower)
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{n}] baseline={sum(base.values()) / len(base) * 100:.1f}% "
                  f"directive={sum(var.values()) / len(var) * 100:.1f}%", flush=True)

    b = sum(base.values())
    v = sum(var.values())
    gain = sum(1 for k in var if var[k] and not base[k])
    loss = sum(1 for k in var if not var[k] and base[k])
    print(f"\ncat5: n={n} baseline {b / n * 100:.1f}% -> directive {v / n * 100:.1f}% "
          f"({(v - b) / n * 100:+.1f}pp, gain {gain} / loss {loss})")

    out = Path("benchmark_results/ab_cat5_prompt.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"baseline": base, "directive": var}, fh, indent=1)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
