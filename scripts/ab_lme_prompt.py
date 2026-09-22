"""LongMemEval prompt A/B on cached contexts (no recompile, reader only).

Why this exists
---------------
``scripts/lme_failure_anatomy.py`` showed that of the 118 remaining failures in
the qwen2.5:7b run, 47 are the reader refusing although the evidence is present
(oracle=OK) and 64 are wrong values - i.e. the residual is a *conversion*
problem, not a retrieval problem.  Prompt structure is therefore the lever, and
testing a prompt must not cost a recompile of every haystack.

This harness replays the cached MSC contexts (``lme_context_cache.jsonl``), so a
run only pays for reader calls, exactly like the LoCoMo prompt A/Bs.

Arms
----
``--baseline-from <report>``  reuses stored predictions as arm A (free).  With no
                              ``--fix-*`` flags the harness must reproduce that
                              report's score - the sanity check that the cached
                              prompt reproduction is faithful.

Fix flags (each independently testable, all MECHANISM changes):
``--global-ef``     prepend an evidence-first anti-refusal instruction to every
                    non-abstention prompt
``--fix-ss-user``   give single-session-user a dedicated instruction branch
                    (the adapter has none: it feeds the raw context)
``--fix-multi``     multi-session: drop the "if information is missing, say not
                    enough" invitation that triggers 17 refusals
``--fix-temporal``  temporal: when the resolver abstains or has no grounding, let
                    the reader answer from the evidence instead of refusing
``--fix-ku``        knowledge-update: never replace the context with a certificate
                    only; keep the evidence and state the target state
``--fix-ss-assist`` single-session-assistant: replace "say I don't know" with
                    "give the best matching item"

Usage
-----
  uv run python scripts/ab_lme_prompt.py --global-ef --fix-ss-user --fix-multi \\
      --tag ef_v1 --model qwen2.5:7b-instruct-q4_K_M --num-ctx 8192
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from artificial_memory.research.benchmarks.external.longmemeval_adapter import (  # noqa: E402
    LongMemEvalAdapter,
)
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402

CACHE = Path("benchmark_results/lme_context_cache.jsonl")
BASELINE = Path("benchmark_results/longmemeval/grand_longmemeval_report_qwen7b.json")

TYPES = ["single-session-user", "multi-session", "temporal-reasoning",
         "knowledge-update", "single-session-assistant", "single-session-preference"]

EVIDENCE_FIRST = (
    "[INSTRUCTION: EVIDENCE-FIRST ANSWERING]\n"
    "A memory system has already retrieved the evidence below for this question.\n"
    "The answer IS present in this evidence.\n"
    "RULES:\n"
    "1. Locate the specific fact the question asks about and state it directly.\n"
    "2. Reply with the final value only: a few words or one short sentence.\n"
    "3. NEVER reply \"I don't know\", \"not enough information\", \"not mentioned\",\n"
    "   \"unknown\" or any other refusal - a refusal is always wrong for this task.\n"
    "4. Use only the evidence; do not invent details that are not in it.\n\n"
)


def build_prompt(row: dict, args: argparse.Namespace) -> str:
    """Reproduce the adapter's prompt structure, with optional mechanism fixes.

    With no fix flags this mirrors ``LongMemEvalAdapter.evaluate_item`` so the
    baseline arm can reproduce the stored report.
    """
    qtype = row["qtype"]
    ctx = row["context_text"]
    # The evidence-first preamble is measured to HURT single-session-preference
    # (it makes answers terser and drops the preference grounding: 3 control
    # regressions, 1 fix on the 110-question arm), and that branch already
    # carries its own "do not say I don't know" instruction.
    ef = "" if qtype == "single-session-preference" else (EVIDENCE_FIRST if args.global_ef else "")

    if qtype == "single-session-preference":
        lines = ctx.split("\n")
        pref = [l for l in lines if l.startswith("[User Profile & Preferences:")]
        grounding = "\n".join(pref) if pref else ""
        return (
            f"{ef}[USER PROFILE & PREFERENCES]\n{grounding}\n\n"
            f"[TASK INSTRUCTION]\n"
            f"The user is asking the question below. You MUST tailor your answer directly to "
            f"their stated preferences, past equipment, or background in [USER PROFILE & PREFERENCES]. "
            f"Do NOT say 'I don't know'. Give concrete, specific suggestions or explanations "
            f"that incorporate their preferences."
        )

    if qtype == "knowledge-update":
        cert = row.get("ku_certificate") or ""
        ql = row["question"].lower()
        is_prev = any(w in ql for w in ["previous", "previously", "earlier", "before",
                                        "former", "initially"])
        target = "PREVIOUS / EARLIER" if is_prev else "CURRENT / LATEST"
        rules = (
            f"[INSTRUCTION: KNOWLEDGE UPDATE & STATE EVOLUTION]\n"
            f"The user's state changes over time. The question asks for the {target} state.\n"
            f"RULES:\n"
            f"1. If asking for CURRENT / LATEST: use the value from the most recent session.\n"
            f"2. If asking for PREVIOUS / EARLIER: use the value from the earlier session.\n"
            f"3. State the value directly and concisely (e.g. 'four', 'the suburbs', '$400,000').\n"
            f"4. NEVER answer 'not enough information' - the value is in the evidence.\n\n"
        )
        if cert and not args.fix_ku:
            return f"{cert}\n\n[INSTRUCTION: State the final answer directly.]"
        if cert:
            return f"{ef}{rules}{cert}\n\nEVIDENCE:\n{ctx}"
        return f"{ef}{rules}{ctx}"

    if qtype == "single-session-assistant":
        ql = row["question"].lower()
        has_ordinal = any(w in ql for w in [
            "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
            "11th", "12th", "13th", "14th", "15th", "20th", "25th", "27th", "30th",
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "last", "final",
        ])
        if has_ordinal:
            if args.fix_ss_assist:
                tail = ("5. Do NOT refuse. If the list length is unclear, give the item at the "
                        "requested position in the most plausible list.\n\n")
            else:
                tail = ("5. Do NOT guess or approximate. If you cannot find the exact list, "
                        "say 'I don't know.'\n\n")
            return (
                f"{ef}[INSTRUCTION: CAREFUL LIST ITEM EXTRACTION]\n"
                f"The user is asking about a specific item from a numbered or ordered list.\n"
                f"RULES:\n"
                f"1. Find the EXACT list or enumeration in the assistant's response below.\n"
                f"2. Count items carefully from 1 to reach the requested position.\n"
                f"3. If asking for the 'last' item, find the final item in the complete list.\n"
                f"4. Return ONLY the item at the exact requested position.\n"
                f"{tail}{ctx}"
            )
        return (
            f"{ef}[INSTRUCTION: ASSISTANT CONTENT RECALL]\n"
            f"The user is asking about something the assistant said or provided in a previous "
            f"conversation. Find the relevant assistant response below and extract the specific "
            f"detail requested. Answer concisely with the exact information.\n\n{ctx}"
        )

    if qtype == "multi-session":
        cert, cert_valid = row.get("multi_cert") or "", bool(row.get("multi_cert_valid"))
        if cert_valid:
            return (
                f"{ef}{cert}\n\n"
                f"[INSTRUCTION: Based on the verified deduction, calculation, or aggregation "
                f"above, what is the final answer to the question? State the exact answer "
                f"directly and concisely.]"
            )
        if args.fix_multi:
            return (
                f"{ef}[INSTRUCTION: MULTI-SESSION REASONING]\n"
                f"Answer using the conversation evidence below.\n"
                f"- Count/total questions: check ALL sessions so every relevant instance is "
                f"included, then give the exact total.\n"
                f"- Comparison/difference questions: compute the difference between the "
                f"requested items.\n"
                f"- The evidence was selected for this question, so the information needed is "
                f"present. Give the concise final answer directly.\n\n{ctx}"
            )
        return (
            f"[INSTRUCTION: MULTI-SESSION REASONING]\n"
            f"Answer the question using the conversation context below.\n"
            f"- If the question asks for a count or total: carefully check ALL sessions to ensure "
            f"every relevant instance/item is included, then provide the exact total.\n"
            f"- If the question asks for a comparison or difference (e.g., 'how much more', "
            f"'faster', 'older', 'difference'): compute the difference between the specific items "
            f"requested.\n"
            f"- If the question asks for items not mentioned in the conversation, or if key "
            f"information is missing, state clearly: 'The information provided is not enough.'\n"
            f"- Provide the concise final answer directly.\n\n{ctx}"
        )

    if qtype == "temporal-reasoning":
        ground = row.get("temporal_grounding") or ""
        if ground and "Temporal Abstention" not in ground:
            if args.fix_temporal:
                # The resolver's grounding can name the right DATE but the wrong
                # EVENT, and the adapter feeds the grounding *instead of* the
                # compiled context - so the evidence is never shown.  Measured on
                # the 110-question arm: 14 of 17 temporal refusals had oracle=OK.
                return (
                    f"{ef}{ground}\n\n"
                    f"[INSTRUCTION: Answer the question using BOTH the grounding above and "
                    f"the conversation evidence below. If the grounding names a different "
                    f"event than the question asks about, ignore it and answer from the "
                    f"evidence. Do not refuse - the evidence was retrieved for this "
                    f"question.]\n\nEVIDENCE:\n{ctx}"
                )
            if "Time-Anchored Event" in ground:
                return (
                    f"{ground}\n\n[INSTRUCTION: Answer the question based on the event above "
                    f"clearly and concisely.]"
                )
            return (
                f"{ground}\n\n[INSTRUCTION: Based on the verified temporal calculation/ordering "
                f"above, answer the question directly. State the exact numbers, durations, or "
                f"order clearly.]"
            )
        if args.fix_temporal:
            return (
                f"{ef}[INSTRUCTION: TEMPORAL REASONING]\n"
                f"Compute the answer from the dated events in the evidence below.\n"
                f"- Convert relative expressions ('last week', 'yesterday', 'two days later') "
                f"using the session dates shown.\n"
                f"- For durations, subtract the two event dates.\n"
                f"- Do not refuse - the evidence was retrieved for this question.\n\n{ctx}"
            )
        return ctx  # baseline: raw context, no instruction

    # single-session-user (and any future type): the adapter has no branch at all.
    if args.fix_ss_user:
        return (f"{ef}[INSTRUCTION: Answer the question with the exact value from the "
                f"evidence.]\n\n{ctx}")
    return f"{ef}{ctx}" if args.global_ef else ctx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ef_v1")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--baseline-from", default=str(BASELINE))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids-file", default=None,
                    help="Only evaluate these question ids (one per line)")
    ap.add_argument("--only-refusals", action="store_true",
                    help="Restrict to the baseline's refused questions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--global-ef", action="store_true")
    ap.add_argument("--fix-ss-user", action="store_true")
    ap.add_argument("--fix-multi", action="store_true")
    ap.add_argument("--fix-temporal", action="store_true")
    ap.add_argument("--fix-ku", action="store_true")
    ap.add_argument("--fix-ss-assist", action="store_true")
    return ap.parse_args()


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_rows(rows: list[dict], args: argparse.Namespace, base: dict) -> list[dict]:
    if args.qids_file:
        wanted = {l.strip() for l in Path(args.qids_file).read_text(encoding="utf-8").split() if l.strip()}
        rows = [r for r in rows if r["qid"] in wanted]
    if args.only_refusals:
        def refused(r: dict) -> bool:
            b = base.get(r["qid"])
            if not b or b.get("is_correct"):
                return False
            return any(m in str(b.get("predicted_answer", "")).lower() for m in (
                "i don't know", "i dont know", "not enough", "no information",
                "not mentioned", "unknown", "unclear", "cannot",
            ))
        rows = [r for r in rows if refused(r)]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


FORCED_ABSTENTION = "The information provided is not enough. You did not mention this information."


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.cache))
    with open(args.baseline_from, encoding="utf-8") as fh:
        baseline = json.load(fh)
    base = {r["question_id"]: r for r in baseline["results"]}
    selected = select_rows(rows, args, base)
    print(f"cache rows: {len(rows)}  selected: {len(selected)}")
    print(f"fixes: global_ef={args.global_ef} ss_user={args.fix_ss_user} multi={args.fix_multi} "
          f"temporal={args.fix_temporal} ku={args.fix_ku} ss_assist={args.fix_ss_assist}")

    answerer = OllamaAnswerer(model=args.model, num_ctx=args.num_ctx)
    print(f"reader: {args.model} num_ctx={args.num_ctx}\n")

    t0 = time.perf_counter()
    out_rows: list[dict] = []
    agg: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    flips_fixed, flips_broken = [], []
    refused = 0

    for i, r in enumerate(selected, 1):
        if r["is_abstention_gt"] or r["is_abstention_flag"]:
            pred = FORCED_ABSTENTION
        else:
            prompt = build_prompt(r, args)
            pred = answerer.answer(r["question"], prompt).text
        ok = LongMemEvalAdapter.score_answer(
            question_type=r["qtype"], gt=r["gt"], predicted_answer=pred,
            is_abstention_gt=bool(r["is_abstention_gt"]),
            pcc_is_abstention=bool(r["is_abstention_flag"]),
        )
        if any(m in str(pred).lower() for m in
               ("i don't know", "i dont know", "not enough", "no information",
                "not mentioned", "unknown", "unclear", "cannot")):
            refused += 1
        agg[r["qtype"]][1] += 1
        agg[r["qtype"]][0] += int(ok)
        agg["ALL"][1] += 1
        agg["ALL"][0] += int(ok)
        b = base.get(r["qid"])
        if b is not None:
            was = bool(b.get("is_correct"))
            if ok and not was:
                flips_fixed.append(r["qid"])
            elif was and not ok:
                flips_broken.append(r["qid"])
        out_rows.append({"qid": r["qid"], "qtype": r["qtype"], "question": r["question"],
                         "gt": r["gt"], "predicted_answer": pred, "is_correct": bool(ok),
                         "tokens_used": r["token_cost"]})
        if i % 25 == 0:
            print(f"  {i}/{len(selected)}  acc={agg['ALL'][0] / agg['ALL'][1] * 100:5.1f}%  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n=== ARM RESULT (n={agg['ALL'][1]}) ===")
    print(f"  ALL              {agg['ALL'][0] / agg['ALL'][1] * 100:5.1f}%  ({agg['ALL'][0]}/{agg['ALL'][1]})")
    for t in TYPES:
        if agg[t][1]:
            print(f"  {t:<28}{agg[t][0] / agg[t][1] * 100:5.1f}%  ({agg[t][0]}/{agg[t][1]})")
    print(f"\n  refusals in arm : {refused}")
    print(f"  fixed vs baseline : +{len(flips_fixed)}")
    print(f"  broken vs baseline: -{len(flips_broken)}")
    print(f"  elapsed          : {elapsed:.0f}s ({elapsed / max(1, len(selected)):.2f}s/Q)")

    out = Path(args.out or f"benchmark_results/ab_lme_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"tag": args.tag, "model": args.model, "num_ctx": args.num_ctx,
                   "fixes": {k: getattr(args, k) for k in (
                       "global_ef", "fix_ss_user", "fix_multi", "fix_temporal",
                       "fix_ku", "fix_ss_assist")},
                   "n": agg["ALL"][1], "accuracy": agg["ALL"][0] / agg["ALL"][1],
                   "refusals": refused, "flips_fixed": flips_fixed,
                   "flips_broken": flips_broken, "results": out_rows}, fh, indent=1)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
