"""LoCoMo Selection-Stage Attribution Diagnostic (Phase 1).

Answers one question precisely: **WHERE does the ground-truth turn get lost?**

For every question the ground-truth ``dia_id`` is tracked through the MSC
pipeline stages.  A stage either still contains a gold-bearing record (``O``)
or has already dropped it (``X``):

  CORPUS     ingested ``StructuredIR`` records   (ingestion sanity check)
  WIDE_POOL  WideSlicer multi-channel union      (C_sem U C_lex U C_entity U ...)
  EXP_POOL   WideSlicer + bounded graph expansion
  CAND_PRE   ranked candidate units BEFORE evidence widening
  CAND_POST  ranked candidate units AFTER evidence widening
  CONTEXT    final compiled context (identical to the reported oracle recall)

It also reports the **rank of the first gold-bearing unit** inside the widened
candidate list, which separates the two possible root causes:

  * gold rank deep (>= window)          -> RANKING problem
  * gold rank inside window but dropped -> BUDGET / selection problem

No production code is modified: the compiler is instrumented with
monkeypatches and restored afterwards.

Usage:
  python scripts/diagnose_locomo_selection_stages.py --convs 0,3,5
  python scripts/diagnose_locomo_selection_stages.py --convs 0 --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from artificial_memory.context import msc_compiler as msc_module
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

STAGES = ["CORPUS", "WIDE_POOL", "EXP_POOL", "CAND_PRE", "CAND_POST", "CONTEXT"]
WINDOW_EDGES = [(0, 5), (6, 19), (20, 49), (50, 10**9)]


def _gold_ids(q) -> list[str]:
    """Normalise LoCoMo evidence ids ('D1:3', 'D1:3;D2:7', 'D1:3,D2:7')."""
    out: list[str] = []
    for raw in q.evidence_ids or []:
        for part in str(raw).replace(",", ";").split(";"):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _hits(gold: list[str], blobs) -> bool:
    for g in gold:
        for blob in blobs:
            if blob and g in blob:
                return True
    return False


def _first_rank(gold: list[str], contents: list[str]) -> int | None:
    for idx, blob in enumerate(contents):
        if _hits(gold, [blob]):
            return idx
    return None



class _StageProbe:
    """Monkeypatch harness capturing each pipeline stage for one question."""

    def __init__(self) -> None:
        self.orig_widen = msc_module.MinimumSufficientContextCompiler._apply_evidence_widening

    def __enter__(self) -> "_StageProbe":
        self.state: dict = {}
        probe = self

        def patched_widen(self, query, candidate_units, working_records, graph,
                          reference_date_str=None):
            probe.state["cand_pre"] = [u.ir.raw_content for u in candidate_units]
            try:
                slice_res = self.wide_slicer.slice(
                    query, list(working_records), reference_date_str=reference_date_str
                )
                probe.state["wide_pool"] = [r.raw_content for r in slice_res.candidate_records]
                exp_res = self.graph_expander.expand(
                    query, slice_res.candidate_records, list(working_records), graph=graph
                )
                probe.state["exp_pool"] = [r.raw_content for r in exp_res.evidence_pool]
            except Exception:
                probe.state.setdefault("wide_pool", [])
                probe.state.setdefault("exp_pool", [])
            out = probe.orig_widen(
                self, query, candidate_units, working_records, graph, reference_date_str
            )
            probe.state["cand_post"] = [u.ir.raw_content for u in out]
            return out

        msc_module.MinimumSufficientContextCompiler._apply_evidence_widening = patched_widen
        return self

    def __exit__(self, *exc) -> None:
        msc_module.MinimumSufficientContextCompiler._apply_evidence_widening = self.orig_widen


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "absent"
    for lo, hi in WINDOW_EDGES:
        if lo <= rank <= hi:
            return f"{lo}-{hi if hi < 10**9 else '+'}"
    return "absent"



def run(conv_idx: int, limit: int | None, verbose: bool,
        window_cap: int = 24, token_bonus: int = 130) -> dict:
    adapter = LoCoMoAdapter()
    adapter.compiler = msc_module.MinimumSufficientContextCompiler(
        selection_window_cap=window_cap,
        rescue_token_bonus_per_unit=token_bonus,
    )
    _, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
    if limit is not None:
        questions = questions[:limit]
    weights = EvidenceScoreWeights()
    corpus_blobs = [r.raw_content for r in ir_records]

    per_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, **{s: 0 for s in STAGES}})
    rank_hist: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    reasons_by_cat: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    post_sizes: list[int] = []
    token_costs: list[int] = []

    with _StageProbe() as probe:
        compiler = adapter.compiler
        orig_check = compiler.checker.check

        def traced_check(query, intent, units, *a, **kw):
            probe.state["selected"] = [u.ir.raw_content for u in units]
            return orig_check(query, intent, units, *a, **kw)

        compiler.checker.check = traced_check
        try:
            for q in questions:
                cat = adapter.CATEGORY_NAMES.get(q.category, f"cat-{q.category}")
                gold = _gold_ids(q)
                per_cat[cat]["n"] += 1
                if not gold:
                    # No ground-truth turn: oracle recall counts as satisfied.
                    for stage in STAGES:
                        per_cat[cat][stage] += 1
                    continue

                probe.state = {}
                pcc = adapter.compiler.compile(q.question, ir_records, weights=weights)
                st = probe.state

                present = {
                    "CORPUS": _hits(gold, corpus_blobs),
                    "WIDE_POOL": _hits(gold, st.get("wide_pool", [])),
                    "EXP_POOL": _hits(gold, st.get("exp_pool", [])),
                    "CAND_PRE": _hits(gold, st.get("cand_pre", [])),
                    "CAND_POST": _hits(gold, st.get("cand_post", [])),
                    "CONTEXT": any(g in pcc.context_text for g in gold),
                }
                for stage, ok in present.items():
                    if ok:
                        per_cat[cat][stage] += 1

                post = st.get("cand_post", [])
                sel = st.get("selected", [])
                post_sizes.append(len(post))
                token_costs.append(pcc.token_cost)
                rank = _first_rank(gold, post)
                rank_hist[_rank_bucket(rank)] += 1

                if present["CONTEXT"]:
                    reason = "OK"
                elif _hits(gold, sel):
                    reason = "FORMAT_MISS"
                elif not present["WIDE_POOL"] and not present["CAND_PRE"]:
                    reason = "POOL_MISS"
                elif rank is None:
                    reason = "POOL_DROP"
                elif rank >= 20:
                    reason = "WINDOW_TOO_SMALL"
                elif len(sel) <= rank:
                    reason = "EARLY_STOP"
                else:
                    reason = "SKIPPED_IN_WINDOW"
                reasons[reason] += 1
                reasons_by_cat[cat][reason] += 1

                if verbose and reason != "OK":
                    print(f"  [{reason:<17}] {q.question_id[:26]:<26} cat={cat:<13} "
                          f"rank={rank} sel={len(sel):>3}/{len(post):<4} "
                          f"stages={''.join('O' if present[s] else 'X' for s in STAGES)} "
                          f"| {q.question[:44]}")
        finally:
            compiler.checker.check = orig_check

    return {
        "conv_idx": conv_idx,
        "n": len(questions),
        "per_cat": {k: dict(v) for k, v in per_cat.items()},
        "rank_hist": dict(rank_hist),
        "reasons": dict(reasons),
        "reasons_by_cat": {k: dict(v) for k, v in reasons_by_cat.items()},
        "oracle_share": (
            sum(v["CONTEXT"] for v in per_cat.values()) / len(questions)
            if questions else 0.0
        ),
        "mean_candidates": sum(post_sizes) / len(post_sizes) if post_sizes else 0.0,
        "mean_tokens": sum(token_costs) / len(token_costs) if token_costs else 0.0,
    }


def _print_report(results: list[dict]) -> None:
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, **{s: 0 for s in STAGES}})
    hist: dict[str, int] = defaultdict(int)
    for res in results:
        for cat, vals in res["per_cat"].items():
            tgt = agg[cat]
            tgt["n"] += vals["n"]
            for stage in STAGES:
                tgt[stage] += vals[stage]
        for k, v in res["rank_hist"].items():
            hist[k] += v

    print("\n" + "=" * 96)
    print("STAGE ATTRIBUTION (share of questions where the gold turn is still present)")
    print("=" * 96)
    print(f"{'category':<15}{'n':>6}" + "".join(f"{s:>11}" for s in STAGES))
    grand = {"n": 0, **{s: 0 for s in STAGES}}
    for cat in sorted(agg, key=lambda c: -agg[c]["n"]):
        v = agg[cat]
        cnt = v["n"]
        print(f"{cat:<15}{cnt:>6}" + "".join(f"{v[s] / cnt * 100:>10.1f}%" for s in STAGES))
        grand["n"] += cnt
        for s in STAGES:
            grand[s] += v[s]
    cnt = grand["n"]
    print("-" * 96)
    print(f"{'ALL':<15}{cnt:>6}" + "".join(f"{grand[s] / cnt * 100:>10.1f}%" for s in STAGES))

    print("\n" + "-" * 96)
    print("FUNNEL LOSS (pp lost at each transition)")
    print("-" * 96)
    for a, b in zip(STAGES, STAGES[1:]):
        loss = (grand[a] - grand[b]) / cnt * 100
        flag = "  <== PRIMARY LOSS" if loss >= 8 else ""
        print(f"  {a:<12} -> {b:<12} {loss:>+7.2f}pp{flag}")

    print("\n" + "-" * 96)
    print("ROOT CAUSE OF CONTEXT MISSES")
    print("-" * 96)
    r_agg: dict[str, int] = defaultdict(int)
    for res in results:
        for k, v in res.get("reasons", {}).items():
            r_agg[k] += v
    total_q = sum(r_agg.values())
    for k in sorted(r_agg, key=lambda x: -r_agg[x]):
        c = r_agg[k]
        print(f"  {k:<18}{c:>6}  {c / total_q * 100:>5.1f}%  {'#' * int(c / total_q * 50)}")

    print("\n" + "-" * 96)
    print("RANK OF FIRST GOLD-BEARING CANDIDATE (after widening)")
    print("-" * 96)
    total = sum(hist.values())
    for lo, hi in WINDOW_EDGES:
        key = f"{lo}-{hi if hi < 10**9 else '+'}"
        c = hist.get(key, 0)
        print(f"  rank {key:<10}{c:>6}  {c / total * 100:>5.1f}%  {'#' * int(c / total * 50)}")
    c = hist.get("absent", 0)
    print(f"  {'absent':<15}{c:>6}  {c / total * 100:>5.1f}%  {'#' * int(c / total * 50)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo selection-stage attribution")
    parser.add_argument("--convs", type=str, default="0,3,5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print per-question misses")
    parser.add_argument("--json", type=str, default=None, help="Save raw report to this path")
    parser.add_argument("--window-cap", type=int, default=24,
                        help="Max selected units when widening is active")
    parser.add_argument("--token-bonus", type=int, default=130,
                        help="Extra token budget per promoted rescue unit")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Pareto sweep over window caps, e.g. '8,12,16,24,40'")
    args = parser.parse_args()

    convs = [int(c) for c in args.convs.split(",") if c.strip()]

    if args.sweep:
        caps = [int(c) for c in args.sweep.split(",") if c.strip()]
        print("\n" + "=" * 78)
        print("PARETO SWEEP (retrieval-only: oracle recall vs tokens/context)")
        print("=" * 78)
        print(f"{'window_cap':>11}{'oracle':>10}{'tokens/Q':>11}{'gold@6-19 kept':>17}")
        for cap in caps:
            res = run(convs[0], args.limit, False, window_cap=cap,
                      token_bonus=args.token_bonus)
            print(f"{cap:>11}{res['oracle_share'] * 100:>9.1f}%{res['mean_tokens']:>11.1f}"
                  f"{res['rank_hist'].get('6-19', 0):>17}")
        return

    results = []
    for c in convs:
        print(f"\n### Diagnosing conv {c} ...")
        res = run(c, args.limit, args.verbose, window_cap=args.window_cap,
                  token_bonus=args.token_bonus)
        results.append(res)
        print(f"  conv {c}: n={res['n']} mean_candidates={res['mean_candidates']:.1f} "
              f"mean_tokens={res['mean_tokens']:.1f}")

    _print_report(results)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved: {args.json}")


if __name__ == "__main__":
    main()

