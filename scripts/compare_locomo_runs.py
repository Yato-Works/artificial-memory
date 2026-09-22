"""LoCoMo Run Comparator (Phase 0: Diff Harness).

Diffs two LoCoMo result directories (same schema as
``run_locomo_full_suite.py`` / ``run_locomo_smoke.py``) at question_id level:

  * FIXED   : fail -> pass (LLM mode) / oracle miss -> hit (retrieval mode)
  * REGRESS : pass -> fail / oracle hit -> miss
  * Oracle gains / losses per category
  * Failure taxonomy deltas (re-classified with FailureClassifierV2)
  * Token / latency Pareto deltas

Usage:
  python scripts/compare_locomo_runs.py --baseline benchmark_results/locomo10 \
      --experiment benchmark_results/locomo10_runs/phase1_wideslicer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from artificial_memory.research.benchmarks.failure_taxonomy_v2 import FailureClassifierV2

sys.stdout.reconfigure(encoding="utf-8")

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def _question_text_map() -> dict[str, str]:
    """question_id -> question text, from the frozen dataset (for taxonomy)."""
    dataset = Path("datasets/external/locomo10.json")
    if not dataset.exists():
        return {}
    with open(dataset, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, str] = {}
    for conv_idx, conv in enumerate(raw):
        sample_id = conv.get("sample_id", f"conv-{conv_idx}")
        for i, qa in enumerate(conv.get("qa", [])):
            out[f"{sample_id}-qa-{i:03d}"] = qa.get("question", "")
    return out


def load_run(run_dir: Path) -> dict[str, dict]:
    """Load all conv_*.json files -> {question_id: result_dict}."""
    q_text = _question_text_map()
    out: dict[str, dict] = {}
    for f in sorted(run_dir.glob("conv_*_results.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data["results"]:
            r = dict(r)
            r["conv_idx"] = data["summary"]["conv_idx"]
            r.setdefault("question", q_text.get(r["question_id"], ""))
            out[r["question_id"]] = r
    return out


def classify_failure(r: dict) -> str | None:
    """Re-classify a failure with Taxonomy v2 (needs predicted answer)."""
    if not r.get("predicted_answer"):
        return None
    if r.get("is_correct"):
        return None
    diag = FailureClassifierV2().classify(
        question=r.get("question", ""),
        ground_truth=str(r.get("ground_truth", "")),
        predicted_answer=r.get("predicted_answer", ""),
        context="",
        oracle_recall=r.get("oracle_recall", False),
        is_correct=bool(r.get("is_correct")),
        question_type=CATEGORY_NAMES.get(r.get("category"), "general"),
    )
    return diag.category.value if diag else None


def has_llm_answers(results: dict[str, dict]) -> bool:
    return any(bool(r.get("predicted_answer")) for r in results.values())


def agg(results: list[dict]) -> tuple[float, float, float, float]:
    n = len(results)
    ora = sum(1 for r in results if r["oracle_recall"]) / n if n else 0.0
    cor = sum(1 for r in results if r.get("is_correct")) / n if n else 0.0
    tok = sum(r.get("tokens_used", 0) for r in results) / n if n else 0.0
    lat = sum(r.get("latency_ms", 0) for r in results) / n if n else 0.0
    return cor, ora, tok, lat


def show_samples(base: dict, exp: dict, qids: list[str], label: str, limit: int) -> None:


    n = len(results)
    ora = sum(1 for r in results if r["oracle_recall"]) / n if n else 0.0
    cor = sum(1 for r in results if r.get("is_correct")) / n if n else 0.0
    tok = sum(r.get("tokens_used", 0) for r in results) / n if n else 0.0
    lat = sum(r.get("latency_ms", 0) for r in results) / n if n else 0.0
    return cor, ora, tok, lat


def show_samples(base: dict, exp: dict, qids: list[str], label: str, limit: int) -> None:
    print(f"\n--- {label} (up to {limit}) ---")
    for q in qids[:limit]:
        b, e = base[q], exp[q]
        print(f"  [{CATEGORY_NAMES.get(b['category'], b['category'])}] {q}")
        print(f"    Q  : {b.get('question', '')[:90]}")
        print(f"    GT : {str(b.get('ground_truth', ''))[:60]}")
        print(f"    old: {str(b.get('predicted_answer', ''))[:60]!r} (ora={b['oracle_recall']})")
        print(f"    new: {str(e.get('predicted_answer', ''))[:60]!r} (ora={e['oracle_recall']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two LoCoMo runs question-by-question")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args()

    base = load_run(args.baseline)
    exp = load_run(args.experiment)
    common = sorted(set(base) & set(exp))
    only_base = set(base) - set(exp)
    only_exp = set(exp) - set(base)
    llm_mode = has_llm_answers(exp) and all("is_correct" in r for r in exp.values())

    print("=" * 84)
    print("                    LOCOMO RUN COMPARISON (question-level diff)")
    print("=" * 84)
    print(f"Baseline   : {args.baseline}  ({len(base)} questions)")
    print(f"Experiment : {args.experiment}  ({len(exp)} questions)")
    print(f"Common     : {len(common)} | baseline-only: {len(only_base)} | experiment-only: {len(only_exp)}")
    print(f"Metric     : {'Accuracy + Oracle' if llm_mode else 'Oracle Recall (retrieval-only run)'}")
    print("=" * 84)

    b_cor, b_ora, b_tok, b_lat = agg([base[q] for q in common])
    e_cor, e_ora, e_tok, e_lat = agg([exp[q] for q in common])
    print(f"\n{'':16}{'Baseline':>12}{'Experiment':>12}{'+Delta':>12}")
    print(f"{'Accuracy':<16}{b_cor*100:>11.2f}%{e_cor*100:>11.2f}%{(e_cor-b_cor)*100:>+11.2f}pp")
    print(f"{'Oracle Recall':<16}{b_ora*100:>11.2f}%{e_ora*100:>11.2f}%{(e_ora-b_ora)*100:>+11.2f}pp")
    print(f"{'Tokens/Q':<16}{b_tok:>12.1f}{e_tok:>12.1f}{e_tok-b_tok:>+12.1f}")
    if llm_mode:
        print(f"{'Latency ms/Q':<16}{b_lat:>12.1f}{e_lat:>12.1f}{e_lat-b_lat:>+12.1f}")

    if llm_mode:
        fixed = [q for q in common if not base[q]["is_correct"] and exp[q]["is_correct"]]
        regress = [q for q in common if base[q]["is_correct"] and not exp[q]["is_correct"]]
        print(f"\nFIXED   (fail -> pass): {len(fixed)}")
        print(f"REGRESS (pass -> fail): {len(regress)}")
        print(f"Net accuracy change   : {len(fixed) - len(regress):+d} questions")

    o_gain = [q for q in common if not base[q]["oracle_recall"] and exp[q]["oracle_recall"]]
    o_loss = [q for q in common if base[q]["oracle_recall"] and not exp[q]["oracle_recall"]]
    print(f"\nOracle gained (X -> O): {len(o_gain)}")
    print(f"Oracle lost   (O -> X): {len(o_loss)}")
    print(f"Net oracle change     : {len(o_gain) - len(o_loss):+d} questions")

    print("\n" + "-" * 84)
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'category':<14}{'n':>6}{'base acc':>10}{'exp acc':>10}{'delta':>9}"
          f"{'base ora':>10}{'exp ora':>10}{'delta':>9}")
    for c in sorted({base[q]["category"] for q in common}):
        qs = [q for q in common if base[q]["category"] == c]
        n = len(qs)
        bo = sum(1 for q in qs if base[q]["oracle_recall"]) / n * 100
        eo = sum(1 for q in qs if exp[q]["oracle_recall"]) / n * 100
        name = CATEGORY_NAMES.get(c, f"cat-{c}")
        if llm_mode:
            bc = sum(1 for q in qs if base[q].get("is_correct")) / n * 100
            ec = sum(1 for q in qs if exp[q].get("is_correct")) / n * 100
            print(f"{name:<14}{n:>6}{bc:>9.1f}%{ec:>9.1f}%{ec-bc:>+8.1f}p"
                  f"{bo:>9.1f}%{eo:>9.1f}%{eo-bo:>+8.1f}p")
        else:
            print(f"{name:<14}{n:>6}{'—':>10}{'—':>10}{'—':>9}"
                  f"{bo:>9.1f}%{eo:>9.1f}%{eo-bo:>+8.1f}p")

    if llm_mode:
        print("\n" + "-" * 84)
        print("FAILURE TAXONOMY v2 DELTA")
        taxo_b: dict[str, int] = {}
        taxo_e: dict[str, int] = {}
        for q in common:
            cb = classify_failure(base[q])
            if cb:
                taxo_b[cb] = taxo_b.get(cb, 0) + 1
            ce = classify_failure(exp[q])
            if ce:
                taxo_e[ce] = taxo_e.get(ce, 0) + 1
        keys = sorted(set(taxo_b) | set(taxo_e), key=lambda k: -taxo_e.get(k, 0))
        print(f"{'category':<24}{'baseline':>10}{'experiment':>12}{'delta':>8}")
        for k in keys:
            print(f"{k:<24}{taxo_b.get(k, 0):>10}{taxo_e.get(k, 0):>12}"
                  f"{taxo_e.get(k, 0) - taxo_b.get(k, 0):>+8}")

        if fixed:
            show_samples(base, exp, fixed, "SAMPLE FIXES", args.max_samples)
        if regress:
            show_samples(base, exp, regress, "SAMPLE REGRESSIONS (inspect first!)", args.max_samples)
    elif o_gain:
        show_samples(base, exp, o_gain, "SAMPLE ORACLE GAINS", args.max_samples)
    if o_loss:
        show_samples(base, exp, o_loss, "SAMPLE ORACLE LOSSES (inspect first!)", args.max_samples)

    print("\n" + "=" * 84)


if __name__ == "__main__":
    main()

