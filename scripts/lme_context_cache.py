"""Cache the compiled LongMemEval contexts so prompt/reader A-B tests are cheap.

Compiling one LongMemEval context (ingest the haystack, run the IR extractor,
compile the Minimum Sufficient Context, then the session fuser / temporal
resolver / timeline certificates) costs seconds of CPU per question.  Prompt,
guard and reader experiments do not change any of that, so paying it on every
iteration would make the failure loop unusable.

This script pays the cost ONCE and writes one JSONL row per question holding
everything a prompt variant needs to be evaluated without recompiling:

    qid, qtype, question, question_date, gt, answer_session_ids,
    is_abstention_gt, oracle_recall, token_cost, is_abstention_flag,
    context_text,                    # raw MSC context
    multi_cert,                      # SessionFuser certificate text (or "")
    multi_cert_valid,                # the adapter's cert_is_valid decision
    temporal_grounding,              # TemporalResolver grounding text (or "")
    ku_certificate                   # StateTimelineEngine certificate text (or "")

A later prompt A/B therefore only needs the reader model, exactly like the
LoCoMo harness uses benchmark_results/locomo_context_cache.jsonl.

Usage:
  # One-time (CPU only; safe to run while a GPU benchmark is in flight)
  uv run python scripts/lme_context_cache.py --out benchmark_results/lme_context_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from artificial_memory.research.benchmarks.external.longmemeval_adapter import (  # noqa: E402
    LongMemEvalAdapter,
)


def _ingest(adapter: LongMemEvalAdapter, item):
    """Exactly the adapter's ingestion path: sessions -> StructuredIR records."""
    all_records = []
    for s_idx, session in enumerate(item.haystack_sessions):
        s_date = item.haystack_dates[s_idx] if s_idx < len(item.haystack_dates) else ""
        sid = item.haystack_session_ids[s_idx] if s_idx < len(item.haystack_session_ids) else ""
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            for r in adapter.extractor.extract(content, default_source=speaker):
                r.raw_content = (
                    f"[{sid} on {s_date}] {speaker}: {content}" if s_date
                    else f"[{sid}] {speaker}: {content}"
                )
                r.time_scope = s_date
                all_records.append(r)
    return all_records


def _derived(adapter: LongMemEvalAdapter, item, pcc, all_records):
    """The session-fuser / temporal / timeline artifacts each prompt branch uses."""
    multi_cert, multi_cert_valid = "", False
    fuser = getattr(adapter.compiler, "session_fuser", None)
    if fuser is not None:
        try:
            agg = fuser.fuse(item.question, pcc.context_text)
            multi_cert = agg.certificate or ""
            multi_cert_valid = bool(
                agg.certificate
                and (agg.found_snippets or agg.total_value is not None
                     or agg.is_aggregation_query)
            )
        except Exception as exc:  # pragma: no cover - defensive
            multi_cert = f"<fuser error: {type(exc).__name__}: {exc}>"

    temporal_grounding = ""
    tres = getattr(adapter.compiler, "temporal_resolver", None)
    if tres is not None:
        try:
            t = tres.resolve(item.question, all_records,
                             reference_date_str=item.question_date)
            if t is not None:
                temporal_grounding = t.grounding_text or ""
        except Exception as exc:  # pragma: no cover - defensive
            temporal_grounding = f"<temporal error: {type(exc).__name__}: {exc}>"

    ku_certificate = ""
    engine = (getattr(adapter.compiler, "state_timeline_engine", None)
              or getattr(adapter, "state_timeline_engine", None))
    if engine is not None:
        try:
            cert = engine.build_timeline_certificate(item.question, all_records)
            if cert is not None:
                ku_certificate = cert.certificate or ""
        except Exception as exc:  # pragma: no cover - defensive
            ku_certificate = f"<ku error: {type(exc).__name__}: {exc}>"

    return multi_cert, multi_cert_valid, temporal_grounding, ku_certificate


def build_rows(adapter: LongMemEvalAdapter, items, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            break
        all_records = _ingest(adapter, item)
        pcc = adapter.compiler.compile(
            item.question, all_records, reference_date_str=item.question_date
        )

        gt_lower = item.answer.lower().strip()
        is_abstention_gt = (
            item.question_type == "abstention"
            or not item.answer_session_ids
            or item.question_id.endswith("_abs")
            or "information provided is not enough" in gt_lower
        )
        oracle_recall = False
        if item.answer_session_ids:
            oracle_recall = any(sid in pcc.context_text for sid in item.answer_session_ids)
        elif is_abstention_gt:
            oracle_recall = pcc.is_abstention
        elif gt_lower and gt_lower in pcc.context_text.lower():
            oracle_recall = True

        multi_cert, multi_valid, t_ground, ku_cert = _derived(adapter, item, pcc, all_records)

        rows.append({
            "qid": item.question_id,
            "qtype": item.question_type,
            "question": item.question,
            "question_date": item.question_date,
            "gt": item.answer,
            "answer_session_ids": list(item.answer_session_ids),
            "is_abstention_gt": is_abstention_gt,
            "oracle_recall": bool(oracle_recall),
            "token_cost": int(pcc.token_cost),
            "is_abstention_flag": bool(pcc.is_abstention),
            "context_text": pcc.context_text,
            "multi_cert": multi_cert,
            "multi_cert_valid": bool(multi_valid),
            "temporal_grounding": t_ground,
            "ku_certificate": ku_cert,
        })
        if (idx + 1) % 25 == 0:
            print(f"  cached {idx + 1}/{limit or len(items)}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_results/lme_context_cache.jsonl")
    ap.add_argument("--dataset", default="datasets/external/longmemeval_s_cleaned.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cache only the first N questions (smoke testing)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    adapter = LongMemEvalAdapter(dataset_path=args.dataset)
    print(f"loading dataset {args.dataset} ...", flush=True)
    items = adapter.load_dataset()
    print(f"  {len(items)} items; compiling contexts (CPU only)", flush=True)

    rows = build_rows(adapter, items, args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    mean_tok = sum(r["token_cost"] for r in rows) / len(rows) if rows else 0
    ora = sum(1 for r in rows if r["oracle_recall"]) / len(rows) * 100 if rows else 0
    print(f"\ncached {len(rows)} contexts -> {out}")
    print(f"  mean tokens/Q : {mean_tok:.1f}")
    print(f"  oracle recall : {ora:.1f}%")
    print(f"  elapsed       : {elapsed:.1f}s ({elapsed / max(1, len(rows)):.2f}s/Q)")


if __name__ == "__main__":
    main()
