"""Cached-context A/B: which reader model should answer LoCoMo questions?

Retrieval is held constant by replaying the frozen production contexts
(``benchmark_results/locomo_context_cache.jsonl``), so the only variable is the
LLM that reads them.  Every arm runs the production ``evaluate_question`` path:
category directives, AnswerVerifier guards, official abstention surface and the
frozen category scorers all stay in place.

The measured anatomy motivates this: in single-hop, 117 of the 148 residual
errors are WRONG_VALUE (62 of them with the ground-truth turn already in the
context).  That is a reading failure, not a retrieval failure - so the reader is
the next lever.

Usage:
  python scripts/ab_reader_model.py --models phi4-mini:latest,qwen3:4b --limit 100
  python scripts/ab_reader_model.py --models phi4-mini:latest --convs 0,1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import (  # noqa: E402
    CachedContext,
    LoCoMoAdapter,
)
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402

CACHE = Path("benchmark_results/locomo_context_cache.jsonl")
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load_cache(path: Path) -> dict[str, CachedContext]:
    out: dict[str, CachedContext] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["qid"]] = CachedContext(context_text=d["context"])
    return out


def _evaluate_with_retry(adapter, q, turns, ir_records, failures: list[str], attempts: int = 4):
    """Run one question, tolerating transient Ollama 5xx / toolchain errors.

    A reader A/B is a long sequential run against a local server; a single
    transient ``500 Internal Server Error`` must not destroy hours of work (an
    earlier 7B run died that way after four conversations).  Questions whose
    retries are exhausted are recorded and excluded from the denominator instead
    of being scored as wrong.
    """
    for i in range(attempts):
        try:
            return adapter.evaluate_question(q, turns, ir_records, adapter.answerer)
        except Exception as exc:  # noqa: BLE001 - local-server robustness
            if i == attempts - 1:
                failures.append(q.question_id)
                print(f"    !! skipped {q.question_id}: {type(exc).__name__}", flush=True)
                return None
            time.sleep(2.0 * (i + 1))
    return None



def run_arm(adapter: LoCoMoAdapter, model: str, convs: list[int], limit: int | None,
            qid_filter: set | None = None, num_ctx: int | None = None) -> dict:
    adapter.answerer = OllamaAnswerer(model=model, num_ctx=num_ctx)
    totals: dict[str, list[int]] = {"ALL": [0, 0]}
    t0 = time.perf_counter()
    n = 0
    with_ora = 0
    failures: list[str] = []
    done = 0
    for conv_idx in convs:
        turns, questions, ir_records = adapter.load_conversation(conv_idx)
        qs = questions if limit is None else questions[:limit]
        for q in qs:
            if qid_filter is not None and q.question_id not in qid_filter:
                continue
            res = _evaluate_with_retry(adapter, q, turns, ir_records, failures)
            if res is None:
                continue
            cat = CAT.get(q.category, str(q.category))
            totals["ALL"][0] += int(res.is_correct)
            totals["ALL"][1] += 1
            totals.setdefault(cat, [0, 0])
            totals[cat][0] += int(res.is_correct)
            totals[cat][1] += 1
            with_ora += int(res.oracle_recall)
            n += 1
            done += 1
            if done % 50 == 0:
                acc = totals["ALL"][0] / totals["ALL"][1] * 100
                print(f"    [{model}] {done} done  acc={acc:5.1f}%  "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    return {"model": model, "n": n, "totals": totals,
            "oracle": with_ora / n * 100 if n else 0.0, "sec": elapsed,
            "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="phi4-mini:latest",
                    help="Comma-separated Ollama model tags (each becomes one arm)")
    ap.add_argument("--convs", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--limit", type=int, default=None, help="Questions per conversation")
    ap.add_argument("--qids-file", default=None,
                    help="Optional file of question ids; when given, only these "
                         "questions run (decision-focused subset A/B)")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Shared Ollama context window for every arm (A/B only). "
                         "Large model tags default to a 32K KV cache and can return "
                         "HTTP 500 on an 8 GB GPU.")
    ap.add_argument("--out", default="benchmark_results/ab_reader_model.json")
    args = ap.parse_args()

    convs = [int(x) for x in args.convs.split(",") if x.strip() != ""]
    qid_filter = None
    if args.qids_file:
        qid_filter = set(Path(args.qids_file).read_text(encoding="utf-8").split())
        print(f"qid subset: {len(qid_filter)} questions")
    cache = load_cache(Path(args.cache))
    print(f"cached contexts: {len(cache)}  convs={convs}  limit={args.limit}")

    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        adapter = LoCoMoAdapter()
        adapter.context_override = cache  # retrieval held constant
        r = run_arm(adapter, model, convs, args.limit, qid_filter=qid_filter,
                    num_ctx=args.num_ctx)
        results.append(r)
        t = r["totals"]
        line = (f"{model:<22} acc={t['ALL'][0] / t['ALL'][1] * 100:5.1f}% "
                f"({t['ALL'][0]}/{t['ALL'][1]})  oracle={r['oracle']:5.1f}%  {r['sec']:.0f}s")
        for c in [4, 1, 2, 5, 3]:
            name = CAT[c]
            if name in t and t[name][1]:
                line += f"  {name}={t[name][0] / t[name][1] * 100:5.1f}%"
        print(line, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()


