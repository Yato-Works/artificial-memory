"""Scaled & Adversarial Benchmark Runner (Phase 3).

Evaluates Artificial Memory against competitor baselines (Mem0, MemGPT, MemoryBank, Simple RAG)
on the 100-question hierarchical dataset and the AM Adversarial Collision suite.

Usage:
    # Run the Adversarial Collision suite (AM destruction set):
    python scripts/run_scaled_benchmark.py --suite adversarial

    # Run the full 100-question hierarchical dataset:
    python scripts/run_scaled_benchmark.py --suite 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir import StructuredIR
from artificial_memory.recall.ir_resolver import UniversalIRResolver
from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaDataset,
    ArenaQuestion,
    ArenaScenario,
    ArenaTurn,
)
from artificial_memory.research.benchmarks.attribution import FailureAttributor, FailureTrace
from artificial_memory.research.benchmarks.decoupled_eval import evaluate_retrieval_only
from artificial_memory.research.benchmarks.llm import ABSTENTION_TEXT, OllamaAnswerer
from artificial_memory.research.benchmarks.metrics_extended import compute_extended_metrics
from artificial_memory.research.benchmarks.scorer import score_answer
from competitors import (
    Mem0Adapter,
    MemGPTAdapter,
    MemoryBankAdapter,
    SimpleRAGAdapter,
)

sys.stdout.reconfigure(encoding="utf-8")


def load_dataset_from_json(path: Path) -> ArenaDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    for s_dict in data["scenarios"]:
        turns = [
            ArenaTurn(
                session_index=t.get("session_index", 1),
                speaker=t["speaker"],
                content=t["content"],
                day=t.get("day", 0),
            )
            for t in s_dict["turns"]
        ]
        scenarios.append(ArenaScenario(
            scenario_id=s_dict["scenario_id"],
            title=s_dict.get("title", s_dict["scenario_id"]),
            day=s_dict.get("day", 0),
            turns=tuple(turns),
        ))
    questions = []
    for q_dict in data["questions"]:
        gt = tuple(tuple(group) for group in q_dict["ground_truth"])
        forbidden = tuple(q_dict.get("forbidden", ()))
        questions.append(ArenaQuestion(
            question_id=q_dict["question_id"],
            category=ArenaCategory(q_dict["category"]),
            question=q_dict["question"],
            ground_truth=gt,
            forbidden=forbidden,
            expected_abstention=q_dict.get("expected_abstention", False),
        ))
    return ArenaDataset(
        version=data.get("version", "v0.2.0"),
        scenarios=tuple(scenarios),
        questions=tuple(questions),
    )


def run_am_player(dataset: ArenaDataset, answerer: OllamaAnswerer, diagnose: bool = False):
    """Run Universal Cognitive IR AM Player with Decoupled Evaluation and Failure Attribution."""
    extractor = UniversalIRExtractor()
    resolver = UniversalIRResolver()
    attributor = FailureAttributor()

    all_ir_records: list[StructuredIR] = []
    raw_corpus: list[str] = []

    t0 = time.perf_counter()
    for sc in dataset.scenarios:
        for turn in sc.turns:
            raw_corpus.append(turn.content)
            all_ir_records.extend(extractor.extract(turn.content, default_source=turn.speaker))
    write_time_ms = (time.perf_counter() - t0) * 1000

    scores = []
    latencies = []
    context_tokens_list = []
    retrieval_scores = []
    traces: list[FailureTrace] = []

    for q in dataset.questions:
        qt0 = time.perf_counter()
        resolved = resolver.resolve(q.question, all_ir_records)
        ctx_tokens = len(resolved.context_text) // 4
        context_tokens_list.append(ctx_tokens)

        # Test A: Decoupled Retrieval-only evaluation
        r_score = evaluate_retrieval_only(q, resolved)
        retrieval_scores.append(r_score)

        # Test B: Downstream LLM Answering
        llm_input = ""
        if resolved.is_abstention:
            ans_text = ABSTENTION_TEXT
            llm_input = "[ABSTENTION SHORT-CIRCUIT: No LLM call]"
        elif resolved.has_conflict:
            prompt_ext = q.question + " (Note: State both reports and whether there is an unresolved conflict)"
            llm_input = f"{prompt_ext}\n\nContext:\n{resolved.context_text}"
            ans = answerer.answer(prompt_ext, resolved.context_text)
            ans_text = ans.text
        else:
            llm_input = f"{q.question}\n\nContext:\n{resolved.context_text}"
            ans = answerer.answer(q.question, resolved.context_text)
            ans_text = ans.text

        lat_ms = (time.perf_counter() - qt0) * 1000
        latencies.append(lat_ms)

        sc_res = score_answer(q, ans_text, "AM-Universal-IR")
        scores.append(sc_res)

        # Failure Attribution Diagnostic
        trace = attributor.diagnose(
            question=q,
            resolved=resolved,
            llm_input=llm_input,
            llm_output=ans_text,
            score=sc_res,
            raw_corpus=raw_corpus,
        )
        traces.append(trace)

    metrics = compute_extended_metrics(
        player_name="Artificial Memory (AM SOTA)",
        scores=scores,
        latencies_ms=latencies,
        context_tokens=context_tokens_list,
        write_calls=0,
        read_calls=len(dataset.questions),
    )

    retrieval_acc = sum(s.score for s in retrieval_scores) / len(retrieval_scores)

    return metrics, scores, retrieval_acc, traces


from artificial_memory.context.compiler import CognitiveContextCompiler
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler


def run_apex_msc_player(dataset: ArenaDataset, answerer: OllamaAnswerer):
    """Run Apex AM: Minimum Sufficient Context (MSC) with State Reconstruction & Recovery."""
    extractor = UniversalIRExtractor()
    compiler = MinimumSufficientContextCompiler()

    all_ir_records: list[StructuredIR] = []
    for sc in dataset.scenarios:
        for turn in sc.turns:
            all_ir_records.extend(extractor.extract(turn.content, default_source=turn.speaker))

    scores = []
    latencies = []
    context_tokens_list = []

    for q in dataset.questions:
        qt0 = time.perf_counter()
        pcc = compiler.compile(q.question, all_ir_records)
        context_tokens_list.append(pcc.token_cost)

        if pcc.is_abstention:
            ans_text = ABSTENTION_TEXT
        else:
            ans = answerer.answer(q.question, pcc.context_text)
            ans_text = ans.text

        lat_ms = (time.perf_counter() - qt0) * 1000
        latencies.append(lat_ms)

        sc_res = score_answer(q, ans_text, "AM-Apex-MSC")
        scores.append(sc_res)

    metrics = compute_extended_metrics(
        player_name="Artificial Memory: Apex MSC",
        scores=scores,
        latencies_ms=latencies,
        context_tokens=context_tokens_list,
        write_calls=0,
        read_calls=len(dataset.questions),
    )
    return metrics, scores


def run_am_compiler_player(dataset: ArenaDataset, answerer: OllamaAnswerer):
    """Run AM with Cognitive Context Compiler & Conflict State Gating."""
    extractor = UniversalIRExtractor()
    compiler = CognitiveContextCompiler()

    all_ir_records: list[StructuredIR] = []
    for sc in dataset.scenarios:
        for turn in sc.turns:
            all_ir_records.extend(extractor.extract(turn.content, default_source=turn.speaker))

    scores = []
    latencies = []
    context_tokens_list = []

    for q in dataset.questions:
        qt0 = time.perf_counter()
        compiled = compiler.compile(q.question, all_ir_records)
        ctx_tokens = compiled.token_estimate
        context_tokens_list.append(ctx_tokens)

        if compiled.is_abstention:
            ans_text = ABSTENTION_TEXT
        else:
            ans = answerer.answer(q.question, compiled.text)
            ans_text = ans.text

        lat_ms = (time.perf_counter() - qt0) * 1000
        latencies.append(lat_ms)

        sc_res = score_answer(q, ans_text, "AM-Compiler-Full")
        scores.append(sc_res)

    metrics = compute_extended_metrics(
        player_name="AM + Context Compiler (Full)",
        scores=scores,
        latencies_ms=latencies,
        context_tokens=context_tokens_list,
        write_calls=0,
        read_calls=len(dataset.questions),
    )
    return metrics, scores


def print_multiaxis_report(metrics_list: list[Any], suite_name: str):
    """Print multi-axis evaluation report without flattening into a single score."""
    print("\n" + "=" * 100)
    print(f"               MULTI-AXIS RESEARCH EVALUATION MATRIX ({suite_name.upper()} SUITE)")
    print("=" * 100)
    print(f"  {'Player / Architecture':<30} | {'Answer Acc':<10} | {'FPR (Lie)':<10} | {'Abst Acc':<10} | {'Tokens/Q':<10} | {'Write LLM':<10} | {'p50 Lat'}")
    print("-" * 100)
    for m in metrics_list:
        print(
            f"  {m.player_name:<30} | "
            f"{m.accuracy * 100:>9.1f}% | "
            f"{m.false_positive_rate * 100:>9.1f}% | "
            f"{m.abstention_accuracy * 100:>9.1f}% | "
            f"{m.mean_context_tokens:>9.0f} | "
            f"{m.total_write_llm_calls:>10d} | "
            f"{m.latency_p50_ms:>6.0f}ms"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Run Scaled & Adversarial Memory Benchmark")
    parser.add_argument("--suite", choices=["adversarial", "100"], default="adversarial",
                        help="Benchmark suite to run: 'adversarial' (default) or '100'")
    parser.add_argument("--players", choices=["all", "am_only", "ablation"], default="all",
                        help="Players to run: 'all' (AM + competitors), 'am_only', or 'ablation' (AM versions)")
    parser.add_argument("--diagnose", action="store_true", default=True,
                        help="Perform Failure Attribution on any failed questions")
    args = parser.parse_args()

    suite_name = "adversarial.json" if args.suite == "adversarial" else "arena_100.json"
    dataset_path = REPO_ROOT / "dataset_hidden" / suite_name

    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found. Run generator first.")
        sys.exit(1)

    dataset = load_dataset_from_json(dataset_path)
    print("=" * 80)
    print(f"      SCALED BENCHMARK RUNNER (Multi-Axis & Ablation Edition)")
    print(f"      Suite: {args.suite.upper()} | Scenarios: {len(dataset.scenarios)} | Questions: {len(dataset.questions)}")
    print("=" * 80)

    answerer = OllamaAnswerer()
    all_metrics = []

    # 1. Run AM (Universal Cognitive IR)
    print("\n[Running Artificial Memory: Universal IR (Baseline)]...")
    am_metrics, am_scores, am_retrieval_acc, am_traces = run_am_player(dataset, answerer, diagnose=args.diagnose)
    all_metrics.append(am_metrics)

    print(f"\n>> AM Decoupled Scores:")
    print(f"   - Test A: Retrieval-only Accuracy : {am_retrieval_acc * 100:.1f}%")
    print(f"   - Test B: Answer Accuracy (w/ LLM): {am_metrics.accuracy * 100:.1f}%")

    # 2. Run Apex MSC (Minimum Sufficient Context with State Reconstruction & Recovery)
    if args.players in ["all", "ablation"]:
        print("\n[Running Artificial Memory: Apex MSC (State Reconstruction & Recovery)]...")
        apex_metrics, _ = run_apex_msc_player(dataset, answerer)
        all_metrics.append(apex_metrics)

    # 3. Run AM + Context Compiler (Ablation)
    if args.players == "ablation":
        print("\n[Running Artificial Memory: + Context Compiler & Conflict Gating (Ablation)]...")
        compiler_metrics, _ = run_am_compiler_player(dataset, answerer)
        all_metrics.append(compiler_metrics)

    # 4. Run Competitors
    if args.players == "all":
        competitors = [
            SimpleRAGAdapter({"context_budget": 2000}, answerer=answerer),
            MemoryBankAdapter({"context_budget": 2000}, answerer=answerer),
            MemGPTAdapter({"context_budget": 2000}, answerer=answerer),
            Mem0Adapter({"context_budget": 2000}, answerer=answerer),
        ]

        for comp in competitors:
            print(f"\n[Running {comp.name}]...")
            comp.ingest(dataset.scenarios)
            comp_scores = []
            latencies = []
            tokens = []

            for q in dataset.questions:
                qt0 = time.perf_counter()
                ans = comp.answer(q, context_budget=2000)
                lat_ms = (time.perf_counter() - qt0) * 1000
                latencies.append(lat_ms)
                tokens.append(ans.context_tokens_used)
                sc_res = score_answer(q, ans.text, comp.name)
                comp_scores.append(sc_res)

            m = compute_extended_metrics(
                player_name=comp.name,
                scores=comp_scores,
                latencies_ms=latencies,
                context_tokens=tokens,
                write_calls=comp.costs.write_llm_calls,
                read_calls=comp.costs.answer_llm_calls,
            )
            all_metrics.append(m)

    # 4. Print Multi-Axis Evaluation Matrix
    print_multiaxis_report(all_metrics, args.suite)

    # 5. Failure Attribution Output
    if args.diagnose and am_traces:
        fails = [t for t in am_traces if t.failure_origin != "None"]
        if fails:
            print("\n" + "=" * 90)
            print(f"                        FAILURE ATTRIBUTION AUDIT ({len(fails)} FAILURES)")
            print("=" * 90)
            for f in fails:
                print(f"\n[FAIL DIAGNOSIS] Question ID: {f.question_id} ({f.category})")
                print(f"  1. Question            : {f.question_text}")
                print(f"  2. Ground Truth        : {f.ground_truth}")
                print(f"  3. Context IR Text     : {f.context_ir_text}")
                print(f"  4. LLM Output          : {f.llm_output_text}")
                print(f"  5. Evaluator Decision  : {f.outcome} (score={f.score})")
                print(f"  6. Failure Origin      : >> {f.failure_origin} <<")
                print(f"  7. Diagnostic Analysis : {f.diagnostic_reasoning}")

            # Save traces
            out_dir = REPO_ROOT / "benchmark" / "results" / "attribution"
            out_dir.mkdir(parents=True, exist_ok=True)
            trace_path = out_dir / f"failure_traces_{args.suite}.jsonl"
            with open(trace_path, "w", encoding="utf-8") as tf:
                for t in am_traces:
                    tf.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            print(f"\nFull audit traces saved: {trace_path}")


if __name__ == "__main__":
    main()
