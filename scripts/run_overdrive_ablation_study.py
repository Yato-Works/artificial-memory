"""Ablation Study for AM Apex Overdrive Core.

Systematically measures the causal contribution of each component:
1. Full Overdrive Core (All active)
2. - Temporal Engine (disables date arithmetic)
3. - Adaptive Search (disables bounded 3-hop expansion)
4. - Proposition Gate (disables integrity check)
5. - State Supersession (disables state history)
6. - Answer Verifier (disables answer filtering)

Evaluates on balanced subsets across LoCoMo and LongMemEval failure categories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ablation_eval(
    name: str,
    compiler: MinimumSufficientContextCompiler,
    locomo_adapter: LoCoMoAdapter,
    longmem_adapter: LongMemEvalAdapter,
    answerer: OllamaAnswerer,
    locomo_turns: list,
    locomo_qs: list,
    locomo_ir: list,
    longmem_items: list,
) -> dict[str, float]:
    """Evaluate a specific configuration on balanced multi-category subsets."""
    print(f"\n--- Running Configuration: {name} ---")
    t0 = time.perf_counter()
    locomo_adapter.compiler = compiler
    longmem_adapter.compiler = compiler

    # 1. LoCoMo Multi-Hop (10 Qs)
    mh_qs = [q for q in locomo_qs if q.category == 1][:10]
    mh_correct = sum(1 for q in mh_qs if locomo_adapter.evaluate_question(q, locomo_turns, locomo_ir, answerer).is_correct)

    # 2. LoCoMo Adversarial (10 Qs)
    adv_qs = [q for q in locomo_qs if q.category == 5][:10]
    adv_correct = sum(1 for q in adv_qs if locomo_adapter.evaluate_question(q, locomo_turns, locomo_ir, answerer).is_correct)

    # 3. LongMemEval Temporal (10 Qs)
    temp_items = [it for it in longmem_items if it.question_type == "temporal-reasoning"][:10]
    temp_correct = sum(1 for it in temp_items if longmem_adapter.evaluate_item(it, answerer).is_correct)

    # 4. LongMemEval Knowledge Update (10 Qs)
    upd_items = [it for it in longmem_items if it.question_type == "knowledge-update"][:10]
    upd_correct = sum(1 for it in upd_items if longmem_adapter.evaluate_item(it, answerer).is_correct)

    total_q = len(mh_qs) + len(adv_qs) + len(temp_items) + len(upd_items)
    total_correct = mh_correct + adv_correct + temp_correct + upd_correct
    overall_acc = total_correct / total_q * 100 if total_q > 0 else 0.0

    elapsed = time.perf_counter() - t0
    print(f"[{name}] Overall: {overall_acc:.1f}% ({total_correct}/{total_q}) | MH: {mh_correct}/10 | Adv: {adv_correct}/10 | Temp: {temp_correct}/10 | Upd: {upd_correct}/10 in {elapsed:.1f}s")

    return {
        "overall": overall_acc,
        "multi_hop": mh_correct / 10.0 * 100,
        "adversarial": adv_correct / 10.0 * 100,
        "temporal": temp_correct / 10.0 * 100,
        "knowledge_update": upd_correct / 10.0 * 100,
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 80)
    print("        AM APEX OVERDRIVE CORE: ABLATION STUDY (FROZEN SPEC)")
    print("=" * 80)

    answerer = OllamaAnswerer()
    locomo_adapter = LoCoMoAdapter()
    longmem_adapter = LongMemEvalAdapter()

    print("Loading datasets...")
    locomo_turns, locomo_qs, locomo_ir = locomo_adapter.load_conversation(conv_idx=0)
    longmem_items = longmem_adapter.load_dataset()
    print("Datasets loaded successfully.\n")

    ablation_results = {}

    # 1. Full Overdrive Core
    compiler_full = MinimumSufficientContextCompiler()
    ablation_results["Full Overdrive Core"] = run_ablation_eval(
        "Full Overdrive Core", compiler_full, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # 2. - Temporal Engine
    compiler_no_temp = MinimumSufficientContextCompiler()
    compiler_no_temp.temporal_resolver.resolve = lambda *args, **kwargs: None
    ablation_results["- Temporal Engine"] = run_ablation_eval(
        "- Temporal Engine", compiler_no_temp, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # 3. - Adaptive Search
    compiler_no_adapt = MinimumSufficientContextCompiler()
    compiler_no_adapt.adaptive_searcher.max_hops = 0
    ablation_results["- Adaptive Search"] = run_ablation_eval(
        "- Adaptive Search", compiler_no_adapt, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # 4. - Proposition Gate
    compiler_no_gate = MinimumSufficientContextCompiler()
    compiler_no_gate.integrity_gate.check = lambda plan, props: type("Obj", (), {"is_valid": True, "subject_matched": True, "predicate_matched": True, "grounding_note": None, "recommended_abstention": False})()
    ablation_results["- Proposition Gate"] = run_ablation_eval(
        "- Proposition Gate", compiler_no_gate, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # 5. - State Supersession
    compiler_no_state = MinimumSufficientContextCompiler()
    compiler_no_state.state_engine.resolve = lambda *args, **kwargs: None
    ablation_results["- State Supersession"] = run_ablation_eval(
        "- State Supersession", compiler_no_state, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # 6. - Answer Verifier
    compiler_no_verif = MinimumSufficientContextCompiler()
    def mock_verify(*args, **kwargs):
        ans = kwargs.get("predicted_answer")
        if ans is None and len(args) > 1:
            ans = args[1]
        elif ans is None and len(args) > 0:
            ans = args[0]
        return type("Obj", (), {"is_verified": True, "verified_answer": ans or "", "hallucination_detected": False, "notes": None})()
    compiler_no_verif.answer_verifier.verify = mock_verify
    ablation_results["- Answer Verifier"] = run_ablation_eval(
        "- Answer Verifier", compiler_no_verif, locomo_adapter, longmem_adapter, answerer,
        locomo_turns, locomo_qs, locomo_ir, longmem_items
    )

    # Save summary
    out_file = RESULTS_DIR / "overdrive_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("                    ABLATION STUDY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'Overall':<8} | {'Multi-Hop':<10} | {'Adversarial':<12} | {'Temporal':<10} | {'Update':<8}")
    print("-" * 80)
    base_acc = ablation_results["Full Overdrive Core"]["overall"]
    for name, r in ablation_results.items():
        delta = r["overall"] - base_acc
        delta_str = f"({delta:+.1f}%)" if name != "Full Overdrive Core" else "(baseline)"
        print(f"{name:<25} | {r['overall']:5.1f}%  | {r['multi_hop']:5.1f}%     | {r['adversarial']:5.1f}%       | {r['temporal']:5.1f}%     | {r['knowledge_update']:5.1f}%")
    print("=" * 80)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
