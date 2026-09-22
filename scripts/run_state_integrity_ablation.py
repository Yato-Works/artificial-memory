"""State Integrity Ablation Study on LoCoMo-10 Multi-Hop (Phase C3).

Evaluates the 32 Multi-Hop questions across 4 State Quality conditions:
  Condition A: No STATE (Baseline P4)
  Condition B: Gold / Oracle STATE (Theoretical Ceiling)
  Condition C: Retrieved / Synthesized STATE (Current StateSynthesizer)
  Condition D: Corrupted / Adversarial STATE (Poison Susceptibility)

Measures:
- Accuracy (%) per condition
- Ceiling Gap: Delta_ceiling = Acc(B) - Acc(C)
- Poison Susceptibility: Delta_poison = Acc(A) - Acc(D)
- Token overhead

Saves full telemetry to benchmark_results/protein/state_integrity_ablation_study.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.core.ir.memory_types import CoverageCertificate, ProofCarryingContext, QueryIntent
from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.protein.state_synthesizer import StateSynthesizer
from artificial_memory.protein.state_trigger import StateTrigger
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, LoCoMoQuestion
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Value corruption mapping for Condition D
CORRUPTIONS = {
    "sweden": "Norway",
    "single": "married",
    "counseling": "software engineering",
    "mental health": "accounting",
    "pottery": "origami",
    "camping": "scuba diving",
    "painting": "wood carving",
    "swimming": "archery",
    "figurines": "laptops",
    "shoes": "cameras",
    "sunset": "abstract cube",
    "rainbow flag": "pirate skull",
    "3": "5",
    "2": "10",
    "twice": "ten times",
}


def corrupt_text(text: str) -> str:
    """Deterministically corrupt entities/values in a state string."""
    corrupted = text
    for orig, repl in CORRUPTIONS.items():
        if orig.lower() in corrupted.lower():
            pattern = re.compile(re.escape(orig), re.IGNORECASE)
            corrupted = pattern.sub(repl, corrupted)
    if corrupted == text:
        # Fallback perturbation
        corrupted = f"{text} (unverified/false)"
    return corrupted


def generate_gold_state(question: LoCoMoQuestion) -> str:
    """Generate a crisp Gold STATE from question ground truth."""
    q_lower = question.question.lower()
    gt = question.ground_truth.strip()
    ent = "Caroline" if "caroline" in q_lower else "Melanie"

    if "move" in q_lower:
        return f"[STATE] {ent}'s home country is {gt}."
    elif "relationship" in q_lower:
        return f"[STATE] {ent}'s relationship status is {gt}."
    elif "career" in q_lower:
        return f"[STATE] {ent}'s career path is {gt}."
    elif "activities" in q_lower or "hobbies" in q_lower:
        return f"[STATE] {ent}'s activities include {gt}."
    elif "bought" in q_lower or "items" in q_lower:
        return f"[STATE] {ent} bought {gt}."
    elif "paint" in q_lower:
        return f"[STATE] {ent}'s painting is {gt}."
    elif "pottery" in q_lower:
        return f"[STATE] {ent}'s pottery types include {gt}."
    elif "symbols" in q_lower:
        return f"[STATE] {ent}'s important symbols include {gt}."
    elif "children" in q_lower or "kids" in q_lower:
        return f"[STATE] {ent} has {gt} children."
    elif "beach" in q_lower:
        return f"[STATE] {ent} went to the beach {gt} times."
    else:
        return f"[STATE] {ent} fact: {gt}."


def main():
    print("=" * 85)
    print("      AM APEX PHASE C3: STATE INTEGRITY ABLATION (32 QUESTIONS)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    # Compiler instance for baseline Top-10 compilation
    compiler_base = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=False,
    )

    compiler_syn = ProteinContextCompiler(
        policy=ContextPolicy.PRECISION,
        top_k_evidence=10,
        enable_chain_retention=False,
        enable_state_synthesis=True,
    )

    results_a = []
    results_b = []
    results_c = []
    results_d = []

    print("\n>>> Running Condition A: No STATE (Baseline P4)...")
    adapter.compiler = compiler_base
    for i, q in enumerate(multihop_questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results_a.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        print(f"  [Cond A {i+1:02d}/32] {status} | {res.tokens_used:3d} tok | Q: {q.question[:36]}")

    print("\n>>> Running Condition B: Gold / Oracle STATE (Ceiling)...")
    for i, q in enumerate(multihop_questions):
        # Compile base context and prepend Gold STATE
        pcc_base = compiler_base.compile(q.question, ir_records)
        gold_state = generate_gold_state(q)
        augmented_context = f"{gold_state}\n{pcc_base.context_text}"

        # Evaluate directly
        ans = answerer.answer(q.question, augmented_context)
        v_res = compiler_base.answer_verifier.verify(
            question=q.question,
            predicted_answer=ans.text,
            context=augmented_context,
            propositions=[],
        )
        # Score
        gt_lower = str(q.ground_truth).lower().strip()
        ans_lower = v_res.verified_answer.lower().strip()
        is_corr = gt_lower in ans_lower or ans_lower in gt_lower or any(w in ans_lower for w in gt_lower.split())

        from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoEvalResult
        res = LoCoMoEvalResult(
            question_id=q.question_id,
            category=q.category,
            oracle_recall=pcc_base.certificate.entity_coverage,
            predicted_answer=v_res.verified_answer,
            ground_truth=q.ground_truth,
            tokens_used=len(augmented_context.split()),
            latency_ms=ans.latency_ms,
            is_correct=is_corr,
        )
        results_b.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        print(f"  [Cond B {i+1:02d}/32] {status} | {res.tokens_used:3d} tok | Q: {q.question[:36]}")

    print("\n>>> Running Condition C: Retrieved STATE (Current StateSynthesizer)...")
    adapter.compiler = compiler_syn
    for i, q in enumerate(multihop_questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results_c.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        print(f"  [Cond C {i+1:02d}/32] {status} | {res.tokens_used:3d} tok | Q: {q.question[:36]}")

    print("\n>>> Running Condition D: Corrupted / Adversarial STATE (Poison Vulnerability)...")
    for i, q in enumerate(multihop_questions):
        pcc_base = compiler_base.compile(q.question, ir_records)
        gold_state = generate_gold_state(q)
        corrupt_state = corrupt_text(gold_state)
        corrupted_context = f"{corrupt_state}\n{pcc_base.context_text}"

        ans = answerer.answer(q.question, corrupted_context)
        v_res = compiler_base.answer_verifier.verify(
            question=q.question,
            predicted_answer=ans.text,
            context=corrupted_context,
            propositions=[],
        )
        gt_lower = str(q.ground_truth).lower().strip()
        ans_lower = v_res.verified_answer.lower().strip()
        is_corr = gt_lower in ans_lower or ans_lower in gt_lower

        res = LoCoMoEvalResult(
            question_id=q.question_id,
            category=q.category,
            oracle_recall=pcc_base.certificate.entity_coverage,
            predicted_answer=v_res.verified_answer,
            ground_truth=q.ground_truth,
            tokens_used=len(corrupted_context.split()),
            latency_ms=ans.latency_ms,
            is_correct=is_corr,
        )
        results_d.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        print(f"  [Cond D {i+1:02d}/32] {status} | {res.tokens_used:3d} tok | Q: {q.question[:36]}")

    # Metrics Summary
    acc_a = sum(1 for r in results_a if r.is_correct) / len(results_a) * 100
    acc_b = sum(1 for r in results_b if r.is_correct) / len(results_b) * 100
    acc_c = sum(1 for r in results_c if r.is_correct) / len(results_c) * 100
    acc_d = sum(1 for r in results_d if r.is_correct) / len(results_d) * 100

    tok_a = sum(r.tokens_used for r in results_a) / len(results_a)
    tok_b = sum(r.tokens_used for r in results_b) / len(results_b)
    tok_c = sum(r.tokens_used for r in results_c) / len(results_c)
    tok_d = sum(r.tokens_used for r in results_d) / len(results_d)

    print("\n" + "=" * 85)
    print("      PHASE C3: STATE INTEGRITY ABLATION SUMMARY")
    print("=" * 85)
    print(f"{'Condition':<32} | {'Accuracy':<14} | {'Tokens/Q':<10} | {'Interpretation':<22}")
    print("-" * 85)
    print(f"{'Condition A (No STATE)':<32} | {acc_a:5.1f}% ({sum(1 for r in results_a if r.is_correct):02d}/32) | {tok_a:5.1f} tok  | Baseline")
    print(f"{'Condition B (Gold STATE)':<32} | {acc_b:5.1f}% ({sum(1 for r in results_b if r.is_correct):02d}/32) | {tok_b:5.1f} tok  | Theoretical Ceiling")
    print(f"{'Condition C (Retrieved STATE)':<32} | {acc_c:5.1f}% ({sum(1 for r in results_c if r.is_correct):02d}/32) | {tok_c:5.1f} tok  | Current Synthesizer")
    print(f"{'Condition D (Corrupted STATE)':<32} | {acc_d:5.1f}% ({sum(1 for r in results_d if r.is_correct):02d}/32) | {tok_d:5.1f} tok  | Poison Vulnerability")
    print("=" * 85)
    print(f"  * Headroom to Ceiling (B - C):      {acc_b - acc_c:+5.1f} pt")
    print(f"  * Poison Susceptibility (A - D):    {acc_a - acc_d:+5.1f} pt")
    print("=" * 85)

    out_file = RESULTS_DIR / "state_integrity_ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(multihop_questions),
                "summary": {
                    "accuracy_a_no_state": acc_a,
                    "accuracy_b_gold_state": acc_b,
                    "accuracy_c_retrieved_state": acc_c,
                    "accuracy_d_corrupted_state": acc_d,
                    "ceiling_gap_b_minus_c": acc_b - acc_c,
                    "poison_drop_a_minus_d": acc_a - acc_d,
                },
                "questions": [
                    {
                        "question_id": q.question_id,
                        "question": q.question,
                        "ground_truth": q.ground_truth,
                        "cond_a": {"correct": ra.is_correct, "pred": ra.predicted_answer},
                        "cond_b": {"correct": rb.is_correct, "pred": rb.predicted_answer},
                        "cond_c": {"correct": rc.is_correct, "pred": rc.predicted_answer},
                        "cond_d": {"correct": rd.is_correct, "pred": rd.predicted_answer},
                    }
                    for q, ra, rb, rc, rd in zip(multihop_questions, results_a, results_b, results_c, results_d)
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
