"""Benchmark AM Apex Phase QUAD-CONQUEST on LoCoMo-10 Adversarial (47 Questions)."""

import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.protein.protein_compiler import ProteinContextCompiler, ContextPolicy
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(conv_idx=0)
answerer = OllamaAnswerer()

compiler = ProteinContextCompiler(
    policy=ContextPolicy.PRECISION,
    top_k_evidence=10,
    enable_chain_retention=False,
    enable_state_synthesis=True,
)
adapter.compiler = compiler

adv_qs = [q for q in questions if q.category == 5]
print(f"Loaded {len(adv_qs)} adversarial questions from LoCoMo Conv 0.")

correct = 0
recalled = 0
total_tokens = 0
t0 = time.perf_counter()

print("\n" + f"{'Q#':<5} | {'Status':<6} | {'Ora':<4} | {'Tokens':<8} | Question")
print("-" * 80)

for i, q in enumerate(adv_qs):
    res = adapter.evaluate_question(q, turns, ir_records, answerer)
    if res.is_correct:
        correct += 1
    if res.oracle_recall:
        recalled += 1
    total_tokens += res.tokens_used

    status = "PASS" if res.is_correct else "FAIL"
    ora = "YES" if res.oracle_recall else "NO"
    print(f"[{i+1:02d}/{len(adv_qs):02d}] | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {q.question[:35]}")

elapsed = time.perf_counter() - t0
acc = correct / len(adv_qs) * 100
ora_acc = recalled / len(adv_qs) * 100
mean_tok = total_tokens / len(adv_qs)

print("\n" + "=" * 80)
print(f"LoCoMo Adversarial Accuracy: {correct}/{len(adv_qs)} ({acc:.1f}%)")
print(f"LoCoMo Adversarial Oracle:   {recalled}/{len(adv_qs)} ({ora_acc:.1f}%)")
print(f"Mean Tokens / Question:      {mean_tok:.1f} tokens")
print(f"Total Time:                  {elapsed:.1f}s ({elapsed/len(adv_qs):.2f}s/Q)")
print("=" * 80)
