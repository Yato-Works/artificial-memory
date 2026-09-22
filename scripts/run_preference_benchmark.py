"""Benchmark AM Apex Phase QUAD-CONQUEST on LongMemEval Preference (30 Questions)."""

import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.protein.protein_compiler import ProteinContextCompiler, ContextPolicy
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()
answerer = OllamaAnswerer()

compiler = ProteinContextCompiler(
    policy=ContextPolicy.PRECISION,
    top_k_evidence=10,
    enable_chain_retention=False,
    enable_state_synthesis=True,
)
adapter.compiler = compiler

prefs = [it for it in items if it.question_type == "single-session-preference"]
print(f"Loaded {len(prefs)} preference questions from LongMemEval.")

correct = 0
recalled = 0
total_tokens = 0
t0 = time.perf_counter()

print("\n" + f"{'Q#':<5} | {'Status':<6} | {'Ora':<4} | {'Tokens':<8} | Question")
print("-" * 80)

for i, it in enumerate(prefs):
    res = adapter.evaluate_item(it, answerer)
    if res.is_correct:
        correct += 1
    if res.oracle_recall:
        recalled += 1
    total_tokens += res.tokens_used

    status = "PASS" if res.is_correct else "FAIL"
    ora = "YES" if res.oracle_recall else "NO"
    print(f"[{i+1:02d}/{len(prefs):02d}] | {status:<6} | {ora:<4} | {res.tokens_used:3d} tok | Q: {it.question[:35]}")

elapsed = time.perf_counter() - t0
acc = correct / len(prefs) * 100
ora_acc = recalled / len(prefs) * 100
mean_tok = total_tokens / len(prefs)

print("\n" + "=" * 80)
print(f"LongMemEval Preference Accuracy: {correct}/{len(prefs)} ({acc:.1f}%)")
print(f"LongMemEval Preference Oracle:   {recalled}/{len(prefs)} ({ora_acc:.1f}%)")
print(f"Mean Tokens / Question:         {mean_tok:.1f} tokens")
print(f"Total Time:                     {elapsed:.1f}s ({elapsed/len(prefs):.2f}s/Q)")
print("=" * 80)
