#!/usr/bin/env python3
"""Run a single question with the improved code."""
import sys, json, time
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, FROZEN_MODEL

adapter = LoCoMoAdapter()
answerer = OllamaAnswerer(model=FROZEN_MODEL, timeout_seconds=300.0)
turns, questions, ir = adapter.load_conversation(0)

# Run open-domain question (cat 3)
for q in questions[:15]:
    if q.category == 3:
        print(f"Q: {q.question[:70]}")
        print(f"Evidence: {q.evidence_ids}")
        print(f"GT: {q.ground_truth[:80]}")
        t0 = time.time()
        result = adapter.evaluate_question(q, turns, ir, answerer)
        elapsed = time.time() - t0
        print(f"Pred: {result.predicted_answer[:80]}")
        print(f"Correct: {result.is_correct} | Oracle: {result.oracle_recall} | {elapsed:.1f}s")
        print()
        break  # Just one question
