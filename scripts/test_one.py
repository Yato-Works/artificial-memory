#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LoCoMoAdapter()
answerer = OllamaAnswerer()
turns, questions, ir = adapter.load_conversation(0)
q = questions[0]
print('Q:', q.question[:60])
print('Evidence:', q.evidence_ids)
result = adapter.evaluate_question(q, turns, ir, answerer)
print('Pred:', result.predicted_answer[:80])
print('GT:', result.ground_truth[:80])
print('Correct:', result.is_correct)
print('Oracle:', result.oracle_recall)
