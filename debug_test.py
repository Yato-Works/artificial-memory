from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == '0a995998':
        res = adapter.evaluate_item(item, OllamaAnswerer())
        print('Q:', item.question)
        print('GT:', item.answer)
        print('Pred:', res.predicted_answer)
        print('Correct:', res.is_correct)
        print('Oracle:', res.oracle_recall)
        break