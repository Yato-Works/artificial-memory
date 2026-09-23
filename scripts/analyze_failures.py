import json

with open('/home/eli/Projects/artificial_memory/benchmark_results/frozen/failure_analysis/failure_ledger.jsonl') as f:
    entries = [json.loads(line) for line in f]

print('=== Single-Hop Oracle Gap Failures ===')
for e in entries:
    if e['category'] == 'single-hop' and e['oracle_gap']:
        print(f"  {e['question_id']} | {e['failure_type']}")

print()
print('=== Open-Domain Failures ===')
for e in entries:
    if e['category'] == 'open-domain':
        ret = e.get('retrieval', {})
        print(f"  {e['question_id']} | {e['failure_type']} | oracle={ret.get('oracle_recall', 'N/A')}")

# Write summary
from collections import defaultdict
by_category = defaultdict(lambda: {'total': 0, 'oracle_gaps': 0})
for e in entries:
    cat = e['category']
    by_category[cat]['total'] += 1
    if e['oracle_gap']:
        by_category[cat]['oracle_gaps'] += 1

with open('/home/eli/Projects/artificial_memory/benchmark_results/frozen/failure_analysis/summary.json', 'w') as f:
    json.dump(dict(by_category), f, indent=2)
