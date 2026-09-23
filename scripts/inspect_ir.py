#!/usr/bin/env python3
"""Inspect the IR records and evidence IDs for Conv 0 questions."""
import json, re, sys
sys.path.insert(0, '/home/eli/Projects/artificial_memory/src')

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(0)

# Show IR record IDs for a few failed questions
target_qids = ["conv-26-qa-084", "conv-26-qa-085", "conv-26-qa-099", "conv-26-qa-104"]
evidence_map = {qid: qa["evidence"] for q in [questions[i] for i in range(len(questions))] 
                for qid in [q.question_id] if qid in target_qids for qa in [next(
                    (qa for qa in json.load(open('/home/eli/Projects/artificial_memory/datasets/external/locomo10.json'))[0]['qa']), None)]}

# Simpler: load raw dataset
with open('/home/eli/Projects/artificial_memory/datasets/external/locomo10.json') as f:
    data = json.load(f)
conv = data[0]

for i, qa in enumerate(conv['qa']):
    qid = f"conv-26-qa-{i:03d}"
    if qid in target_qids:
        print(f"\n=== {qid} ===")
        print(f"Q: {qa['question'][:70]}")
        print(f"Evidence IDs: {qa['evidence']}")
        for evid in qa['evidence']:
            dia_id = evid.split(':')[0] if ':' in evid else evid
            turn_num = int(evid.split(':')[1]) if ':' in evid else 1
            # Find the matching turn
            for t in turns:
                if t.dia_id == dia_id:
                    print(f"  Turn {t.dia_id}: [{t.session_num}] {t.text[:80]}...")
                    break
