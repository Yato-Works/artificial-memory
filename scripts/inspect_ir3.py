#!/usr/bin/env python3
import json, sys, re
sys.path.insert(0, '/home/eli/Projects/artificial_memory/src')

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(0)

print(f"Total IR records: {len(ir_records)}")
print(f"Total turns: {len(turns)}")

for i, ir in enumerate(ir_records[:10]):
    rc = ir.raw_content[:100] if hasattr(ir, 'raw_content') else str(ir)[:100]
    print(f"  IR[{i}]: raw_content={rc}...")

print("\n--- Turns ---")
for t in turns[:10]:
    print(f"  dia_id={t.dia_id}, session={t.session_num}, text={t.text[:60]}...")

# Check the conversation structure
with open('/home/eli/Projects/artificial_memory/datasets/external/locomo10.json') as f:
    data = json.load(f)
conv = data[0]
session_keys = [k for k in conv.keys() if k.startswith('session_')]
print(f"\nSession keys: {session_keys}")

# Show raw conversation structure for session 2 and 5
for sk in ['session_2', 'session_5']:
    if sk in conv:
        val = conv[sk]
        if isinstance(val, list):
            print(f"\n{sk}: {len(val)} turns")
            for t in val:
                print(f"  dia_id={t['dia_id']}, speaker={t['speaker']}, text={t['text'][:60]}")

# Now check what the IR raw_content looks like
print("\n--- IR raw_content sample ---")
for ir in ir_records[:5]:
    print(f"  {ir.raw_content[:120]}")
    print(f"  attrs: dia_id={getattr(ir, 'dia_id', '?')}, source={ir.source}, entity={ir.entity}, value={ir.value[:50] if ir.value else '?'}")
    print()

# Check evidence IDs in QA
print("\n--- Evidence ID lookup ---")
for i, qa in enumerate(conv['qa']):
    qid = f"conv-26-qa-{i:03d}"
    if qid == "conv-26-qa-084":
        print(f"Q: {qa['question'][:70]}")
        print(f"Evidence: {qa['evidence']}")
        # The evidence is D2:5 - search for this in raw conversation
        ev = qa['evidence'][0]
        dia_prefix = ev.split(':')[0]  # D2
        turn_num = int(ev.split(':')[1])  # 5
        print(f"  Looking for dia_id containing '{dia_prefix}' with turn {turn_num}")
        # Find turns with matching dia_id
        for t in turns:
            if dia_prefix in t.dia_id or t.dia_id.startswith(dia_prefix.replace('D','')):
                print(f"  Turn: dia_id={t.dia_id}, text={t.text[:80]}")
