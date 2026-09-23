#!/usr/bin/env python3
import json, sys, re
sys.path.insert(0, '/home/eli/Projects/artificial_memory/src')

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(0)

# Show the first few IR records
print(f"Total IR records: {len(ir_records)}")
print(f"Total turns: {len(turns)}")
for i, ir in enumerate(ir_records[:10]):
    print(f"  IR[{i}]: dia_id={ir.dia_id}, raw_content={ir.raw_content[:80]}...")

print("\n--- Turns ---")
for t in turns[:10]:
    print(f"  dia_id={t.dia_id}, session={t.session_num}, text={t.text[:60]}...")

# Check if D2:5 appears in any IR raw_content  
print("\n--- Searching for D2:5 in IR ---")
for ir in ir_records:
    if "D2:5" in ir.raw_content or re.search(r"D2:5", ir.raw_content):
        print(f"  Found: {ir.raw_content[:100]}")

# Check the conversation structure
with open('/home/eli/Projects/artificial_memory/datasets/external/locomo10.json') as f:
    data = json.load(f)
conv = data[0]
session_keys = [k for k in conv.keys() if k.startswith('session_')]
print(f"\nSession keys: {session_keys}")
for sk in session_keys:
    val = conv[sk]
    if isinstance(val, list):
        print(f"  {sk}: {len(val)} turns")
        for t in val[:2]:
            print(f"    dia_id={t['dia_id']}, speaker={t['speaker']}, text={t['text'][:50]}")
