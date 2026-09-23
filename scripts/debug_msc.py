#!/usr/bin/env python3
"""Debug MSC compilation for a single-hop question."""
import json, sys
sys.path.insert(0, '/home/eli/Projects/artificial_memory/src')

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

adapter = LoCoMoAdapter()
turns, questions, ir_records = adapter.load_conversation(0)

# Find the question
target = "conv-26-qa-084"  # How does Melanie prioritize self-care? Evidence: D2:5
for q in questions:
    if q.question_id == target:
        print(f"Q: {q.question}")
        print(f"Evidence IDs: {q.evidence_ids}")
        print(f"Category: {q.category} ({adapter.CATEGORY_NAMES.get(q.category)})")
        break

# Compile MSC
pcc = adapter.compiler.compile(q.question, ir_records)
print(f"\nMSC token cost: {pcc.token_cost}")
print(f"Is abstention: {pcc.is_abstention}")
print(f"Context length: {len(pcc.context_text)} chars")
print(f"\nContext:\n{pcc.context_text[:2000]}")

# Check if evidence ID is in context
ev_in_ctx = any(ev in pcc.context_text for ev in q.evidence_ids)
print(f"\nEvidence in context: {ev_in_ctx}")
for ev in q.evidence_ids:
    print(f"  {ev} in context: {ev in pcc.context_text}")

# Check what the working records are
print(f"\nTotal IR records: {len(ir_records)}")
working = adapter.compiler._working_records(q.question, ir_records)
print(f"Working records: {len(working)}")

# Show working records that match query tokens
q_tokens = set(q.question.lower().split())
print(f"\nQuery tokens: {q_tokens}")
for i, ir in enumerate(working):
    content_lower = ir.raw_content.lower()
    matched = q_tokens & set(content_lower.split())
    if matched:
        print(f"  IR[{i}]: {ir.raw_content[:80]}... matched: {matched}")
