import json, re

with open('/home/eli/Projects/artificial_memory/datasets/external/locomo10.json') as f:
    data = json.load(f)

conv = data[0]
print(f"Conv 0: {conv.get('sample_id')}")
print(f"Session dates: {[(k,v) for k,v in conv.items() if 'date' in k]}")

# Show single-hop questions that failed (Q84-150 range)
for i, qa in enumerate(conv.get('qa', [])):
    if qa.get('category') == 4 and 84 <= i <= 150:
        qid = f"conv-26-qa-{i:03d}"
        print(f"  Q{i:3d} [{qid}]: {qa['question'][:70]}")
        if qa.get('evidence'):
            print(f"       Evidence IDs: {qa['evidence']}")
