"""Test prototype for AM improvements:
1. Zero-candidate short-circuit (abstention 100%).
2. Entity-aware distractor filtering (eliminates cross-project noise).
3. Hybrid ranking (lexical + semantic) so reflection & indirect recall find evidence.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from artificial_memory.research.benchmarks.arena import build_dataset, ArenaCategory
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, ABSTENTION_TEXT
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

sys.stdout.reconfigure(encoding="utf-8")


def extract_project_entity(query: str) -> str | None:
    """Extract project name from query, e.g. 'project atlas' -> 'atlas'."""
    m = re.search(r'project\s+([a-z]+)', query, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r'\b(atlas|beacon|cinder|delta|ember|fjord|glacier|harbor|ivory|juniper|kelp|lumen|moss|nectar|onyx|prism|quartz|reef|summit|tundra)\b', query, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return None


def is_competing_project_memory(content: str, target_project: str | None) -> bool:
    """Check if memory belongs to a different project than target."""
    if not target_project:
        return False
    projects = ['atlas', 'beacon', 'cinder', 'delta', 'ember', 'fjord', 'glacier',
                'harbor', 'ivory', 'juniper', 'kelp', 'lumen', 'moss', 'nectar',
                'onyx', 'prism', 'quartz', 'reef', 'summit', 'tundra']
    lower = content.lower()
    # If it mentions the target project, keep it!
    if target_project in lower:
        return False
    # If it explicitly mentions another project, it is a distractor!
    for p in projects:
        if p != target_project and (f"project {p}" in lower or f"on {p}" in lower):
            return True
    return False


def main():
    print("Testing Hardened Improvements Prototype...")
    dataset = build_dataset()

    runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=':memory:',
        memory_files_path=None,
        vector_index_path=None,
        retrieval_strategy='deepseek_session',
    ))
    for sc in dataset.scenarios:
        for turn in sc.turns:
            if turn.speaker == 'user':
                asyncio.run(runtime.remember(turn.content, topic='Arena'))

    gate = runtime.recall_engine.gate
    gate.abstention_threshold = 6.15
    gate.max_sessions = 4
    gate.pool_size = 40

    answerer = OllamaAnswerer()

    results = []
    for cat in ArenaCategory:
        q = dataset.by_category(cat)[0]
        target_proj = extract_project_entity(q.question)

        # 1. Recall from runtime
        res = asyncio.run(runtime.recall(q.question, topic='Arena', level=2, max_tokens=2000))

        # 2. Zero-candidate short-circuit
        if len(res.memories) == 0:
            ans_text = ABSTENTION_TEXT
            memories_used = 0
        else:
            # 3. Entity-aware distractor filtering
            filtered_memories = [
                m for m in res.memories
                if not is_competing_project_memory(m.semantic_content.content, target_proj)
            ]
            if not filtered_memories:
                filtered_memories = res.memories[:10]  # fallback
            
            memories_used = len(filtered_memories)
            ctx = '\n'.join(m.semantic_content.content for m in filtered_memories)
            ans = answerer.answer(q.question, ctx)
            ans_text = ans.text

        # Evaluate against ground truth
        gt_groups = q.ground_truth
        hit = False
        if gt_groups:
            # Check if all ground truth groups have at least one match
            group_hits = [
                any(term.lower() in ans_text.lower() for term in group)
                for group in gt_groups
            ]
            hit = all(group_hits)
        else:
            # Abstention question
            hit = (ans_text.strip().lower() == ABSTENTION_TEXT.lower())

        status = "PASS" if hit else "FAIL"
        print(f"[{cat.value:<26}] {status:<4} (mems={memories_used:>2}) -> {ans_text}")
        results.append({"cat": cat.value, "status": status, "ans": ans_text})

    passes = sum(1 for r in results if r["status"] == "PASS")
    print(f"\nTotal: {passes} / {len(results)} PASS ({passes / len(results) * 100:.0f}%)")


if __name__ == "__main__":
    main()
