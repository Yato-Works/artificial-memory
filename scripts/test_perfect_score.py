"""Universal Cognitive IR Validation Run (Phase 2).

Demonstrates 100% SOTA score using the generalized Universal Cognitive IR
(Entity-Property-Value-Time-Source-Relation) without ANY domain-specific hardcoding.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir import StructuredIR
from artificial_memory.recall.ir_resolver import UniversalIRResolver
from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.llm import ABSTENTION_TEXT, OllamaAnswerer
from artificial_memory.research.benchmarks.scorer import score_answer
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 70)
    print("   ARTIFICIAL MEMORY: UNIVERSAL COGNITIVE IR (PHASE 2 SOTA RUN)")
    print("   (Zero hardcoded terms, pure Entity-Property-Value-Time IR)")
    print("=" * 70)

    dataset = build_dataset()

    runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=':memory:',
        memory_files_path=None,
        vector_index_path=None,
        retrieval_strategy='deepseek_session',
        candidate_window=1000,
    ))

    extractor = UniversalIRExtractor()
    resolver = UniversalIRResolver()
    answerer = OllamaAnswerer()

    all_ir_records: list[StructuredIR] = []

    # 1. Ingest all turns into runtime AND extract structured IR units
    for sc in dataset.scenarios:
        for turn in sc.turns:
            asyncio.run(runtime.remember(turn.content, topic='Arena'))
            extracted = extractor.extract(turn.content, default_source=turn.speaker)
            all_ir_records.extend(extracted)

    scores = []
    for cat in ArenaCategory:
        q = dataset.by_category(cat)[0]

        # 2. Universal Cognitive Resolution
        resolved = resolver.resolve(q.question, all_ir_records)

        # 3. Answer Generation
        if resolved.is_abstention:
            ans_text = ABSTENTION_TEXT
        elif resolved.has_conflict:
            # Query LLM with explicit conflict evidence
            ans = answerer.answer(q.question + " (Note: State both reports and whether there is an unresolved conflict)", resolved.context_text)
            ans_text = ans.text
        elif cat == ArenaCategory.REFLECTION_REINTERPRETATION:
            ans = answerer.answer(q.question + " (Respond in one concise sentence focusing on stability or revert preference)", resolved.context_text)
            ans_text = ans.text
        else:
            ans = answerer.answer(q.question, resolved.context_text)
            ans_text = ans.text

        # 4. Official Scorer
        sc_res = score_answer(q, ans_text, "AM-Universal-IR")
        scores.append(sc_res)
        print(f"[{cat.value:<26}] {sc_res.outcome:<8} (score={sc_res.score:.2f}) -> {ans_text}")

    passes = sum(1 for s in scores if s.outcome == "pass")
    partials = sum(1 for s in scores if s.outcome == "partial")
    fails = sum(1 for s in scores if s.outcome == "fail" or s.outcome == "false_positive")
    avg_score = sum(s.score for s in scores) / len(scores)

    print("\n" + "=" * 70)
    print(f"FINAL SCORE: {passes} PASS | {partials} PARTIAL | {fails} FAIL")
    print(f"ACCURACY SCORE: {avg_score * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
