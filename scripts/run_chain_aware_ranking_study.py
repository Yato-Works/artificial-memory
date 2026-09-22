"""Chain-Aware Ranking & Dynamic Budget Study (Phase C2/C3).

Tests:
1. Baseline P4: K=10, Chain OFF
2. Chain-Aware Append (C1): K=10 + Reserve 3 (465 tokens)
3. In-Budget Chain-Aware Re-ranking (C3): K=10, S_chain boosted into ranker (Budget strictly 10 turns)
4. Dynamic Chain Reserve (C2): K=8 base + up to 2 chain turns (Budget strictly <= 10 turns)

Evaluates on the 32 Multi-Hop questions (Category 1) of LoCoMo-10 Conversation 0.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Sequence

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.protein.chain_retention import GraphChainRetainer
from artificial_memory.protein.evidence_ranker import EvidenceRanker
from artificial_memory.protein.protein_compiler import ContextPolicy, ProteinContextCompiler
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ChainAwareInBudgetCompiler(ProteinContextCompiler):
    """Compiler that integrates S_chain directly into ranking to maintain strictly K=10."""

    def __init__(self, top_k: int = 10, chain_boost_weight: float = 1.5, **kwargs):
        super().__init__(top_k_evidence=top_k, enable_chain_retention=False, **kwargs)
        self.chain_boost_weight = chain_boost_weight
        self.retainer = GraphChainRetainer(max_chain_reserve=3)

    def compile(self, query: str, records: Sequence[StructuredIR], **kwargs):
        # 1. Steroid Layer
        slice_res = self.wide_slicer.slice(query, records)
        exp_res = self.graph_expander.expand(query, slice_res.candidate_records, records)
        raw_evidence = exp_res.evidence_pool

        # 2. Protein: Dedup & Fusion
        fused_records = self.fuser.fuse(query, raw_evidence)
        active_records, _ = self.supersession_protein.resolve(query, fused_records)

        # 3. Base Ranking (get top 5 seed anchors)
        base_ranked = self.ranker.rank(query, active_records, top_k=len(active_records))
        top_seeds = [item.record for item in base_ranked[:4]]

        # 4. Chain Scoring with respect to top seeds
        top_ids = {id(r) for r in top_seeds}
        top_entities = {r.entity.lower().strip() for r in top_seeds if r.entity and len(r.entity) > 1}
        top_values = {
            r.value.lower().strip()
            for r in top_seeds
            if r.value and len(r.value) > 2 and r.value.lower() not in ["true", "false", "yes", "no", "unknown", "none"]
        }

        # Boost records that connect to top seeds
        boosted_ranked = []
        for item in base_ranked:
            rec = item.record
            c_ent = (rec.entity or "").lower().strip()
            c_val = (rec.value or "").lower().strip()
            c_content = (rec.raw_content or "").lower()

            chain_score = 0.0
            if id(rec) not in top_ids:
                # Forward hop
                if top_values & {c_ent}:
                    chain_score += 15.0
                # Backward hop
                if top_entities & {c_val}:
                    chain_score += 12.0
                # Co-occurrence
                if any(e in c_content for e in top_entities):
                    chain_score += 6.0

            total_score = item.score + self.chain_boost_weight * chain_score
            boosted_ranked.append((total_score, rec))

        boosted_ranked.sort(key=lambda x: -x[0])
        # Strictly select Top K=10!
        top_records = [rec for _, rec in boosted_ranked[:self.top_k_evidence]]

        # Format context (Precision mode)
        context_parts = [r.raw_content for r in top_records if r.raw_content]
        context_text = "\n".join(context_parts)

        from artificial_memory.core.ir.memory_types import CoverageCertificate, ProofCarryingContext, QueryIntent
        return ProofCarryingContext(
            context_text=context_text,
            certificate=CoverageCertificate(
                is_sufficient=True,
                entity_coverage=True,
            ),
            intent=QueryIntent.STATE_QUERY,
            token_cost=len(context_text.split()),
            is_abstention=not bool(context_parts),
        )


def main():
    print("=" * 85)
    print("      AM APEX PHASE C3: IN-BUDGET CHAIN-AWARE RE-RANKING STUDY")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer()

    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Targeting {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    # Condition C3: In-Budget Chain-Aware Re-ranking (Strictly K=10)
    compiler_c3 = ChainAwareInBudgetCompiler(top_k=10, chain_boost_weight=1.5)
    adapter.compiler = compiler_c3

    results_c3 = []
    t0 = time.perf_counter()
    for i, q in enumerate(multihop_questions):
        res = adapter.evaluate_question(q, turns, ir_records, answerer)
        results_c3.append(res)
        status = "PASS" if res.is_correct else "FAIL"
        ora = "O" if res.oracle_recall else "X"
        print(f"  [C3-InBudget {i+1:02d}/32] {status} (Ora:{ora}) | {res.tokens_used:3d} tok | Q: {q.question[:40]}")
    elapsed = time.perf_counter() - t0

    acc = sum(1 for r in results_c3 if r.is_correct) / len(results_c3) * 100
    ora = sum(1 for r in results_c3 if r.oracle_recall) / len(results_c3) * 100
    tok = sum(r.tokens_used for r in results_c3) / len(results_c3)

    print("\n" + "=" * 85)
    print("      C3 IN-BUDGET CHAIN RE-RANKING RESULTS (32 QUESTIONS)")
    print("=" * 85)
    print(f"  * Multi-Hop Accuracy:  {acc:5.1f}% ({sum(1 for r in results_c3 if r.is_correct):02d}/32)")
    print(f"  * Oracle Recall:       {ora:5.1f}% ({sum(1 for r in results_c3 if r.oracle_recall):02d}/32)")
    print(f"  * Mean Tokens/Q:       {tok:5.1f} tok")
    print(f"  * Elapsed Time:        {elapsed:.1f} s")
    print("=" * 85)

    out_file = RESULTS_DIR / "chain_in_budget_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "oracle_recall": ora,
                "mean_tokens": tok,
                "elapsed_s": elapsed,
                "results": [
                    {
                        "question_id": q.question_id,
                        "question": q.question,
                        "ground_truth": q.ground_truth,
                        "pred": r.predicted_answer,
                        "is_correct": r.is_correct,
                        "oracle_recall": r.oracle_recall,
                        "tokens": r.tokens_used,
                    }
                    for q, r in zip(multihop_questions, results_c3)
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
