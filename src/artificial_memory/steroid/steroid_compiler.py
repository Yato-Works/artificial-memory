"""Steroid Context Compiler for AM Apex (Steroid Composition Layer).

Combines:
1. Wide Slicer (Multi-Channel Union: Semantic, Lexical, Entity, Temporal, Session)
2. Adaptive Graph Expander (Quality-Gated 1-to-3 Hop Expansion)
3. Overdrive Core Cognitive Runtime (TemporalResolver, PersonaStore, IntegrityGate, AnswerVerifier)

Maintains complete separation from the frozen baseline MinimumSufficientContextCompiler.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.core.ir.memory_types import (
    ApexMemoryUnit,
    CoverageCertificate,
    MemoryRole,
    ProofCarryingContext,
    QueryIntent,
)
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
from artificial_memory.steroid.adaptive_graph_expander import AdaptiveGraphExpander
from artificial_memory.steroid.wide_slicer import WideSlicer


class SteroidContextCompiler(MinimumSufficientContextCompiler):
    """Steroid-enhanced context compiler for AM Apex."""

    def __init__(self, per_channel_budget: int = 40, max_hops: int = 3) -> None:
        super().__init__()
        self.wide_slicer = WideSlicer(per_channel_budget=per_channel_budget)
        self.graph_expander = AdaptiveGraphExpander(max_hops=max_hops)

    def compile(
        self,
        query: str,
        records: Sequence[StructuredIR],
        target_token_budget: int = 240,
        weights: Optional[Any] = None,
        enabled_temporal_rules: Optional[set[str]] = None,
        reference_date_str: Optional[str] = None,
    ) -> ProofCarryingContext:
        """Compile context using Wide Slicing and Adaptive Graph Expansion."""
        # Step 1: Wide Slicing (Multi-channel union)
        slice_res = self.wide_slicer.slice(query, records, reference_date_str=reference_date_str)
        wide_candidates = slice_res.candidate_records

        # Step 2: Adaptive Graph Expansion (1-3 hops based on completeness)
        expansion_res = self.graph_expander.expand(query, wide_candidates, records)
        expanded_evidence = expansion_res.evidence_pool

        # Step 3: Overdrive Core Cognitive Planning & Graph
        plan = self.planner.plan(query)
        graph = UnifiedPropositionGraph()
        graph.build_from_records(expanded_evidence)

        # Step 4: Deterministic Groundings
        temporal_grounding = self.temporal_resolver.resolve(
            query,
            expanded_evidence,
            reference_date_str=reference_date_str,
        )

        self.persona_store.attributes.clear()
        self.persona_store.ingest_records(expanded_evidence)
        persona_grounding = self.persona_store.get_persona_grounding(query)

        state_resolution = self.state_engine.resolve(plan, graph)

        # Step 5: State Reconstruction on expanded evidence
        intent, candidate_units = self.reconstructor.reconstruct_world(query, expanded_evidence, weights=weights)

        # Adaptive search & integrity gate
        search_res = self.adaptive_searcher.search(plan, graph, candidate_units)
        integrity = self.integrity_gate.check(plan, search_res.selected_propositions)

        # Build final context text
        context_parts = []
        if temporal_grounding:
            context_parts.append(temporal_grounding)
        if persona_grounding:
            context_parts.append(persona_grounding)
        if state_resolution:
            context_parts.append(state_resolution)

        # Add top evidence units up to target_token_budget
        curr_tokens = sum(len(p.split()) for p in context_parts)
        for u in candidate_units:
            u_tok = len(u.ir.raw_content.split())
            if curr_tokens + u_tok > target_token_budget:
                break
            context_parts.append(u.ir.raw_content)
            curr_tokens += u_tok

        if not context_parts:
            context_parts.append("I don't know.")
            is_abstention = True
        else:
            is_abstention = False

        if integrity.recommended_abstention:
            context_parts.append(f"[Proposition Integrity Warning: {integrity.grounding_note}]")

        full_context = "\n".join(context_parts)
        cert = CoverageCertificate(
            is_sufficient=not is_abstention,
            entity_coverage=integrity.subject_matched,
            property_coverage=integrity.predicate_matched,
            temporal_coverage=True,
            conflict_coverage=True,
        )

        return ProofCarryingContext(
            context_text=full_context,
            certificate=cert,
            intent=intent,
            token_cost=len(full_context.split()),
            is_abstention=is_abstention,
        )
