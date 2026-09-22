"""Protein Context Compiler for AM Apex (Synthesis & Density Layer).

Coordinates the full Protein Phase pipeline:
    Query
      ↓
    Steroid Engine (Wide Slicing + Adaptive Graph Expansion)
      ↓
    Protein Core:
      1. Evidence Fuser (Deduplication & Entity Clustering)
      2. Temporal Supersession (E_i ≺ E_j State Ordering)
      3. Evidence Ranker (Multi-Objective Top 8-12 Selection)
      4. Context IR Compressor (Dense [STATE], [EVENT], [DATE])
      5. Provenance Tracker (Traceability & Audit Certificate)
      ↓
    Overdrive Core (Temporal Grounding, Persona, Integrity Gate, Answer Verifier)
      ↓
    High-Density Proof-Carrying Context (≤ 120 tokens/Q)
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Optional, Sequence

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.core.ir.memory_types import (
    CoverageCertificate,
    ProofCarryingContext,
)
from artificial_memory.core.ir.structured import IRStatus, StructuredIR
from artificial_memory.protein.chain_retention import GraphChainRetainer
from artificial_memory.protein.context_ir_compressor import ContextIRCompressor
from artificial_memory.protein.evidence_fuser import EvidenceFuser
from artificial_memory.protein.evidence_ranker import EvidenceRanker
from artificial_memory.protein.provenance_tracker import ProvenanceTracker
from artificial_memory.protein.state_compiler import StateCompiler
from artificial_memory.protein.state_synthesizer import StateSynthesizer
from artificial_memory.protein.state_trigger import StateTrigger
from artificial_memory.protein.temporal_supersession_protein import TemporalSupersessionProtein
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
from artificial_memory.steroid.adaptive_graph_expander import AdaptiveGraphExpander
from artificial_memory.steroid.wide_slicer import WideSlicer


class ContextPolicy(StrEnum):
    """Context compilation policy for AM Apex Protein Phase."""
    PRECISION = "precision"  # P4: Top-K natural dialogue turns (maximum accuracy)
    ADAPTIVE = "adaptive"    # Tiered natural + compressed IR (balanced Pareto)
    COMPACT = "compact"      # P6: High-density pure Context IR (sub-500ms, minimal tokens)


class ProteinContextCompiler(MinimumSufficientContextCompiler):
    """Synthesis & density context compiler for AM Apex."""

    def __init__(
        self,
        per_channel_budget: int = 40,
        max_hops: int = 3,
        top_k_evidence: int = 10,
        target_token_budget: int = 150,
        policy: ContextPolicy = ContextPolicy.PRECISION,
        ranker_weights: dict[str, float] | None = None,
        enable_dedup: bool = True,
        enable_entity_fusion: bool = True,
        enable_supersession: bool = True,
        enable_ranker: bool = True,
        enable_compression: bool = True,
        enable_chain_retention: bool = True,
        max_chain_reserve: int = 3,
        enable_state_synthesis: bool = False,
    ) -> None:
        super().__init__()
        # Steroid Layer
        self.wide_slicer = WideSlicer(per_channel_budget=per_channel_budget)
        self.graph_expander = AdaptiveGraphExpander(max_hops=max_hops)

        # Protein Layer
        self.fuser = EvidenceFuser()
        self.supersession_protein = TemporalSupersessionProtein()
        self.ranker = EvidenceRanker(top_k=top_k_evidence)
        self.chain_retainer = GraphChainRetainer(max_chain_reserve=max_chain_reserve)
        self.state_compiler = StateCompiler(max_states=2)
        self.state_trigger = StateTrigger()
        self.state_synthesizer = StateSynthesizer(max_states=2)
        from artificial_memory.temporal.temporal_compiler import TemporalCompiler
        self.temporal_compiler = TemporalCompiler()
        from artificial_memory.protein.session_fuser import SessionFuser
        self.session_fuser = SessionFuser()
        self.compressor = ContextIRCompressor()
        self.provenance = ProvenanceTracker()
        self.target_token_budget = target_token_budget
        self.policy = policy
        self.ranker_weights = ranker_weights or {}
        self.top_k_evidence = top_k_evidence

        # Feature flags for ablation / muscle-training stages
        self.enable_dedup = enable_dedup
        self.enable_entity_fusion = enable_entity_fusion
        self.enable_supersession = enable_supersession
        self.enable_ranker = enable_ranker
        self.enable_compression = enable_compression
        self.enable_chain_retention = enable_chain_retention
        self.max_chain_reserve = max_chain_reserve
        self.enable_state_synthesis = enable_state_synthesis

    def compile(
        self,
        query: str,
        records: Sequence[StructuredIR],
        target_token_budget: int = 150,
        weights: Optional[Any] = None,
        enabled_temporal_rules: Optional[set[str]] = None,
        reference_date_str: Optional[str] = None,
        policy: Optional[ContextPolicy] = None,
        ranker_weights: Optional[dict[str, float]] = None,
        top_k: Optional[int] = None,
        enable_state_synthesis: Optional[bool] = None,
    ) -> ProofCarryingContext:
        """Compile high-density Context IR using Steroid + Protein pipeline."""
        budget = target_token_budget or self.target_token_budget
        active_policy = policy or self.policy
        k_val = top_k if top_k is not None else self.top_k_evidence
        w_dict = ranker_weights if ranker_weights is not None else self.ranker_weights

        # At large scale, perform deterministic shard selection before any
        # graph/ranker stage.  This keeps every downstream component bounded
        # while the index retains adjacent turns for conversational anaphora.
        working_records = self._working_records(query, records)

        # 1. Steroid Layer: Wide Slicing & Adaptive Graph Expansion
        slice_res = self.wide_slicer.slice(query, working_records, reference_date_str=reference_date_str)
        exp_res = self.graph_expander.expand(query, slice_res.candidate_records, working_records)
        raw_evidence = exp_res.evidence_pool

        # 2. Protein Layer: Deduplication & Fusion
        if self.enable_dedup and self.enable_entity_fusion:
            fused_records = self.fuser.fuse(query, raw_evidence)
        elif self.enable_dedup:
            fused_records = self.fuser.deduplicate_only(raw_evidence)
        else:
            fused_records = list(raw_evidence)

        # 3. Protein Layer: Temporal Supersession Ordering (E_i < E_j)
        if self.enable_supersession:
            active_records, superseded_groups = self.supersession_protein.resolve(query, fused_records)
        else:
            active_records, superseded_groups = fused_records, []

        # 4. Protein Layer: State Reconstruction & Ranking on Active Records
        if self.enable_ranker:
            ranked_evidence = self.ranker.rank(query, active_records, weights=w_dict, top_k=k_val)
            top_records = [item.record for item in ranked_evidence]
        else:
            top_records = active_records[:k_val]

        # 4b. Protein Layer: Graph Chain Retention (Multi-Hop Preservation)
        if self.enable_chain_retention:
            top_records = self.chain_retainer.retain_chains(
                query,
                top_records=top_records,
                candidate_records=active_records,
                max_reserve=self.max_chain_reserve,
            )

        # 5. Overdrive Core: Planning & Deterministic Groundings
        plan = self.planner.plan(query)
        graph = UnifiedPropositionGraph()
        graph.build_from_records(top_records)

        temporal_grounding = self.temporal_resolver.resolve(
            query,
            top_records,
            reference_date_str=reference_date_str,
        )

        self.persona_store.attributes.clear()
        self.persona_store.ingest_records(working_records)
        persona_grounding = self.persona_store.get_persona_grounding(query)

        # 6. Protein Layer: Context Formatting / Compression based on ContextPolicy
        q_lower = query.lower()
        top_entities = {r.entity.lower() for r in top_records if r.entity}
        relevant_groups = [
            sg for sg in superseded_groups
            if sg.entity.lower() in top_entities or sg.entity.lower() in q_lower or sg.property.lower() in q_lower
        ]

        if not self.enable_compression or active_policy == ContextPolicy.PRECISION:
            # P4: Full natural dialogue turns (Chronological ordering for temporal progression)
            def _get_rec_date(r: StructuredIR) -> float:
                m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", r.time_scope or r.raw_content or "")
                if m:
                    return int(m.group(1)) * 365.0 + int(m.group(2)) * 30.0 + int(m.group(3))
                return 0.0

            ordered_records = sorted(top_records, key=_get_rec_date)
            raw_lines = []
            for r in ordered_records:
                content = (r.raw_content or "").strip()
                m_id = re.search(r"\[(D\d+:\d+|\d+)", content)
                src_id = f"[{m_id.group(1)}] " if m_id else ""

                prefix = ""
                if r.status == IRStatus.SUPERSEDED:
                    prefix = "[SUPERSEDED / FORMER STATE (DO NOT USE)] "
                elif r.status == IRStatus.ACTIVE and any(r == sg.current_record for sg in relevant_groups):
                    prefix = "[CURRENT ACTIVE STATE (LATEST)] "

                if content:
                    base_line = content if content.startswith("[") else f"{src_id}{content}"
                    raw_lines.append(f"{prefix}{base_line}")
                elif r.entity and r.property and r.value:
                    raw_lines.append(f"{prefix}{src_id}{r.entity} -> {r.property}: {r.value}")
            compressed_ir = "\n".join(raw_lines)
        elif active_policy == ContextPolicy.ADAPTIVE:
            # Adaptive tiered IR
            compressed_ir = self.compressor.compress_adaptive(
                top_records,
                superseded_groups=relevant_groups,
                target_token_budget=budget,
            )
        else:
            # COMPACT (P6)
            compressed_ir = self.compressor.compress(
                top_records,
                superseded_groups=relevant_groups,
                target_token_budget=budget,
            )

        # 6b. State Compilation (Cognitive State Shortcut with Provenance)
        active_state_syn = enable_state_synthesis if enable_state_synthesis is not None else self.enable_state_synthesis
        synthesized_states = []
        if active_state_syn:
            trigger_dec = self.state_trigger.evaluate(query, top_records)
            if trigger_dec.should_synthesize:
                synthesized_states = self.state_compiler.compile_states(query, working_records)

        # 6c. Temporal State Compilation (Phase CHRONOS)
        temporal_states = []
        if self.temporal_compiler.is_temporal_query(query):
            temporal_states = self.temporal_compiler.compile_temporal_states(query, working_records)

        # 6d. Explicit Supersession State Certificates
        supersession_states = []
        for sg in relevant_groups:
            c_val = sg.current_record.value or sg.current_record.raw_content
            s_val = sg.superseded_records[-1].value or sg.superseded_records[-1].raw_content
            ent_name = sg.entity.capitalize() if sg.entity else "The user"
            supersession_states.append(
                f"[STATE UPDATE: For {ent_name}'s {sg.property}, the CURRENT ACTIVE STATE is \"{c_val}\" (supersedes previous state \"{s_val}\").]"
            )

        # 6e. Cross-Session Aggregation (Phase SESSION FUSION)
        aggregation_cert = None
        if self.session_fuser.is_aggregation_query(query):
            agg_res = self.session_fuser.fuse(query, top_records)
            if agg_res.certificate:
                aggregation_cert = agg_res.certificate

        # 7. Assemble Proof-Carrying Context
        context_parts = []
        if aggregation_cert:
            context_parts.append(aggregation_cert)
        if supersession_states:
            for s_st in supersession_states:
                context_parts.append(s_st)
        if synthesized_states:
            for st in synthesized_states:
                context_parts.append(st.format_state())
        if temporal_states:
            for t_st in temporal_states:
                context_parts.append(t_st.format_state())
        if temporal_grounding:
            t_text = temporal_grounding.grounding_text if hasattr(temporal_grounding, "grounding_text") else str(temporal_grounding)
            context_parts.append(t_text)
        if persona_grounding:
            context_parts.append(persona_grounding)
        if compressed_ir:
            context_parts.append(compressed_ir)

        # Proposition integrity check across all propositions in graph
        search_res = self.adaptive_searcher.search(plan, graph, [])
        integrity = self.integrity_gate.check(plan, list(graph.propositions.values()))
        if integrity.recommended_abstention and integrity.grounding_note:
            context_parts.append(integrity.grounding_note)

        if not context_parts:
            context_parts.append("I don't know.")
            is_abstention = True
        else:
            is_abstention = False

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
            intent=plan.intent,
            token_cost=len(full_context.split()),
            is_abstention=is_abstention,
        )
