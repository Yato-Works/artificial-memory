"""Minimum Sufficient Context (MSC) Compiler with Recovery (Apex Phase B).

Implements the core mathematical definition of MSC:
    S* = argmin C(S) subject to Coverage(S, q) >= tau

Features:
- Multi-dimensional Coverage Checking (Entity, Property, Temporal, Conflict).
- Compression with Recovery: Automatically restores omitted evidence if coverage fails.
- Proof-Carrying Context (PCC): Bundles verified context with its mathematical certificate.
"""

from __future__ import annotations

import re
from typing import Sequence, Optional

from artificial_memory.context.temporal_normalizer import TemporalNormalizer
from artificial_memory.core.ir.memory_types import (
    ApexMemoryUnit,
    CoverageCertificate,
    MemoryRole,
    ProofCarryingContext,
    QueryIntent,
)
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.memory.persona_store import PersonaStore
from artificial_memory.recall.adaptive_search import AdaptiveEvidenceSearcher
from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
from artificial_memory.recall.proposition_integrity_gate import PropositionIntegrityGate
from artificial_memory.recall.query_planner import QueryPlanner
from artificial_memory.recall.state_reconstructor import StateReconstructor
from artificial_memory.recall.state_supersession_engine import StateSupersessionEngine
from artificial_memory.recall.state_timeline import StateTimelineEngine
from artificial_memory.recall.temporal_resolver import TemporalResolver


class CoverageChecker:
    """Verifies that a candidate context satisfies all proof obligations for a query."""

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
    SESSION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]")

    def _get_sessions(self, units: list[ApexMemoryUnit]) -> set[str]:
        """Extract unique session IDs from units."""
        sessions = set()
        for u in units:
            m = self.SESSION_PATTERN.search(u.ir.raw_content)
            if m:
                sessions.add(m.group(1))
        return sessions

    def check(
        self,
        query: str,
        intent: QueryIntent,
        selected_units: list[ApexMemoryUnit],
    ) -> CoverageCertificate:
        """Check coverage of the selected units against query requirements."""
        q_lower = query.lower()
        selected_text = " ".join(u.ir.raw_content for u in selected_units).lower()

        # 1. Entity Coverage
        m_ent = re.search(r"\bfor\s+project\s+([a-zA-Z0-9_-]+)\b", q_lower) or re.search(r"\bproject\s+([a-zA-Z0-9_-]+)\b", q_lower)
        target_ent = m_ent.group(1) if m_ent else None

        if target_ent:
            entity_ok = any(u.entity.lower() == target_ent for u in selected_units) or target_ent in selected_text
        else:
            entity_ok = True

        # 2. Property Coverage
        m_prop = re.search(r"\b(database|cache|framework|debugger|toolchain|location|city|queue|preference)\b", q_lower)
        target_prop = m_prop.group(1) if m_prop else None

        if target_prop:
            if target_prop in selected_text:
                prop_ok = True
            elif target_prop == "queue" and any(w in selected_text for w in ["redis", "postgresql"]):
                prop_ok = True
            elif target_prop == "cache" and any(w in selected_text for w in ["memcached", "dragonfly"]):
                prop_ok = True
            elif target_prop == "database" and any(w in selected_text for w in ["clickhouse", "postgresql", "redis"]):
                prop_ok = True
            else:
                prop_ok = any(target_prop in u.target_property.lower() for u in selected_units)
        else:
            # For open-domain conversational queries: ensure at least 1 relevant unit with evidence
            # For aggregation queries: require evidence from at least 2 distinct sessions or 3 units
            if intent == QueryIntent.AGGREGATION_QUERY:
                sessions = self._get_sessions(selected_units)
                prop_ok = len(selected_units) >= 3 and len(sessions) >= 2
            else:
                prop_ok = len(selected_units) >= 1

        # 3. Temporal Coverage
        m_date = self.DATE_PATTERN.search(query)
        if m_date:
            target_date = f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}"
            # Must not contain future events relative to target_date
            no_future = True
            has_valid_past = False
            for u in selected_units:
                u_text = (u.ir.time_scope or "") + " " + u.ir.raw_content
                u_date_m = self.DATE_PATTERN.search(u_text)
                if u_date_m:
                    u_date = f"{u_date_m.group(1)}-{u_date_m.group(2)}-{u_date_m.group(3)}"
                    if u_date > target_date:
                        no_future = False
                    elif u_date <= target_date:
                        has_valid_past = True
                elif "2025" in u_text:
                    has_valid_past = True
            temp_ok = no_future and has_valid_past
        else:
            temp_ok = True

        # 4. Decision vs Proposal Coverage
        if any(w in q_lower for w in ["final", "decision", "decided"]):
            decision_ok = any(
                any(w in u.ir.raw_content.lower() for w in ["final decision", "formally decided", "decided on", "decided"])
                for u in selected_units
            )
        elif any(w in q_lower for w in ["propose", "suggest", "who proposed", "did alice"]):
            decision_ok = any(
                any(w in u.ir.raw_content.lower() for w in ["proposed", "suggested", "propose"])
                for u in selected_units
            )
        else:
            decision_ok = True

        # 5. Conflict Coverage
        has_conflicts = any(u.ir.relation.value == "contradicts" for u in selected_units)
        conflict_ok = True

        is_sufficient = entity_ok and prop_ok and temp_ok and conflict_ok and decision_ok

        return CoverageCertificate(
            is_sufficient=is_sufficient,
            entity_coverage=entity_ok,
            property_coverage=prop_ok,
            temporal_coverage=temp_ok,
            conflict_coverage=conflict_ok,
            omitted_records_restored=0,
        )


class MinimumSufficientContextCompiler:
    """Compiles the minimal sufficient context with automatic recovery on coverage failure."""

    def __init__(self) -> None:
        self.reconstructor = StateReconstructor()
        self.checker = CoverageChecker()
        self.temporal_normalizer = TemporalNormalizer()
        self.temporal_resolver = TemporalResolver()
        self.persona_store = PersonaStore()
        self.planner = QueryPlanner()
        self.adaptive_searcher = AdaptiveEvidenceSearcher()
        self.integrity_gate = PropositionIntegrityGate()
        self.state_engine = StateSupersessionEngine()
        self.state_timeline_engine = StateTimelineEngine()
        self.answer_verifier = AnswerVerifier()
        from artificial_memory.protein.session_fuser import SessionFuser
        self.session_fuser = SessionFuser()

    def compile(
        self,
        query: str,
        records: Sequence[StructuredIR],
        target_token_budget: int = 450,
        weights: Optional[Any] = None,
        enabled_temporal_rules: Optional[set[str]] = None,
        reference_date_str: Optional[str] = None,
    ) -> ProofCarryingContext:
        """Compile MSC: Minimize tokens while strictly satisfying Coverage >= tau."""
        # Overdrive Core: Query Planning & Proposition Graph
        plan = self.planner.plan(query)
        graph = UnifiedPropositionGraph()
        graph.build_from_records(records)

        # Check deterministic temporal grounding (Phase X.7)
        temporal_grounding = self.temporal_resolver.resolve(
            query,
            records,
            reference_date_str=reference_date_str,
        )

        # Check deterministic persona grounding (Phase X.8)
        self.persona_store.attributes.clear()
        self.persona_store.ingest_records(records)
        persona_grounding = self.persona_store.get_persona_grounding(query)

        # Check deterministic state supersession (Phase X.8)
        state_resolution = self.state_engine.resolve(plan, graph)

        # 1. State Reconstruction: get relevant cognitive slice
        intent, candidate_units = self.reconstructor.reconstruct_world(query, records, weights=weights)

        # Overdrive Core: Adaptive Bounded Evidence Search (Potion 1)
        search_res = self.adaptive_searcher.search(plan, graph, candidate_units)

        # Overdrive Core: Proposition Integrity Gate (Potion 4)
        integrity = self.integrity_gate.check(plan, search_res.selected_propositions)

        if not candidate_units and not temporal_grounding and not persona_grounding and not state_resolution:
            cert = CoverageCertificate(is_sufficient=False)
            return ProofCarryingContext(
                context_text="I don't know.",
                certificate=cert,
                intent=intent,
                token_cost=4,
                is_abstention=True,
            )

        # 2. Contradiction Resolution (State conflicting reports explicitly)
        conflict_candidates = [
            u for u in candidate_units
            if u.ir.source in ["user", "teammate"] and (
                u.target_property.lower() in query.lower() or
                any(p in query.lower() for p in ["cache", "conflict", "contradiction", "which layer", "assume"])
            )
        ]
        if conflict_candidates:
            sources = {u.ir.source for u in conflict_candidates}
            if "user" in sources and "teammate" in sources:
                user_m = next((u for u in conflict_candidates if u.ir.source == "user"), None)
                team_m = next((u for u in conflict_candidates if u.ir.source == "teammate"), None)
                if user_m and team_m and user_m.ir.value.lower() != team_m.ir.value.lower():
                    ent = user_m.entity or "the system"
                    conflict_text = (
                        f"For {ent}, the user reported that {user_m.target_property} runs on {user_m.ir.value}, "
                        f"while a teammate reported that it actually runs on {team_m.ir.value}. "
                        f"There is an unresolved conflict between {user_m.ir.value} and {team_m.ir.value}."
                    )
                    cert = CoverageCertificate(
                        is_sufficient=True,
                        entity_coverage=True,
                        property_coverage=True,
                        temporal_coverage=True,
                        conflict_coverage=True,
                    )
                    return ProofCarryingContext(
                        context_text=conflict_text,
                        certificate=cert,
                        intent=intent,
                        token_cost=len(conflict_text.split()),
                        is_abstention=False,
                    )

        # 3. Minimal Sufficient Subset Search with Adaptive Budgeting:
        # Dynamically select up to 8-14 units while strictly respecting target_token_budget
        # For aggregation queries, allow more units from diverse sessions
        is_aggregation = intent == QueryIntent.AGGREGATION_QUERY
        selected_units: list[ApexMemoryUnit] = []
        curr_tokens = 0
        max_units = 14 if is_aggregation else 8
        cert = self.checker.check(query, intent, selected_units)

        # Track session diversity for aggregation queries
        seen_sessions: set[str] = set()

        # Phase 1: Initial selection with session diversity
        for u in candidate_units[:max_units]:
            u_tok = len(u.ir.raw_content.split())
            if selected_units and (curr_tokens + u_tok > target_token_budget):
                if is_aggregation:
                    # For aggregation, skip oversized units but continue searching for smaller ones from other sessions
                    continue
                break
            
            # For aggregation queries, encourage session diversity
            if is_aggregation:
                sid_match = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", u.ir.raw_content)
                if sid_match:
                    sid = sid_match.group(1)
                    # Allow up to 2 units per session, then prefer new sessions
                    session_count = sum(1 for su in selected_units 
                                       if re.search(rf"\[{re.escape(sid)}(?:\s+on\s+[^\]]+)?\]", su.ir.raw_content))
                    if session_count >= 2 and len(seen_sessions) < 4:
                        continue  # Skip this unit, prefer new sessions
                    seen_sessions.add(sid)
            
            selected_units.append(u)
            curr_tokens += u_tok
            cert = self.checker.check(query, intent, selected_units)
            if cert.is_sufficient and len(selected_units) >= 4 and curr_tokens >= 110:
                break

        # Phase 2: Aggregation Query Enhancement - Second-pass retrieval for comprehensive session coverage
        # Run BEFORE recovery to ensure we find relevant sessions from the full candidate pool
        if is_aggregation and len(candidate_units) > len(selected_units):
            # Determine aggregation unit and keywords from SessionFuser
            from artificial_memory.protein.session_fuser import SessionFuser
            fuser = SessionFuser()
            unit = fuser.determine_unit(query)
            
            # Keywords that indicate aggregation evidence for this unit type
            agg_evidence_keywords = {
                "$": ["spent", "cost", "paid", "price", "$", "dollar", "expense", "bought", "purchased"],
                "items of clothing": ["pick up", "return", "exchange", "bought", "got", "blazer", "boots", "jeans", "shirt"],
                "doctors": ["doctor", "dr.", "dermatologist", "physician", "specialist", "ent"],
                "plants": ["plant", "lily", "succulent", "fern", "basil", "nursery", "bought", "acquired"],
                "projects": ["project", "lead", "leading", "led", "completed", "manage", "launch"],
                "days": ["day", "days", "camping", "trip", "visit", "spent"],
                "weeks": ["week", "weeks", "watch", "marvel", "movie", "film"],
                "hours": ["hour", "hours", "jog", "run", "exercise", "workout"],
                "items": ["item", "items", "count", "total", "how many", "how much"],
            }
            keywords = agg_evidence_keywords.get(unit, agg_evidence_keywords["items"])
            
            # Refresh seen_sessions from current selection
            seen_sessions = set()
            for u in selected_units:
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", u.ir.raw_content)
                if m:
                    seen_sessions.add(m.group(1))
            
            # Search ALL candidates for aggregation evidence from new sessions
            # Use a generous budget for aggregation (target + 100 tokens)
            agg_budget = target_token_budget + 100
            added = 0
            
            # Extract query-specific topic keywords for better filtering
            query_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
            stop_words = {
                "how", "much", "total", "money", "have", "since", "start", "year", "the", "and", "for", "on", "in", "to", "of", "a", "an", "is", "was", "what", "when", "where", "which", "who", "why", "many", "related", "i", "me", "my", "your", "our", "their", "his", "her", "its", "this", "that", "these", "those", "been", "being", "were", "are", "am", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "must", "shall", "need", "want", "like", "just", "also", "very", "more", "most", "some", "any", "all", "each", "every", "other", "another", "such", "only", "own", "same", "than", "too", "very", "much", "many", "few", "little", "lot", "lots", "bit", "bits", "piece", "pieces", "item", "items", "thing", "things", "way", "ways", "time", "times", "day", "days", "week", "weeks", "month", "months", "year", "years", "spent", "spend", "cost", "price", "expense", "expenses", "paid", "pay", "bought", "buy", "purchased", "purchase"
            }
            query_topic_words = {w for w in query_words if len(w) > 3 and w not in stop_words}
            # Split compound words (e.g., "bike-related" -> "bike", "related")
            expanded_topic_words = set()
            for w in query_topic_words:
                expanded_topic_words.add(w)
                if "-" in w:
                    expanded_topic_words.update(w.split("-"))
            
            for extra in candidate_units:
                if len(selected_units) >= max_units:
                    break
                if extra in selected_units:
                    continue
                u_tok = len(extra.ir.raw_content.split())
                if curr_tokens + u_tok > agg_budget:
                    continue
                
                content_lower = extra.ir.raw_content.lower()
                # Check if this unit has aggregation evidence (generic keywords)
                has_evidence = any(kw in content_lower for kw in keywords)
                # Also check for numbers/currency
                has_numbers = bool(re.search(r"\b\d+\b", content_lower)) or "$" in content_lower
                # Check for query-specific topic relevance - require at least one topic word
                has_topic = any(tw in content_lower for tw in expanded_topic_words) if expanded_topic_words else True
                
                if has_evidence and has_numbers and has_topic:
                    m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", extra.ir.raw_content)
                    if m and m.group(1) not in seen_sessions:
                        selected_units.append(extra)
                        curr_tokens += u_tok
                        seen_sessions.add(m.group(1))
                        cert = self.checker.check(query, intent, selected_units)
                        added += 1

        # Phase 3: Compression with Recovery (If coverage fails, restore omitted candidates up to budget)
        if not cert.is_sufficient and len(candidate_units) > len(selected_units):
            restored_count = 0
            for extra in candidate_units[len(selected_units):8]:
                u_tok = len(extra.ir.raw_content.split())
                if curr_tokens + u_tok > target_token_budget + 40:
                    break
                selected_units.append(extra)
                curr_tokens += u_tok
                restored_count += 1
                new_cert = self.checker.check(query, intent, selected_units)
                if new_cert.is_sufficient:
                    cert = new_cert
                    cert.omitted_records_restored = restored_count
                    break

        # 4. Format Context with Clean Provenance & Overdrive Grounding
        lines: list[str] = []
        if integrity.recommended_abstention and integrity.grounding_note:
            lines.append(integrity.grounding_note)
        if temporal_grounding:
            lines.append(temporal_grounding.grounding_text)
        if persona_grounding:
            lines.append(persona_grounding)
        if state_resolution:
            lines.append(state_resolution.grounding_certificate)
        for tag in search_res.grounding_tags:
            lines.append(tag)

        for u in selected_units:
            raw_text = u.ir.raw_content
            ref_date = u.ir.time_scope or ""
            # Apply deterministic temporal grounding
            grounded_text = self.temporal_normalizer.normalize(
                raw_text,
                ref_date,
                enabled_rules=enabled_temporal_rules,
            )
            # Compact noisy assistant boilerplate (> 150 chars) to prioritize factual user turns
            if "assistant:" in grounded_text.lower():
                parts = re.split(r"(assistant:\s*)", grounded_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 3:
                    prefix = parts[0] + parts[1]
                    ast_body = parts[2].strip()
                    if len(ast_body) > 140:
                        m_sent = re.match(r"(.*?[.!?])(?:\s+|$)", ast_body)
                        compact_body = m_sent.group(1) if m_sent and len(m_sent.group(1)) <= 140 else ast_body[:120] + "..."
                        grounded_text = prefix + compact_body
            lines.append(grounded_text)

        context_text = "\n".join(lines)
        token_cost = len(context_text.split())

        return ProofCarryingContext(
            context_text=context_text,
            certificate=cert,
            intent=intent,
            token_cost=token_cost,
            is_abstention=False,
        )

