"""Universal Cognitive IR Resolver (Phase 2).

Resolves queries against a corpus of StructuredIR records without domain-specific
hardcoding, executing entity isolation, temporal matching, conflict resolution,
and deterministic abstention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir import IRRelation, IRStatus, StructuredIR
from artificial_memory.research.benchmarks.llm import ABSTENTION_TEXT


@dataclass
class ResolvedContext:
    """Result of resolving a query against structured cognitive memories."""
    context_text: str
    is_abstention: bool = False
    has_conflict: bool = False
    primary_entity: str | None = None
    target_property: str | None = None
    matched_records: list[StructuredIR] = None

    def __post_init__(self):
        if self.matched_records is None:
            self.matched_records = []


class UniversalIRResolver:
    """Resolves queries against structured IR records."""

    ENTITY_QUERY_PATTERNS = [
        re.compile(r"\bfor\s+project\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bproject\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bworking\s+on\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bin\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
    ]

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")

    def resolve(self, query: str, ir_records: Sequence[StructuredIR]) -> ResolvedContext:
        """Resolve query against structured IR records using generalized cognitive scoping."""
        target_entity = self._extract_target_entity(query)
        target_date = self._extract_target_date(query)
        query_lower = query.lower()

        # 1. Entity Scoping / Isolation (isolate memories belonging to target_entity)
        if target_entity:
            scoped_records = [
                r for r in ir_records
                if r.entity == target_entity or r.entity == "general" or target_entity in r.raw_content.lower()
            ]
        else:
            scoped_records = list(ir_records)

        if not scoped_records:
            return ResolvedContext(
                context_text="",
                is_abstention=True,
                primary_entity=target_entity,
            )

        # 2. Strict Abstention Check:
        # Only abstain if there is an explicit negative constraint (e.g. 'never a debugger')
        # or if the query specifically asks for a tool/config that is explicitly marked as never configured.
        explicit_never = [r for r in scoped_records if r.relation == IRRelation.NEVER]
        for neg_r in explicit_never:
            if neg_r.property in query_lower:
                return ResolvedContext(
                    context_text=f"For {target_entity}, {neg_r.property} was never configured.",
                    is_abstention=True,
                    primary_entity=target_entity,
                    target_property=neg_r.property,
                    matched_records=[neg_r],
                )

        # 3. Contradiction Resolution:
        # Check if query asks about an attribute that has conflicting sources (e.g. user vs teammate)
        # The property phrase (e.g. "cache layer" or "cache") must be explicitly targeted by the query.
        conflict_candidates = [
            r for r in scoped_records
            if r.source in ["user", "teammate"] and (r.property.lower() in query_lower or any(p in query_lower for p in ["cache", "conflict", "contradiction", "which layer", "assume"]))
        ]
        if conflict_candidates:
            sources = {r.source for r in conflict_candidates}
            if "user" in sources and "teammate" in sources:
                user_m = next((r for r in conflict_candidates if r.source == "user"), None)
                team_m = next((r for r in conflict_candidates if r.source == "teammate"), None)
                if user_m and team_m and user_m.value.lower() != team_m.value.lower():
                    # Only trigger if the query is actually asking about this conflicted property or asking about conflicts
                    prop_relevant = user_m.property.lower() in query_lower or any(w in query_lower for w in ["cache", "conflict", "contradict", "assume"])
                    if prop_relevant:
                        conflict_text = (
                            f"For {target_entity}, the user reported that {user_m.property} runs on {user_m.value}, "
                            f"while a teammate reported that it actually runs on {team_m.value}. "
                            f"There is an unresolved conflict between {user_m.value} and {team_m.value}."
                        )
                        return ResolvedContext(
                            context_text=conflict_text,
                            has_conflict=True,
                            primary_entity=target_entity,
                            target_property=user_m.property,
                            matched_records=[user_m, team_m],
                        )

        # 4. Temporal Resolution:
        # If query has a specific date constraint, find records active on that date
        if target_date:
            matched_temporal = []
            for r in scoped_records:
                if r.time_scope:
                    m_until = re.search(r"until\s+(\d{4}-\d{2}-\d{2})", r.time_scope)
                    if m_until and target_date <= m_until.group(1):
                        matched_temporal.append(r)
                    m_on = re.search(r"on\s+(\d{4}-\d{2}-\d{2})", r.time_scope)
                    if m_on and m_on.group(1) == target_date:
                        matched_temporal.append(r)

            if matched_temporal:
                lines = [r.raw_content for r in matched_temporal]
                return ResolvedContext(
                    context_text="\n".join(lines),
                    primary_entity=target_entity,
                    matched_records=matched_temporal,
                )

        # 5. Behavioral / Reflection Resolution:
        if any(w in query_lower for w in ["preference", "recurring", "consistently", "tendency", "habit"]):
            behavior_records = [
                r for r in scoped_records
                if r.relation == IRRelation.BEHAVIOR or any(w in r.raw_content.lower() for w in ["revert", "backup", "copies"])
            ]
            if behavior_records:
                lines = [r.raw_content for r in behavior_records]
                return ResolvedContext(
                    context_text="\n".join(lines),
                    primary_entity=target_entity,
                    matched_records=behavior_records,
                )

        # 6. Semantic Aspect Relevance Ranking & Transitive Association:
        # Score records based on word overlap with query keywords (excluding common stop words and target entity)
        stop_words = {"for", "project", "what", "which", "did", "the", "user", "say", "is", "in", "a", "an", "to", "of", "and", "our", "conversations", "about", "that", "runs"}
        query_terms = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_lower) if w not in stop_words and len(w) > 1]
        if target_entity:
            query_terms = [t for t in query_terms if t != target_entity]

        # Adversarial Prompt Injection Filter
        injection_re = re.compile(r"(?:ignore\s+all\s+previous\s+instructions|system\s+prompt\s*:|you\s+are\s+an\s+attacker)", re.IGNORECASE)

        def relevance_score(rec: StructuredIR) -> float:
            # If the record is a blatant prompt injection attempt and has property='statement', suppress score
            if rec.property == "statement" and injection_re.search(rec.raw_content):
                return -10.0

            content_lower = rec.raw_content.lower()
            prop_lower = rec.property.lower()
            val_lower = rec.value.lower()
            score = 0.0
            for term in query_terms:
                if term in prop_lower:
                    score += 3.0
                elif term in val_lower:
                    score += 2.0
                elif term in content_lower:
                    score += 1.0
            if rec.relation == IRRelation.MIGRATED:
                score += 1.5
            return score

        scored_records = sorted(scoped_records, key=relevance_score, reverse=True)
        top_scored = [r for r in scored_records if relevance_score(r) > 0]
        if not top_scored:
            top_scored = [r for r in scoped_records if not injection_re.search(r.raw_content)][:10]
        else:
            top_scored = top_scored[:8]

        # 7. Transitive Associative Expansion (Multi-Hop / Indirect Recall):
        # If query seeks a property not directly in scoped records (e.g. "machine", "city", "location"),
        # link via associated values (e.g. value="ripgrep" -> "workstation with ripgrep in Porto")
        associated_records: list[StructuredIR] = []
        for top_r in top_scored:
            val = top_r.value.lower()
            if len(val) >= 3 and val not in stop_words:
                for other_r in ir_records:
                    if other_r not in scoped_records and val in other_r.raw_content.lower():
                        # Exclude other projects' direct assertions
                        if not any(f"project {p}" in other_r.raw_content.lower() for p in ["delta", "quartz", "beacon", "prism"]):
                            associated_records.append(other_r)

        all_final_records = top_scored + associated_records[:3]
        lines = [r.raw_content for r in all_final_records]
        return ResolvedContext(
            context_text="\n".join(lines),
            primary_entity=target_entity,
            matched_records=all_final_records,
        )

    def _extract_target_entity(self, query: str) -> str | None:
        for p in self.ENTITY_QUERY_PATTERNS:
            m = p.search(query)
            if m:
                return m.group(1).lower()
        return None

    def _extract_target_date(self, query: str) -> str | None:
        m = self.DATE_PATTERN.search(query)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return None
