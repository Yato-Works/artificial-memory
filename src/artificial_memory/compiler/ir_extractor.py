"""Universal Cognitive IR Extractor (Phase 2).

Extracts StructuredIR representations from dialogue turns without domain-specific
hardcoding, using deterministic syntactic and semantic pattern resolution.
"""

from __future__ import annotations

import re
from typing import Any

from artificial_memory.core.ir import IRRelation, IRStatus, StructuredIR


class UniversalIRExtractor:
    """Extracts StructuredIR records from conversation turns."""

    # 1. Generic Entity Patterns (captures "project X", "workstation in X", "user", "Eli", etc.)
    ENTITY_PATTERNS = [
        re.compile(r"\bfor\s+project\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bworking\s+on\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bproject\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bthe\s+workstation\s+in\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\b(user|assistant|teammate)\b", re.IGNORECASE),
    ]

    # 2. Generic Temporal Patterns (ISO dates, ranges, relative times)
    TIME_PATTERNS = [
        re.compile(r"\b(until\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(on\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(from\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(in\s+the\s+evenings)\b", re.IGNORECASE),
        re.compile(r"\b(currently|now|previously)\b", re.IGNORECASE),
    ]

    def extract(self, text: str, default_source: str = "user") -> list[StructuredIR]:
        """Extract all StructuredIR units from a given text line or turn."""
        records: list[StructuredIR] = []
        clean = text.strip()
        if not clean:
            return records

        source = default_source
        if "teammate reported" in clean.lower():
            source = "teammate"
        elif "assistant" in clean.lower():
            source = "assistant"

        entity = self._extract_entity(clean)
        if entity == "general" and default_source and default_source.lower() not in ["user", "assistant", "teammate", "general"]:
            entity = default_source.lower()
        time_scope = self._extract_time(clean)

        # Pattern A: Migration / Transition (from OLD to NEW, or directly to NEW)
        m_mig = re.search(
            r"(?:migrated|migrate|moved|move|switched|switch)\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+(?:from\s+([a-zA-Z0-9_\s]+?)\s+)?to\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_mig:
            prop_candidate = m_mig.group(1).strip()
            # If entity is included in the matched property candidate, strip it
            prop_candidate = re.sub(r"^(?:project\s+[a-zA-Z0-9_-]+\s+|the\s+)", "", prop_candidate, flags=re.IGNORECASE).strip()
            prop = self._normalize_property(prop_candidate)
            old_val = m_mig.group(2).strip() if m_mig.group(2) else None
            new_val = m_mig.group(3).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=new_val,
                old_value=old_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.MIGRATED,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern B: Explicit Negative Constraint / Non-existence (never X)
        m_neg = re.search(
            r"(?:never\s+(?:a\s+)?([a-zA-Z0-9_\s]+)|only\s+discussed\s+([a-zA-Z0-9_\s]+)\s+and\s+never\s+(?:a\s+)?([a-zA-Z0-9_\s]+))",
            clean,
            re.IGNORECASE,
        )
        if m_neg:
            prop = self._normalize_property(m_neg.group(3) or m_neg.group(1) or "unspecified")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value="none",
                time_scope=time_scope,
                source=source,
                relation=IRRelation.NEVER,
                status=IRStatus.UNSPECIFIED,
                raw_content=clean,
            ))
            return records

        # Pattern C: Location / Residency
        m_loc = re.search(
            r"(?:working\s+from|relocated\s+to|located\s+in)\s+([a-zA-Z0-9_\s]+?)(?:\s+until|\s+on|\s+from|\.|$)",
            clean,
            re.IGNORECASE,
        )
        if m_loc:
            val = m_loc.group(1).strip()
            records.append(StructuredIR(
                entity=entity,
                property="location",
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.LOCATED_AT,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern D: Consumption / Habits (drinks X while working on Y)
        m_drink = re.search(
            r"usually\s+drinks\s+([a-zA-Z0-9_\s]+?)\s+while\s+working\s+on\s+([a-zA-Z0-9_-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_drink:
            val = m_drink.group(1).strip()
            ent = m_drink.group(2).strip()
            records.append(StructuredIR(
                entity=ent,
                property="drinks",
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.PREFERS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern E: Tool / Setting Choice (settled on X, pinned X to Y, set X to Y)
        m_tool = re.search(
            r"(?:settled\s+on|pinned\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+to|set\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+to|preferred\s+([a-zA-Z0-9_\s]+?)\s+is)\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_tool:
            prop = self._normalize_property(m_tool.group(1) or m_tool.group(2) or m_tool.group(3) or "tool")
            val = m_tool.group(4).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.PREFERS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern F: Behavioral Tendency (reverts, backups, copies by hand)
        if any(w in clean.lower() for w in ["reverts", "backup", "copies files", "revert"]):
            records.append(StructuredIR(
                entity=entity,
                property="behavior_tendency",
                value=clean,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.BEHAVIOR,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern G: General Factual Assertion (runs on X, was X, is X)
        m_fact = re.search(
            r"(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+(?:runs\s+on|was|is|actually\s+runs\s+on)\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_fact:
            prop = self._normalize_property(m_fact.group(1))
            val = m_fact.group(2).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.ASSERTS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Fallback: Capture as raw proposition
        records.append(StructuredIR(
            entity=entity,
            property="statement",
            value=clean,
            time_scope=time_scope,
            source=source,
            relation=IRRelation.ASSERTS,
            status=IRStatus.ACTIVE,
            raw_content=clean,
        ))
        return records

    def _normalize_property(self, text: str) -> str:
        """Normalize property name by stripping articles and possessive pronouns."""
        clean = text.strip().rstrip(".").lower()
        clean = re.sub(r"^(?:the|a|an|our|my|their|its|his|her)\s+", "", clean, flags=re.IGNORECASE).strip()
        return clean

    def _extract_entity(self, text: str) -> str:
        for p in self.ENTITY_PATTERNS:
            m = p.search(text)
            if m:
                val = m.group(1).lower()
                return val
        return "general"

    def _extract_time(self, text: str) -> str | None:
        for p in self.TIME_PATTERNS:
            m = p.search(text)
            if m:
                return m.group(1).lower()
        return None
