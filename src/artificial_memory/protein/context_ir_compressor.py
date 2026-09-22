"""Context IR Compressor for AM Apex Protein Phase (Protein #4).

Transforms conversational evidence turns into dense, structured Context IR:
    [STATE] <Entity> -> <property>: <value>
    [SUPERSEDED] <Entity> -> <property>: <old_value> [date]
    [EVENT] <Entity>: <event_description> [date]
    [RELATION] <Entity> -> <relation> -> <target>

Achieves 3x higher information density per token, strictly maintaining token budgets <= 120 tok.
"""

from __future__ import annotations

import re
from typing import Sequence

from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR
from artificial_memory.protein.temporal_supersession_protein import SupersededEvidenceGroup


class ContextIRCompressor:
    """Renders high-density Context IR from structured records."""

    def _extract_turn_id(self, content: str) -> str:
        """Extract turn ID like [D1:3] from raw content."""
        if not content:
            return ""
        m_id = re.search(r"\[(D\d+:\d+|\d+)", content)
        return f"[{m_id.group(1)}] " if m_id else ""

    def compress(
        self,
        records: Sequence[StructuredIR],
        superseded_groups: list[SupersededEvidenceGroup] | None = None,
        target_token_budget: int = 180,
    ) -> str:
        """Compress records into dense Context IR."""
        lines: list[str] = []
        curr_tokens = 0

        # 1. First format explicit superseded state transitions (up to 2-3 most relevant)
        if superseded_groups:
            for sg in superseded_groups[:3]:
                if curr_tokens >= target_token_budget:
                    break
                c_val = sg.current_record.value or "unknown"
                c_content = self._strip_filler(sg.current_record.raw_content or f"{sg.entity} -> {sg.property}: {c_val}")
                line = f"[CURRENT] {c_content}"
                lines.append(line)
                curr_tokens += len(line.split())

                for old_r in sg.superseded_records[:2]:
                    if curr_tokens >= target_token_budget:
                        break
                    old_content = self._strip_filler(old_r.raw_content or f"{old_r.entity} -> {old_r.property}: {old_r.value}")
                    old_line = f"  [SUPERSEDED] {old_content}"
                    lines.append(old_line)
                    curr_tokens += len(old_line.split())

        # 2. Format remaining factual records
        for r in records:
            if curr_tokens >= target_token_budget:
                break

            # If already covered by superseded groups, skip
            if r.status == IRStatus.SUPERSEDED and superseded_groups:
                continue

            content = (r.raw_content or "").strip()
            ent = r.entity or ""
            prop = r.property or ""
            val = r.value or ""
            date = r.time_scope or ""
            date_tag = f" [{date}]" if date else ""

            src_id = self._extract_turn_id(content)

            if ent and prop and val and prop.lower() not in ["general", "none", "statement", "dialogue", ""]:
                line = f"{src_id}[FACT] {ent} -> {prop}: {val}{date_tag}"
            elif r.relation == IRRelation.MIGRATED:
                line = f"{src_id}[EVENT:MIGRATION] {ent} -> {prop}: {r.old_value} -> {val}{date_tag}"
            elif content:
                clean_content = self._strip_filler(content)
                line = clean_content if clean_content.startswith("[") else f"{src_id}{clean_content}"
            else:
                continue

            lines.append(line)
            curr_tokens += len(line.split())

        return "\n".join(lines)

    def _strip_filler(self, text: str) -> str:
        """Clean conversational fillers to maximize density."""
        cleaned = re.sub(r"\b(yeah|yep|oh|ah|haha|thanks|okay|ok|well)\b,?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def compress_adaptive(
        self,
        records: Sequence[StructuredIR],
        superseded_groups: list[SupersededEvidenceGroup] | None = None,
        target_token_budget: int = 600,
    ) -> str:
        """Adaptive tiered Context IR compression:
        - Tier 1 (Rank 1..3 / High Relevance): Full verbatim natural dialogue turn.
        - Tier 2 (Rank 4..8 / Medium Relevance): Clean dialogue turn with filler stripped.
        - Tier 3 (Rank 9+ / Low Relevance): Compact structured fact triple [FACT] Ent -> Prop: Val.
        - Superseded States: [CURRENT] ... [SUPERSEDED] ...
        """
        lines: list[str] = []
        curr_tokens = 0

        # 1. First format explicit superseded state transitions (up to 3 most relevant)
        if superseded_groups:
            for sg in superseded_groups[:3]:
                if curr_tokens >= target_token_budget:
                    break
                c_val = sg.current_record.value or "unknown"
                c_content = self._strip_filler(sg.current_record.raw_content or f"{sg.entity} -> {sg.property}: {c_val}")
                line = f"[CURRENT] {c_content}"
                lines.append(line)
                curr_tokens += len(line.split())

                for old_r in sg.superseded_records[:2]:
                    if curr_tokens >= target_token_budget:
                        break
                    old_content = self._strip_filler(old_r.raw_content or f"{old_r.entity} -> {old_r.property}: {old_r.value}")
                    old_line = f"  [SUPERSEDED] {old_content}"
                    lines.append(old_line)
                    curr_tokens += len(old_line.split())

        # 2. Format remaining factual records with tiered strategy
        for rank_idx, r in enumerate(records):
            if curr_tokens >= target_token_budget:
                break

            if r.status == IRStatus.SUPERSEDED and superseded_groups:
                continue

            content = (r.raw_content or "").strip()
            ent = r.entity or ""
            prop = r.property or ""
            val = r.value or ""
            date = r.time_scope or ""
            date_tag = f" [{date}]" if date else ""
            src_id = self._extract_turn_id(content)

            if rank_idx < 3:
                # Tier 1 (Top 1..3): Verbatim natural turn
                if content:
                    line = content if content.startswith("[") else f"{src_id}{content}"
                elif ent and prop and val:
                    line = f"{src_id}[FACT] {ent} -> {prop}: {val}{date_tag}"
                else:
                    continue
            elif rank_idx < 8:
                # Tier 2 (Rank 4..8): Cleaned natural turn (filler stripped)
                if content:
                    clean_content = self._strip_filler(content)
                    line = clean_content if clean_content.startswith("[") else f"{src_id}{clean_content}"
                elif ent and prop and val:
                    line = f"{src_id}[FACT] {ent} -> {prop}: {val}{date_tag}"
                else:
                    continue
            else:
                # Tier 3 (Rank 9+): Compact structured fact triple
                if ent and prop and val and prop.lower() not in ["general", "none", "statement", "dialogue", ""]:
                    line = f"{src_id}[FACT] {ent} -> {prop}: {val}{date_tag}"
                elif r.relation == IRRelation.MIGRATED:
                    line = f"{src_id}[EVENT:MIGRATION] {ent} -> {prop}: {r.old_value} -> {val}{date_tag}"
                elif content:
                    clean_content = self._strip_filler(content)
                    line = clean_content if clean_content.startswith("[") else f"{src_id}{clean_content}"
                else:
                    continue

            lines.append(line)
            curr_tokens += len(line.split())

        return "\n".join(lines)
