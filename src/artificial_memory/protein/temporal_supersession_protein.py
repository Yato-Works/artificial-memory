"""Temporal Supersession Protein for AM Apex (Protein #3).

Orders evidence chronologically and resolves state updates:
    E_i < E_j <=> t(E_i) < t(E_j) and Entity(E_i) == Entity(E_j) and Prop(E_i) == Prop(E_j)

Labels:
- [CURRENT_STATE]: The active state at max(t_valid)
- [SUPERSEDED]: Prior historical states with transition/migration pointers
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import IRStatus, StructuredIR


@dataclass
class SupersededEvidenceGroup:
    entity: str
    property: str
    current_record: StructuredIR
    superseded_records: list[StructuredIR]


class TemporalSupersessionProtein:
    """Detects and resolves temporal state updates."""

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")

    def _extract_date(self, record: StructuredIR) -> str:
        """Extract explicit date from time_scope or raw_content."""
        if record.time_scope:
            m = self.DATE_PATTERN.search(record.time_scope)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if record.raw_content:
            m = self.DATE_PATTERN.search(record.raw_content)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return "1970-01-01"

    def resolve(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> tuple[list[StructuredIR], list[SupersededEvidenceGroup]]:
        """Separate records into active states and superseded histories."""
        groups: dict[tuple[str, str], list[StructuredIR]] = defaultdict(list)

        for r in records:
            ent = (r.entity or "").lower().strip()
            prop = (r.property or "").lower().strip()
            if ent and prop and prop not in ["general", "none", ""]:
                groups[(ent, prop)].append(r)
            else:
                # Independent facts without strict property grouping
                groups[((r.raw_content or ""), "raw")].append(r)

        active_records: list[StructuredIR] = []
        superseded_groups: list[SupersededEvidenceGroup] = []

        MUTABLE_PROPERTIES = {
            "residence", "city", "location", "employer", "job", "status",
            "habit", "ratio", "weight", "car", "team", "project", "database",
            "hours", "time", "spent", "count", "own", "attend", "amount",
            "move", "relocate", "price", "cost", "salary", "pet", "dog", "cat",
            "personal best", "frequency", "how often", "schedule", "plan",
        }

        for (ent, prop), recs in groups.items():
            if len(recs) <= 1 or prop == "raw" or prop not in MUTABLE_PROPERTIES:
                active_records.extend(recs)
                continue

            # Check if values actually differ
            distinct_values = {r.value.lower() for r in recs if r.value}
            if len(distinct_values) <= 1:
                active_records.extend(recs)
                continue

            # Sort by date ascending
            recs_sorted = sorted(recs, key=lambda r: self._extract_date(r))
            latest = recs_sorted[-1]
            earlier = recs_sorted[:-1]

            # Mark statuses
            latest.status = IRStatus.ACTIVE
            for old_r in earlier:
                old_r.status = IRStatus.SUPERSEDED

            # Keep all records in active pool to preserve recall!
            active_records.extend(recs_sorted)
            superseded_groups.append(
                SupersededEvidenceGroup(
                    entity=ent,
                    property=prop,
                    current_record=latest,
                    superseded_records=earlier,
                )
            )

        return active_records, superseded_groups
