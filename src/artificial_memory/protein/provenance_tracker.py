"""Provenance Tracker for AM Apex Protein Phase (Protein #5).

Attaches an audit trail and provenance metadata to synthesized evidence:
    Answer -> Claim -> Evidence -> Original Turn / Session

Enables full backward interpretability:
- claim_id: Unique deterministic claim identifier
- source_turns: Origin turn IDs or dates
- hop_distance: Graph expansion distance (0, 1, 2, 3)
- confidence: Grounding score [0.0, 1.0]
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class EvidenceProvenance:
    claim_id: str
    source_turns: list[str]
    hop_distance: int
    confidence: float
    timestamp: str = ""


class ProvenanceTracker:
    """Tracks origin and derivation hops for synthesized evidence."""

    DIA_ID_PATTERN = re.compile(r"\[(D\d+:\d+|\d+)\]")

    def track(
        self,
        claim_text: str,
        supporting_records: Sequence[StructuredIR],
        hop_distance: int = 0,
    ) -> EvidenceProvenance:
        """Create a provenance certificate for an evidence claim."""
        source_turns = []
        timestamps = []

        for r in supporting_records:
            if r.raw_content:
                m = self.DIA_ID_PATTERN.search(r.raw_content)
                if m:
                    source_turns.append(m.group(1))
            if r.time_scope:
                timestamps.append(r.time_scope)

        # Deterministic claim ID from text hash
        c_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:10]
        claim_id = f"claim-{c_hash}"

        # Confidence based on hop distance and source count
        confidence = max(0.5, 1.0 - (hop_distance * 0.15))

        return EvidenceProvenance(
            claim_id=claim_id,
            source_turns=list(set(source_turns)),
            hop_distance=hop_distance,
            confidence=confidence,
            timestamp=timestamps[0] if timestamps else "",
        )
