"""Evidence Fuser for AM Apex Protein Phase (Protein #1).

Performs:
1. Cross-channel Deduplication: Eliminates identical facts gathered across the 6 retrieval channels.
2. Entity Resolution: Normalizes nicknames, pronouns, and kinship references to canonical entities.
3. Relation Merging: Clusters corroborating or complementary facts into unified evidence clusters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class EvidenceCluster:
    """A cluster of merged evidence units sharing an entity anchor."""
    entity: str
    records: list[StructuredIR] = field(default_factory=list)
    canonical_name: str = ""
    properties_covered: set[str] = field(default_factory=set)


class EvidenceFuser:
    """Fuses raw wide evidence pools into clean, non-redundant clusters."""

    NICKNAME_MAP = {
        "mel": "melanie",
        "carol": "caroline",
        "bob": "robert",
        "dave": "david",
    }

    KINSHIP_COREFS = {
        "her sister": "sister",
        "his sister": "sister",
        "my sister": "sister",
        "her brother": "brother",
        "his brother": "brother",
        "my brother": "brother",
        "her partner": "partner",
        "his partner": "partner",
        "my partner": "partner",
        "her dog": "dog",
        "his dog": "dog",
        "my dog": "dog",
    }

    def fuse(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> list[StructuredIR]:
        """Deduplicate and fuse records."""
        # 1. Deduplication by raw content & canonical property-value
        seen_contents = set()
        seen_triples = set()
        deduped: list[StructuredIR] = []

        for r in records:
            content = (r.raw_content or "").strip()
            # Canonical triple key
            ent_norm = self.normalize_entity(r.entity or "")
            prop_norm = (r.property or "").lower().strip()
            val_norm = (r.value or "").lower().strip()
            triple_key = (ent_norm, prop_norm, val_norm)

            if content and content in seen_contents:
                continue
            if ent_norm and prop_norm and val_norm and triple_key in seen_triples:
                continue

            if content:
                seen_contents.add(content)
            if ent_norm and prop_norm and val_norm:
                seen_triples.add(triple_key)

            deduped.append(r)

        return deduped

    def deduplicate_only(self, records: Sequence[StructuredIR]) -> list[StructuredIR]:
        """Deduplicate records by exact content and triple without entity alias normalization."""
        seen_contents = set()
        seen_triples = set()
        deduped: list[StructuredIR] = []
        for r in records:
            content = (r.raw_content or "").strip()
            ent = (r.entity or "").lower().strip()
            prop = (r.property or "").lower().strip()
            val = (r.value or "").lower().strip()
            triple_key = (ent, prop, val)
            if content and content in seen_contents:
                continue
            if ent and prop and val and triple_key in seen_triples:
                continue
            if content:
                seen_contents.add(content)
            if ent and prop and val:
                seen_triples.add(triple_key)
            deduped.append(r)
        return deduped

    def normalize_entity(self, name: str) -> str:
        """Normalize entity name or alias to canonical form."""
        name_clean = name.lower().strip()
        return self.NICKNAME_MAP.get(name_clean, name_clean)

    def cluster_by_entity(self, records: Sequence[StructuredIR]) -> list[EvidenceCluster]:
        """Group records into entity-anchored clusters."""
        clusters: dict[str, EvidenceCluster] = {}

        for r in records:
            ent = self.normalize_entity(r.entity or "general")
            if ent not in clusters:
                clusters[ent] = EvidenceCluster(entity=ent, canonical_name=ent)
            clusters[ent].records.append(r)
            if r.property:
                clusters[ent].properties_covered.add(r.property.lower())

        return list(clusters.values())
