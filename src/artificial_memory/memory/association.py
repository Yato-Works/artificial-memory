from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Association, AssociationType, Memory
from artificial_memory.metrics.collector import get_metrics_collector


@dataclass
class AssociationCandidate:
    """Candidate association with score."""
    source_id: int
    target_id: int
    association_type: AssociationType
    strength: float
    evidence: str


class AssociationEngine:
    """Engine for creating and managing semantic associations between memories."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.collector = get_metrics_collector()

        # Keywords for different association types
        self.association_patterns = {
            AssociationType.CAUSES: [
                r'(ため|原因|理由|引き起こす|導く|結果として|thus|therefore|because)',
            ],
            AssociationType.FOLLOWS: [
                r'(その後|次に|それから|後で|later|then|after|followed by)',
            ],
            AssociationType.CONTRADICTS: [
                r'(しかし|だが| nevertheless|contrary|相反|矛盾|反対|違う|間違い)',
            ],
            AssociationType.ELABORATES: [
                r'(詳細|詳しく|具体的|言い換えると|つまり|in detail|specifically|elaborate)',
            ],
            AssociationType.SUMMARIZES: [
                r'(まとめると|要約|概要|まとめ|summary|in short|tl;dr)',
            ],
        }

    def analyze_and_create_associations(self, topic_id: int, limit: int = 100) -> list[Association]:
        """Analyze memories and create semantic associations."""
        memories = self.store.get_memories(topic_id=topic_id, limit=limit)
        created = []

        # Compare each pair of memories
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i+1:]:
                candidates = self._analyze_pair(mem_a, mem_b)
                for candidate in candidates:
                    if candidate.strength >= 0.3:  # Minimum threshold
                        association = Association(
                            source_memory_id=candidate.source_id,
                            target_memory_id=candidate.target_id,
                            association_type=candidate.association_type,
                            strength=candidate.strength,
                            created_at=datetime.now(),
                        )
                        created_assoc = self.store.create_association(association)
                        created.append(created_assoc)

                        # Record metric
                        if hasattr(self.collector, 'metrics') and self.collector.metrics:
                            pass  # Skip metric recording for now

        return created

    def _analyze_pair(self, mem_a: Memory, mem_b: Memory) -> list[AssociationCandidate]:
        """Analyze a pair of memories for potential associations."""
        candidates = []

        content_a = mem_a.content.lower()
        content_b = mem_b.content.lower()

        # 1. Content overlap (semantic similarity)
        overlap = self._calculate_overlap(content_a, content_b)

        # 2. Temporal proximity
        temporal_score = self._calculate_temporal_proximity(mem_a, mem_b)

        # 3. Explicit causal/follows/contradiction language
        explicit_types = self._detect_explicit_relations(content_a, content_b)

        # 4. Same entities/topics mentioned
        entity_overlap = self._calculate_entity_overlap(mem_a, mem_b)

        # 5. Resolution hierarchy (same content at different resolutions)
        resolution_relation = self._check_resolution_relation(mem_a, mem_b)

        # Create candidates based on analysis
        base_strength = (
            overlap * 0.3 +
            temporal_score * 0.2 +
            entity_overlap * 0.2 +
            resolution_relation * 0.3
        )

        if base_strength >= 0.3:
            candidates.append(AssociationCandidate(
                source_id=mem_a.id,
                target_id=mem_b.id,
                association_type=AssociationType.RELATED,
                strength=base_strength,
                evidence=f"semantic_overlap:{overlap:.2f}, temporal:{temporal_score:.2f}, entity:{entity_overlap:.2f}"
            ))

        # Add explicit relation candidates
        for assoc_type, strength in explicit_types:
            if strength >= 0.4:
                candidates.append(AssociationCandidate(
                    source_id=mem_a.id,
                    target_id=mem_b.id,
                    association_type=assoc_type,
                    strength=strength,
                    evidence=f"explicit_{assoc_type.value}"
                ))

        # Resolution hierarchy relation
        if resolution_relation > 0.5:
            candidates.append(AssociationCandidate(
                source_id=min(mem_a.id, mem_b.id),  # Lower res -> higher res
                target_id=max(mem_a.id, mem_b.id),
                association_type=AssociationType.SUMMARIZES,
                strength=resolution_relation,
                evidence="resolution_hierarchy"
            ))

        return candidates

    def _calculate_overlap(self, text_a: str, text_b: str) -> float:
        """Calculate word-level overlap between two texts."""
        words_a = set(re.findall(r'\w+', text_a))
        words_b = set(re.findall(r'\w+', text_b))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)  # Jaccard similarity

    def _calculate_temporal_proximity(self, mem_a: Memory, mem_b: Memory) -> float:
        """Calculate temporal proximity score (0-1)."""
        if not mem_a.created_at or not mem_b.created_at:
            return 0.0

        delta = abs((mem_a.created_at - mem_b.created_at).total_seconds())
        days = delta / 86400

        # Exponential decay: 1.0 at same time, ~0.37 at 1 day, ~0.01 at 7 days
        import math
        return math.exp(-days / 2.0)

    def _detect_explicit_relations(self, text_a: str, text_b: str) -> list[tuple[AssociationType, float]]:
        """Detect explicit causal/follows/contradiction relations."""
        relations = []

        for assoc_type, patterns in self.association_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_a, re.IGNORECASE) or re.search(pattern, text_b, re.IGNORECASE):
                    relations.append((assoc_type, 0.7))
                    break

        return relations

    def _calculate_entity_overlap(self, mem_a: Memory, mem_b: Memory) -> float:
        """Calculate entity overlap between memories."""
        # Extract capitalized words as entities
        entities_a = set(re.findall(r'\b[A-Z][a-z]{2,}\b', mem_a.content))
        entities_b = set(re.findall(r'\b[A-Z][a-z]{2,}\b', mem_b.content))

        if not entities_a or not entities_b:
            return 0.0

        return len(entities_a & entities_b) / len(entities_a | entities_b)

    def _check_resolution_relation(self, mem_a: Memory, mem_b: Memory) -> float:
        """Check if memories are same content at different resolutions."""
        if mem_a.id == mem_b.id:
            return 0.0

        # Check if they have same source conversation
        if (mem_a.source_conversation_id and mem_b.source_conversation_id and
            mem_a.source_conversation_id == mem_b.source_conversation_id):
            # Check resolution difference
            res_diff = abs(mem_a.resolution.value - mem_b.resolution.value)
            if 1 <= res_diff <= 2:
                return 0.8  # Strong indication of resolution hierarchy

        # Check content similarity for same source
        if mem_a.source_conversation_id == mem_b.source_conversation_id:
            overlap = self._calculate_overlap(mem_a.content.lower(), mem_b.content.lower())
            if overlap > 0.5:
                return 0.6

        return 0.0

    def get_associated_memories(self, memory_id: int, min_strength: float = 0.3, max_depth: int = 2) -> list[tuple[Memory, Association]]:
        """Get associated memories via graph traversal."""
        return self.store.get_related_memories(memory_id, min_strength)

    def find_association_path(self, source_id: int, target_id: int, max_depth: int = 3) -> list[Association] | None:
        """Find shortest association path between two memories."""
        # BFS search
        from collections import deque

        queue = deque([(source_id, [])])
        visited = {source_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == target_id:
                return path

            if len(path) >= max_depth:
                continue

            associations = self.store.get_associations(current_id)
            for assoc in associations:
                next_id = assoc.target_memory_id if assoc.source_memory_id == current_id else assoc.source_memory_id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [assoc]))

        return None


def create_association_engine(store: MemoryStore) -> AssociationEngine:
    """Factory function to create association engine."""
    return AssociationEngine(store)
