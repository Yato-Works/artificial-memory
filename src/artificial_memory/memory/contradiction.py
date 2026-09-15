from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory


class ContradictionType(StrEnum):
    """Types of contradictions."""
    DIRECT = "direct"              # "X is true" vs "X is false"
    ENTITY_VALUE = "entity_value"  # "DB is PostgreSQL" vs "DB is SQLite"
    TEMPORAL = "temporal"          # "X happened before Y" vs "Y happened before X"
    CAUSAL = "causal"              # "A causes B" vs "B causes A"
    SCOPE = "scope"                # "All X are Y" vs "Some X are not Y"
    SOURCE = "source"              # Conflicting sources


@dataclass
class ContradictionPair:
    """A detected contradiction between two memories."""
    memory_a_id: int
    memory_b_id: int
    contradiction_type: ContradictionType
    severity: float  # 0-1
    description: str
    entity: str | None = None      # For entity-value contradictions
    attribute: str | None = None   # For entity-value: the attribute
    value_a: str | None = None     # Value in memory A
    value_b: str | None = None     # Value in memory B
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: str | None = None  # "A_supersedes_B", "B_supersedes_A", "both_valid_contextual"


class ContradictionDetector:
    """Detects contradictions between memories using multiple strategies.

    Strategies:
    1. Rule-based: keyword/negation patterns
    2. Entity-value extraction: compare extracted facts
    3. Temporal logic: check time ordering
    4. (Future) NLI model: natural language inference
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for contradiction detection."""
        # Negation patterns
        self.negation_patterns = [
            (r'\b(is|was|will be)\b', r'\b(is not|was not|will not be)\b'),
            (r'\b(adopted|chose|selected|decided on)\b', r'\b(rejected|did not choose|did not select)\b'),
            (r'\b(agree|agreed|consensus)\b', r'\b(disagree|disagreed|no consensus)\b'),
            (r'\b(true|correct|right)\b', r'\b(false|incorrect|wrong)\b'),
            (r'\b(yes|yeah|yep)\b', r'\b(no|nope|nah)\b'),
            (r'\b(can|could|able to)\b', r'\b(cannot|could not|unable to)\b'),
        ]

        # Entity-value patterns (simple)
        self.entity_value_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:is|was|equals?|==)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s*:\s*(\w+(?:\s+\w+)*)',
        ]

    def detect_contradictions(self, new_memory: Memory,
                              existing_memories: list[Memory]) -> list[ContradictionPair]:
        """Detect all contradictions between new memory and existing ones."""
        contradictions = []

        for existing in existing_memories:
            # Skip if same memory
            if existing.id == new_memory.id:
                continue

            # Check different contradiction types
            pairs = self._check_direct_contradiction(new_memory, existing)
            contradictions.extend(pairs)

            pairs = self._check_entity_value_contradiction(new_memory, existing)
            contradictions.extend(pairs)

            pairs = self._check_temporal_contradiction(new_memory, existing)
            contradictions.extend(pairs)

        return contradictions

    def _check_direct_contradiction(self, mem_a: Memory, mem_b: Memory) -> list[ContradictionPair]:
        """Check for direct negation contradictions."""
        contradictions = []
        text_a = mem_a.content.lower()
        text_b = mem_b.content.lower()

        for pos_pattern, neg_pattern in self.negation_patterns:
            pos_matches = re.findall(pos_pattern, text_a)
            neg_matches = re.findall(neg_pattern, text_b)

            if pos_matches and neg_matches:
                # Check if they're talking about the same subject
                if self._same_subject(text_a, text_b):
                    contradictions.append(ContradictionPair(
                        memory_a_id=mem_a.id,
                        memory_b_id=mem_b.id,
                        contradiction_type=ContradictionType.DIRECT,
                        severity=0.8,
                        description=f"Direct contradiction: '{pos_matches[0]}' vs '{neg_matches[0]}'",
                    ))
                    break  # One contradiction per pair is enough

        return contradictions

    def _check_entity_value_contradiction(self, mem_a: Memory, mem_b: Memory) -> list[ContradictionPair]:
        """Check for entity-value contradictions (X=Y vs X=Z)."""
        contradictions = []

        # Extract entity-value pairs from both memories
        entities_a = self._extract_entities(mem_a.content)
        entities_b = self._extract_entities(mem_b.content)

        # Compare entities
        for entity_a, attr_a, value_a in entities_a:
            for entity_b, attr_b, value_b in entities_b:
                # Same entity and attribute, different values
                if entity_a.lower() == entity_b.lower() and attr_a == attr_b:
                    if value_a != value_b:
                        contradictions.append(ContradictionPair(
                            memory_a_id=mem_a.id,
                            memory_b_id=mem_b.id,
                            contradiction_type=ContradictionType.ENTITY_VALUE,
                            severity=0.9,
                            description=f"Entity '{entity_a}' has conflicting values: '{value_a}' vs '{value_b}'",
                            entity=entity_a,
                            attribute=attr_a,
                            value_a=value_a,
                            value_b=value_b,
                        ))

        return contradictions

    def _check_temporal_contradiction(self, mem_a: Memory, mem_b: Memory) -> list[ContradictionPair]:
        """Check for temporal ordering contradictions."""
        contradictions = []

        # Check if both have temporal markers
        temporal_a = self._extract_temporal_info(mem_a.content)
        temporal_b = self._extract_temporal_info(mem_b.content)

        if temporal_a and temporal_b:
            # Check for contradictory ordering
            if temporal_a.get('before') and temporal_b.get('after'):
                if temporal_a['before'] == temporal_b['after']:
                    contradictions.append(ContradictionPair(
                        memory_a_id=mem_a.id,
                        memory_b_id=mem_b.id,
                        contradiction_type=ContradictionType.TEMPORAL,
                        severity=0.7,
                        description=f"Temporal contradiction: ordering of {temporal_a['before']}",
                    ))

        return contradictions

    def _same_subject(self, text_a: str, text_b: str) -> bool:
        """Heuristic: check if two texts discuss the same subject."""
        # Extract key nouns/entities
        words_a = set(re.findall(r'\b[A-Z][a-z]{2,}\b', text_a))
        words_b = set(re.findall(r'\b[A-Z][a-z]{2,}\b', text_b))

        # Common technical terms
        common = words_a & words_b
        return len(common) >= 1

    def _extract_entities(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (entity, attribute, value) triples from text."""
        entities = []

        for pattern in self.entity_value_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    entity = match[0].strip()
                    value = match[1].strip()
                    entities.append((entity, "value", value))

        # Also check for "X is Y" patterns
        is_pattern = r'(\w+(?:\s+\w+)*)\s+(?:is|was|equals?)\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(is_pattern, text, re.IGNORECASE)
        for entity, value in matches:
            entities.append((entity.strip(), "is", value.strip()))

        return entities

    def _extract_temporal_info(self, text: str) -> dict[str, str] | None:
        """Extract temporal ordering information."""
        info = {}

        # "X before Y" or "X after Y"
        before_pattern = r'(\w+(?:\s+\w+)*)\s+(?:before|earlier than)\s+(\w+(?:\s+\w+)*)'
        after_pattern = r'(\w+(?:\s+\w+)*)\s+(?:after|later than)\s+(\w+(?:\s+\w+)*)'

        before_match = re.search(before_pattern, text, re.IGNORECASE)
        if before_match:
            info['before'] = before_match.group(2).strip()
            info['event'] = before_match.group(1).strip()

        after_match = re.search(after_pattern, text, re.IGNORECASE)
        if after_match:
            info['after'] = after_match.group(2).strip()
            info['event'] = after_match.group(1).strip()

        return info if info else None

    def resolve_contradiction(
        self,
        contradiction: ContradictionPair,
        resolution: str,  # "A_supersedes_B", "B_supersedes_A", "contextual"
    ) -> None:
        """Mark a contradiction as resolved."""
        contradiction.resolved = True
        contradiction.resolution = resolution

    def get_unresolved_contradictions(self) -> list[ContradictionPair]:
        """Get all unresolved contradictions."""
        # In production, this would query a database
        return []


def create_contradiction_detector(store: MemoryStore) -> ContradictionDetector:
    return ContradictionDetector(store)
