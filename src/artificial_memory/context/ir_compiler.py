from __future__ import annotations

import re

from artificial_memory.context.ir_models import (
    IRKey,
    IRSequence,
    IRUnit,
    create_ir_unit,
)
from artificial_memory.core.interfaces import ContextIRCompiler
from artificial_memory.core.models import Message


class RuleBasedIRCompiler:
    """Rule-based Context IR compiler for MVP."""

    def __init__(self):
        self.sequence_counter = 0

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for IR extraction."""
        self.patterns = {
            # Decision patterns
            'decision': re.compile(
                r'(decid\w+|choos\w+|select\s+\w+|adopt\s+\w+|reject\s+\w+|go with|settle on|finaliz\w+)',
                re.IGNORECASE
            ),
            # Reasoning patterns
            'reasoning': re.compile(
                r'(because|reason|since|therefore|thus|logic|rationale|why)',
                re.IGNORECASE
            ),
            # Temporal patterns
            'temporal': re.compile(
                r'\b(now|today|yesterday|tomorrow|later|before|after|currently|recently|earlier|later|soon|eventually|finally|first|then|next)\b',
                re.IGNORECASE
            ),
            # State patterns
            'state': re.compile(
                r'\b(status|current|state|progress|done|pending|complete|in progress|finished|working on)\b',
                re.IGNORECASE
            ),
            # Concern patterns
            'concern': re.compile(
                r'\b(concern|risk|issue|problem|worry|afraid|scared|danger|threat|vulnerab)\b',
                re.IGNORECASE
            ),
            # Tradeoff patterns
            'tradeoff': re.compile(
                r'\b(tradeoff|trade-off|pros and cons|advantage|disadvantage|benefit|drawback|upside|downside)\b',
                re.IGNORECASE
            ),
            # Agreement patterns
            'agreement': re.compile(
                r'\b(agree|consensus|aligned|same page|on board|support|approve)\b',
                re.IGNORECASE
            ),
            # Action patterns
            'action': re.compile(
                r'\b(will|should|need to|must|have to|action|todo|next step|follow up)\b',
                re.IGNORECASE
            ),
            # Knowledge/fact patterns
            'knowledge': re.compile(
                r'\b(fact|define|definition|concept|principle|theorem|rule|law)\b',
                re.IGNORECASE
            ),
            # Casual/informal markers
            'casual': re.compile(
                r'\b(yeah|ok|okay|sure|cool|awesome|great|nice|right|got it|makes sense)\b',
                re.IGNORECASE
            ),
            # Hesitation markers
            'hesitation': re.compile(
                r'\b(um|uh|er|ah|hmm|well|maybe|perhaps|not sure|i think|i guess|sort of|kind of)\b',
                re.IGNORECASE
            ),
        }

    def compile_to_ir(self, messages: list[Message]) -> IRSequence:
        """Compile a list of messages into an IR sequence."""
        sequence = IRSequence()
        self.sequence_counter = 0

        for msg in messages:
            units = self._extract_ir_from_message(msg)
            for unit in units:
                sequence.add_unit(unit)

        return sequence

    def _extract_ir_from_message(self, msg: Message) -> list[IRUnit]:
        """Extract IR units from a single message."""
        units = []
        content = msg.content
        role = msg.role.value

        # Add role marker
        if role == "user":
            units.append(create_ir_unit("role", "usr", None, self.sequence_counter))
        elif role == "assistant":
            units.append(create_ir_unit("role", "ast", None, self.sequence_counter))
        self.sequence_counter += 1

        # Extract based on patterns
        units.extend(self._extract_decisions(content))
        units.extend(self._extract_reasoning(content))
        units.extend(self._extract_temporal(content))
        units.extend(self._extract_state(content))
        units.extend(self._extract_concerns(content))
        units.extend(self._extract_tradeoffs(content))
        units.extend(self._extract_agreements(content))
        units.extend(self._extract_actions(content))
        units.extend(self._extract_knowledge(content))
        units.extend(self._extract_tone_markers(content))

        # Extract entities (simple pattern: capitalized words)
        units.extend(self._extract_entities(content))

        # Extract topics from message metadata or content
        if msg.metadata and 'topic' in msg.metadata:
            units.append(create_ir_unit("topic", IRKey.TOPIC, msg.metadata['topic'], self.sequence_counter))
            self.sequence_counter += 1

        return units

    def _extract_decisions(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['decision'].finditer(content)
        for match in matches:
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "decision", IRKey.DECISION_EVENT, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_reasoning(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['reasoning'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 150)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "reasoning", IRKey.REASONING, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_temporal(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['temporal'].finditer(content)
        for match in matches:
            units.append(create_ir_unit(
                "temporal", IRKey.TEMPORAL, match.group(), self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_state(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['state'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "state", IRKey.STATE, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_concerns(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['concern'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 150)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "concern", IRKey.CONCERN, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_tradeoffs(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['tradeoff'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 150)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "tradeoff", IRKey.TRADEOFF, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_agreements(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['agreement'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "agreement", IRKey.AGREEMENT, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_actions(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['action'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "action", IRKey.ACTION, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_knowledge(self, content: str) -> list[IRUnit]:
        units = []
        matches = self.patterns['knowledge'].finditer(content)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 150)
            context = content[start:end].strip()

            units.append(create_ir_unit(
                "knowledge", IRKey.KNOWLEDGE, context, self.sequence_counter
            ))
            self.sequence_counter += 1
        return units

    def _extract_tone_markers(self, content: str) -> list[IRUnit]:
        units = []

        # Casual markers
        if self.patterns['casual'].search(content):
            units.append(create_ir_unit("tone", IRKey.CASUAL, "true", self.sequence_counter))
            self.sequence_counter += 1

        # Hesitation markers
        if self.patterns['hesitation'].search(content):
            units.append(create_ir_unit("tone", IRKey.HESITATION, "true", self.sequence_counter))
            self.sequence_counter += 1

        # Questioning (check for ?)
        if '?' in content or '？' in content:
            units.append(create_ir_unit("tone", IRKey.QUESTIONING, "true", self.sequence_counter))
            self.sequence_counter += 1

        return units

    def _extract_entities(self, content: str) -> list[IRUnit]:
        """Extract meaningful entities from content."""
        units = []

        # Common words to exclude (false positives from capitalization)
        stop_words = {
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'But', 'Or',
            'If', 'Then', 'Else', 'When', 'Where', 'Why', 'How', 'What', 'Who',
            'Let', 'We', 'You', 'I', 'It', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been',
            'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Could',
            'Should', 'May', 'Might', 'Must', 'Can', 'Need', 'Want', 'Like', 'Just',
            'Also', 'Even', 'Still', 'Already', 'Yet', 'Not', 'No', 'Yes', 'Okay',
            'Sure', 'Right', 'Good', 'Great', 'Fine', 'Sure', 'Agreed', 'Done',
            'Now', 'Then', 'Here', 'There', 'Up', 'Down', 'In', 'Out', 'On', 'Off'
        }

        # Pattern for capitalized words (potential entities) - at least 3 chars
        entity_pattern = re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b')
        known_entities = set()

        for match in entity_pattern.finditer(content):
            entity = match.group(1)
            if entity not in known_entities and entity not in stop_words:
                known_entities.add(entity)
                units.append(create_ir_unit("entity", IRKey.ENTITY, entity, self.sequence_counter))
                self.sequence_counter += 1

        # Also look for known technical terms / project names
        tech_pattern = re.compile(r'\b([A-Z]{2,}(?:[_-][A-Z0-9]+)*)\b')  # e.g., API, REST, JSON, B, C++
        for match in tech_pattern.finditer(content):
            entity = match.group(1)
            if entity not in known_entities and len(entity) > 1:
                known_entities.add(entity)
                units.append(create_ir_unit("entity", IRKey.ENTITY, entity, self.sequence_counter))
                self.sequence_counter += 1

        return units

    def decompile_from_ir(self, sequence: IRSequence) -> str:
        """Decompile IR sequence back to natural language (basic reconstruction)."""
        parts = []

        # Group by type for better reconstruction
        by_type = {}
        for unit in sequence.units:
            if unit.ir_type not in by_type:
                by_type[unit.ir_type] = []
            by_type[unit.ir_type].append(unit)

        # Reconstruct in logical order
        order = ["role", "topic", "entity", "tone", "decision", "reasoning",
                 "concern", "tradeoff", "agreement", "action", "state",
                 "temporal", "knowledge"]

        for ir_type in order:
            if ir_type in by_type:
                for unit in by_type[ir_type]:
                    text = self._unit_to_text(unit)
                    if text:
                        parts.append(text)

        # Add any remaining types
        for ir_type, units in by_type.items():
            if ir_type not in order:
                for unit in units:
                    text = self._unit_to_text(unit)
                    if text:
                        parts.append(text)

        return ' '.join(parts)

    def _unit_to_text(self, unit: IRUnit) -> str:
        """Convert a single IR unit to natural language text."""
        if unit.ir_type == "role":
            return "User:" if unit.ir_key == "usr" else "Assistant:"

        if unit.ir_type == "topic":
            return f"[Topic: {unit.ir_value}]"

        if unit.ir_type == "entity":
            return f"[{unit.ir_value}]"

        if unit.ir_type == "tone":
            if unit.ir_key == IRKey.CASUAL:
                return "[casual tone]"
            elif unit.ir_key == IRKey.HESITATION:
                return "[hesitant]"
            elif unit.ir_key == IRKey.QUESTIONING:
                return "[questioning]"
            return ""

        # For content-bearing units, use the value
        if unit.ir_value:
            # Clean up the context snippet
            value = unit.ir_value.strip()
            # Remove extra whitespace
            value = re.sub(r'\s+', ' ', value)
            return value[:200]  # Limit length

        return ""


class IRCompressor:
    """Compress IR sequences for token efficiency."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def compress(self, sequence: IRSequence) -> IRSequence:
        """Compress IR sequence to fit token budget."""
        if sequence.total_tokens() <= self.max_tokens:
            return sequence

        # Priority order for keeping units (higher = more important)
        priority = {
            "decision": 10,
            "decision_event": 10,
            "reasoning": 9,
            "concern": 8,
            "tradeoff": 8,
            "agreement": 7,
            "action": 7,
            "state": 6,
            "knowledge": 5,
            "entity": 5,
            "topic": 6,
            "temporal": 4,
            "tone": 3,
            "role": 2,
        }

        # Sort units by priority (descending) then by sequence (ascending)
        scored_units = []
        for unit in sequence.units:
            p = priority.get(unit.ir_type, 1)
            scored_units.append((-p, unit.sequence_num, unit))

        scored_units.sort()

        # Select units within budget
        selected = []
        total_tokens = 0

        for _, _, unit in scored_units:
            unit_tokens = unit.estimate_tokens()
            if total_tokens + unit_tokens <= self.max_tokens:
                selected.append(unit)
                total_tokens += unit_tokens
            else:
                break

        # Restore original order
        selected.sort(key=lambda u: u.sequence_num)

        compressed = IRSequence(
            units=selected,
            conversation_id=sequence.conversation_id,
            topic_id=sequence.topic_id,
            metadata={**sequence.metadata, "compressed": True, "original_tokens": sequence.total_tokens()}
        )

        return compressed

    def optimize_for_context(self, sequence: IRSequence, max_units: int = 100) -> IRSequence:
        """Optimize IR for LLM context - remove duplicates, merge similar."""
        # Remove consecutive duplicates
        deduped = []
        prev_key = None
        prev_value = None

        for unit in sequence.units:
            if unit.ir_key != prev_key or unit.ir_value != prev_value:
                deduped.append(unit)
                prev_key = unit.ir_key
                prev_value = unit.ir_value

        # Limit total units
        if len(deduped) > max_units:
            deduped = deduped[:max_units]

        # Re-sequence
        for i, unit in enumerate(deduped):
            unit.sequence_num = i

        return IRSequence(
            units=deduped,
            conversation_id=sequence.conversation_id,
            topic_id=sequence.topic_id,
            metadata=sequence.metadata
        )

    def optimize_ir(self, ir_units: list, max_units: int) -> list:
        """Protocol method: optimize IR units for context."""
        # Convert list to sequence, optimize, return units
        if not ir_units:
            return []
        # Create a temporary sequence
        temp_seq = IRSequence(
            units=ir_units,
            conversation_id=0,
            topic_id=0,
        )
        optimized = self.optimize_for_context(temp_seq, max_units)
        return optimized.units


def create_ir_compiler() -> ContextIRCompiler:
    """Factory function to create IR compiler."""
    return RuleBasedIRCompiler()
