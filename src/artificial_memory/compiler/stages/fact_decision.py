from __future__ import annotations

import hashlib
import re
from datetime import datetime

from artificial_memory.compiler.pipeline import (
    BaseStage,
    CompilerContext,
    Diagnostic,
    DiagnosticSeverity,
)
from artificial_memory.context.ir_models import IRUnit
from artificial_memory.core.ir import (
    AccessRecord,
    ConfidenceProfile,
    LifecycleState,
    MemoryIdentity,
    MemoryIR,
    MemoryType,
    ProvenanceChain,
    ProvenanceLink,
    ResolutionLevel,
    SemanticContent,
    TemporalScope,
)
from artificial_memory.core.models import Message, MessageRole


class FactDecisionIntentStage(BaseStage):
    """Classify IR units into memory types: DECISION, SEMANTIC, EPISODE, TIMELINE, CURRENT."""

    def __init__(self):
        super().__init__("fact_decision_intent")

        # Decision patterns (from existing compiler)
        self.decision_patterns = [
            r'決めた|決定|採用|却下|選択|選んだ|決断',
            r'decided|adopted|rejected|selected|chose|go with|settle on',
        ]

        # Important info patterns
        self.important_patterns = [
            r'重要|大事|クリティカル|必須|マスト',
            r'理由|なぜなら|なぜ|懸念|リスク',
            r'アーキテクチャ|設計|方針',
            r'important|critical|must|reason|architecture|design',
        ]

    def process(self, ctx: CompilerContext) -> CompilerContext:
        if not ctx.ir_units:
            return ctx

        # Group IR units by conversation turn
        turn_units = self._group_by_turn(ctx.ir_units, ctx.messages)

        # Create memory candidates from each assistant turn
        for turn_idx, (msg, units) in enumerate(turn_units):
            if msg.role != MessageRole.ASSISTANT:
                continue

            memory_candidates = self._classify_turn(msg, units, ctx, turn_idx)
            for candidate in memory_candidates:
                ctx.memory_ir_list.append(candidate)

        # Also create TIMELINE memory from conversation
        timeline_memory = self._create_timeline_memory(ctx)
        if timeline_memory:
            ctx.memory_ir_list.append(timeline_memory)

        return ctx

    def _group_by_turn(self, ir_units: list[IRUnit], messages: list[Message]) -> list[tuple]:
        """Group IR units by their originating message."""
        turn_units = []
        unit_idx = 0

        for msg_idx, msg in enumerate(messages):
            msg_units = []
            # Each message roughly gets 10 sequence numbers worth of units
            max_seq = (msg_idx + 1) * 10
            while unit_idx < len(ir_units) and ir_units[unit_idx].sequence_num < max_seq:
                msg_units.append(ir_units[unit_idx])
                unit_idx += 1
            turn_units.append((msg, msg_units))

        # Add any remaining units to the last message
        if unit_idx < len(ir_units) and turn_units:
            turn_units[-1] = (turn_units[-1][0], turn_units[-1][1] + ir_units[unit_idx:])

        return turn_units

    def _classify_turn(self, msg: Message, units: list[IRUnit], ctx: CompilerContext, turn_idx: int) -> list[MemoryIR]:
        """Classify an assistant turn into memory candidates."""
        candidates = []

        # Check for decisions
        is_decision = self._contains_decision(msg.content)
        # Check for important info
        is_important = self._contains_important_info(msg.content)

        if is_decision:
            candidates.append(self._create_memory(
                msg, ctx, MemoryType.DECISION, ResolutionLevel.SEMANTIC,
                importance=0.85, confidence=0.8,
                turn_idx=turn_idx, units=units
            ))

        if is_important or is_decision:
            candidates.append(self._create_memory(
                msg, ctx, MemoryType.SEMANTIC, ResolutionLevel.SEMANTIC,
                importance=0.7 if is_important else 0.6, confidence=0.75,
                turn_idx=turn_idx, units=units
            ))

        # Always create episode memory for substantial turns
        if len(msg.content) > 100:
            candidates.append(self._create_memory(
                msg, ctx, MemoryType.EPISODE, ResolutionLevel.EPISODE,
                importance=0.6, confidence=0.7,
                turn_idx=turn_idx, units=units
            ))

        return candidates

    def _contains_decision(self, content: str) -> bool:
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in self.decision_patterns)

    def _contains_important_info(self, content: str) -> bool:
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in self.important_patterns)

    def _create_memory(
        self, msg: Message, ctx: CompilerContext,
        memory_type: MemoryType, resolution: ResolutionLevel,
        importance: float, confidence: float,
        turn_idx: int, units: list[IRUnit]
    ) -> MemoryIR:
        """Create MemoryIR from message and context."""
        now = datetime.now()
        content_hash = hashlib.sha256(msg.content.encode()).hexdigest()[:16]

        return MemoryIR(
            identity=MemoryIdentity(
                memory_id=0,  # Will be assigned on storage
                version=1,
                content_hash=content_hash,
            ),
            type=memory_type,
            resolution=resolution,
            semantic_content=SemanticContent(content=msg.content),
            source=ProvenanceChain(
                source=ProvenanceLink(
                    conversation_id=ctx.conversation.id,
                    message_id=msg.id,
                    timestamp=msg.created_at,
                ),
                compilation_chain=[],
            ),
            temporal_scope=TemporalScope(
                valid_from=now,
                valid_until=None,
                created_at=now,
                updated_at=now,
            ),
            confidence=ConfidenceProfile(
                memory_confidence=confidence,
                retrieval_confidence=importance,
                temporal_confidence=1.0,
                source_confidence=0.9,
                overall=confidence,
            ),
            importance=importance,
            dependencies=[],
            relations=[],
            compression_history=[],
            access_history=AccessRecord(),
            lifecycle_state=LifecycleState.HOT,
        )

    def _create_timeline_memory(self, ctx: CompilerContext) -> MemoryIR | None:
        """Create a timeline entry for the conversation."""
        if not ctx.messages:
            return None

        user_msgs = [m for m in ctx.messages if m.role == MessageRole.USER]
        assistant_msgs = [m for m in ctx.messages if m.role == MessageRole.ASSISTANT]

        first_user = user_msgs[0].content[:200] if user_msgs else ""

        events = []
        for msg in assistant_msgs:
            if self._contains_decision(msg.content):
                events.append(f"決定: {msg.content[:100]}")
            elif self._contains_important_info(msg.content):
                events.append(f"重要: {msg.content[:100]}")

        timeline_content = f"**{ctx.conversation.started_at.strftime('%Y-%m-%d')}**\n"
        timeline_content += f"トピック: {first_user}\n"
        if events:
            timeline_content += "\n".join(f"- {e}" for e in events[:5])

        now = datetime.now()
        content_hash = hashlib.sha256(timeline_content.encode()).hexdigest()[:16]

        return MemoryIR(
            identity=MemoryIdentity(memory_id=0, version=1, content_hash=content_hash),
            type=MemoryType.TIMELINE,
            resolution=ResolutionLevel.EPISODE,
            semantic_content=SemanticContent(content=timeline_content),
            source=ProvenanceChain(
                source=ProvenanceLink(
                    conversation_id=ctx.conversation.id,
                    message_id=None,
                    timestamp=ctx.conversation.started_at,
                )
            ),
            temporal_scope=TemporalScope(
                valid_from=ctx.conversation.started_at,
                valid_until=None,
                created_at=now,
                updated_at=now,
            ),
            confidence=ConfidenceProfile(
                memory_confidence=0.75, retrieval_confidence=0.7,
                temporal_confidence=1.0, source_confidence=0.9, overall=0.75
            ),
            importance=0.7,
            dependencies=[], relations=[], compression_history=[],
            access_history=AccessRecord(),
            lifecycle_state=LifecycleState.HOT,
        )

    def validate_output(self, ctx: CompilerContext) -> list[Diagnostic]:
        diagnostics = []
        if not ctx.memory_ir_list:
            diagnostics.append(Diagnostic(
                stage=self.name, severity=DiagnosticSeverity.WARNING,
                message="No memory candidates generated",
                suggestion="Check if conversation has assistant messages with decisions/important info",
            ))
        return diagnostics
