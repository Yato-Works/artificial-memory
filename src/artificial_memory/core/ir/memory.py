from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import (
    AccessRecord,
    AssociationRef,
    BeliefRef,
    CompressionRecord,
    ConfidenceProfile,
    DependencyRef,
    LifecycleState,
    MemoryIdentity,
    MemoryStatus,
    MemoryType,
    ProvenanceChain,
    ResolutionLevel,
    SemanticContent,
    TemporalScope,
)


class MemoryIR(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    identity: MemoryIdentity
    type: MemoryType
    resolution: ResolutionLevel
    semantic_content: SemanticContent
    source: ProvenanceChain
    temporal_scope: TemporalScope
    confidence: ConfidenceProfile
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    dependencies: list[DependencyRef] = Field(default_factory=list)
    relations: list[AssociationRef] = Field(default_factory=list)
    compression_history: list[CompressionRecord] = Field(default_factory=list)
    access_history: AccessRecord
    lifecycle_state: LifecycleState
    belief_ref: BeliefRef | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "id": self.identity.memory_id,
            "topic_id": self.source.source.conversation_id,
            "memory_type": self.type,
            "content": self.semantic_content.content,
            "resolution": self.resolution,
            "importance": self.importance,
            # Evidence confidence (not the derived `overall`) - keeps the
            # legacy Memory <-> MemoryIR round-trip lossless (P0-3).
            "confidence": self.confidence.memory_confidence,
            "status": self._lifecycle_to_status(),
            "valid_from": self.temporal_scope.valid_from,
            "valid_until": self.temporal_scope.valid_until,
            "is_current": self.lifecycle_state in (LifecycleState.HOT, LifecycleState.WARM),
            "source_conversation_id": self.source.source.conversation_id,
            "source_message_id": self.source.source.message_id,
            "created_at": self.temporal_scope.created_at,
            "updated_at": self.temporal_scope.updated_at,
            "last_accessed": self.access_history.last_accessed,
            "access_count": self.access_history.access_count,
        }

    def _lifecycle_to_status(self) -> MemoryStatus:
        mapping = {
            LifecycleState.HOT: MemoryStatus.ACTIVE,
            LifecycleState.WARM: MemoryStatus.ACTIVE,
            LifecycleState.COLD: MemoryStatus.DORMANT,
            LifecycleState.ARCHIVED: MemoryStatus.ARCHIVED,
            LifecycleState.DEEP_ARCHIVED: MemoryStatus.DEEP_ARCHIVED,
        }
        return mapping.get(self.lifecycle_state, MemoryStatus.ACTIVE)


__all__ = ["MemoryIR"]
