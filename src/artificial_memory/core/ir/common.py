from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryType(StrEnum):
    CURRENT = "current"
    TIMELINE = "timeline"
    DECISION = "decision"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    CONVERSATION_STYLE = "conversation_style"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DEEP_ARCHIVED = "deep_archived"


class ResolutionLevel(int, Enum):
    RAW = 0
    LIGHT = 1
    EPISODE = 2
    SEMANTIC = 3
    LONG_TERM = 4
    DEEP_LONG_TERM = 5


class RecallLevel(int, Enum):
    CURRENT_ONLY = 0
    LONG_TERM_SUMMARY = 1
    EPISODE = 2
    LIGHT_COMPRESSION = 3
    RAW = 4


class AssociationType(StrEnum):
    RELATED = "related"
    CAUSES = "causes"
    FOLLOWS = "follows"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    SUMMARIZES = "summarizes"


class CompressionMethod(StrEnum):
    LIGHT = "light"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    LONG_TERM = "longterm"
    DEEP_LONG_TERM = "deeplongterm"


class LifecycleState(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"
    DEEP_ARCHIVED = "deep_archived"


class BeliefStatus(StrEnum):
    ACCEPTED = "accepted"
    CONTESTED = "contested"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class PriorityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MemoryIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    memory_id: int
    version: int
    content_hash: str


class SemanticContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    content: str
    structured_data: dict[str, Any] | None = None


class ProvenanceLink(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    conversation_id: int | None = None
    message_id: int | None = None
    timestamp: datetime
    stage: str | None = None
    compiler_version: str | None = None


class ProvenanceChain(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    source: ProvenanceLink
    compilation_chain: list[ProvenanceLink] = Field(default_factory=list)


class TemporalScope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ConfidenceProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    memory_confidence: float = Field(ge=0.0, le=1.0)
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    temporal_confidence: float = Field(ge=0.0, le=1.0)
    source_confidence: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)

    # Weights for combining the four components into `overall`.
    # P0-3 fix: overall must be derived from its components, not passed through.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "memory_confidence": 0.40,
        "retrieval_confidence": 0.25,
        "temporal_confidence": 0.20,
        "source_confidence": 0.15,
    }

    @model_validator(mode="after")
    def _compute_overall(self) -> ConfidenceProfile:
        weighted = sum(
            getattr(self, component) * weight
            for component, weight in self.WEIGHTS.items()
        )
        self.overall = max(0.0, min(1.0, weighted))
        return self


class DependencyRef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    memory_id: int
    dependency_type: AssociationType
    strength: float = Field(ge=0.0, le=1.0)


class AssociationRef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    target_memory_id: int
    association_type: AssociationType
    strength: float = Field(ge=0.0, le=1.0)


class CompressionRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    from_resolution: ResolutionLevel
    to_resolution: ResolutionLevel
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: CompressionMethod
    timestamp: datetime


class AccessRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    last_accessed: datetime | None = None
    access_count: int = 0
    total_tokens_retrieved: int = 0


class BeliefRef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    belief_id: int
    proposition: str
    belief_status: BeliefStatus
    confidence: float = Field(ge=0.0, le=1.0)
