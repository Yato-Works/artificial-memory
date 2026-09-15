from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IRType(StrEnum):
    TONE = "tone"
    STATE = "state"
    DECISION = "decision"
    REASONING = "reasoning"
    TEMPORAL = "temporal"
    KNOWLEDGE = "knowledge"
    EVENT = "event"
    ROLE = "role"
    TOPIC = "topic"
    ENTITY = "entity"
    ACTION = "action"
    CONCERN = "concern"
    TRADEOFF = "tradeoff"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    CLARIFICATION = "clarification"
    SUMMARY = "summary"


class CompressionMethod(StrEnum):
    LIGHT = "light"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    LONG_TERM = "longterm"
    DEEP_LONG_TERM = "deeplongterm"


class Project(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    name: str
    display_name: str | None = None
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True


class Topic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    project_id: int
    parent_id: int | None = None
    name: str
    path: str
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def full_path(self) -> str:
        return self.path


class Conversation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    topic_id: int
    title: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    message_count: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.ACTIVE


class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    conversation_id: int
    role: MessageRole
    content: str
    token_count: int = 0
    sequence_num: int
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    topic_id: int
    memory_type: MemoryType
    content: str
    resolution: ResolutionLevel = ResolutionLevel.RAW
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    is_current: bool = True
    source_conversation_id: int | None = None
    source_message_id: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    access_count: int = 0

    def touch(self) -> None:
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.updated_at = datetime.now()


class MemoryVersion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    memory_id: int
    resolution: ResolutionLevel
    content: str
    compression_ratio: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = "auto"
    # P0-7: provenance for compression decisions - which policy and algorithm
    # produced this version. Required for full traceability / reproducibility.
    policy_version: str = "decision-v1"
    algorithm_version: str = "rule-compressor-v1"


class EvolutionEventRecord(BaseModel):
    """Persisted record of a memory evolution operation (P0-5).

    Mirrors memory.evolution.EvolutionEvent but is a storage-backed model.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    memory_id: int
    operation: str  # EvolutionOperationType value: revise/merge/split/...
    source_memory_ids: list[int] = Field(default_factory=list)
    target_memory_id: int | None = None
    description: str = ""
    old_content: str = ""
    new_content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    triggered_by: str = "auto"
    created_at: datetime = Field(default_factory=datetime.now)


class BeliefState(BaseModel):
    """Persisted belief state (P0-4). Evidence lists stored as JSON/JSONB.

    V1 stores evidence ids as JSON; V2 may normalize into
    belief_evidence / belief_conflict / belief_revision tables.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    belief_key: str  # proposition hash for deduplication
    proposition: str
    status: str = "accepted"  # BeliefStatus value
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence: list[int] = Field(default_factory=list)
    contradicting_evidence: list[int] = Field(default_factory=list)
    source_memories: list[int] = Field(default_factory=list)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None
    revision: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Decision(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    topic_id: int
    memory_id: int | None = None
    decision_text: str
    rejected_options: list[str] = Field(default_factory=list)
    reason: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    decided_at: datetime = Field(default_factory=datetime.now)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None
    is_current: bool = True


class Association(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    source_memory_id: int
    target_memory_id: int
    association_type: AssociationType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)


class RecallEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    query: str
    topic_id: int | None = None
    recall_level: RecallLevel
    memories_retrieved: int = 0
    tokens_returned: int = 0
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class CompressionEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    source_memory_id: int
    target_memory_id: int | None = None
    from_resolution: ResolutionLevel
    to_resolution: ResolutionLevel
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: CompressionMethod
    created_at: datetime = Field(default_factory=datetime.now)


class ContextIR(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    conversation_id: int
    ir_type: IRType
    ir_key: str
    ir_value: str | None = None
    sequence_num: int
    created_at: datetime = Field(default_factory=datetime.now)


class TokenUsage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    conversation_id: int | None = None
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
