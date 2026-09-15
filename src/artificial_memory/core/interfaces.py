from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol

from artificial_memory.core.models import (
    Association,
    AssociationType,
    CompressionEvent,
    ContextIR,
    Conversation,
    ConversationStatus,
    Decision,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Message,
    Project,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
    TokenUsage,
    Topic,
)


class MemoryStore(Protocol):
    """Low-level storage interface for memories and related data."""

    # Project operations
    @abstractmethod
    def create_project(self, project: Project) -> Project: ...

    @abstractmethod
    def get_project(self, project_id: int) -> Project | None: ...

    @abstractmethod
    def get_project_by_name(self, name: str) -> Project | None: ...

    @abstractmethod
    def list_projects(self, active_only: bool = True) -> list[Project]: ...

    @abstractmethod
    def update_project(self, project: Project) -> Project: ...

    # Topic operations
    @abstractmethod
    def create_topic(self, topic: Topic) -> Topic: ...

    @abstractmethod
    def get_topic(self, topic_id: int) -> Topic | None: ...

    @abstractmethod
    def get_topic_by_path(self, path: str) -> Topic | None: ...

    @abstractmethod
    def list_topics(self, project_id: int | None = None, parent_id: int | None = None) -> list[Topic]: ...

    @abstractmethod
    def update_topic(self, topic: Topic) -> Topic: ...

    # Conversation operations
    @abstractmethod
    def create_conversation(self, conversation: Conversation) -> Conversation: ...

    @abstractmethod
    def get_conversation(self, conversation_id: int) -> Conversation | None: ...

    @abstractmethod
    def list_conversations(self, topic_id: int | None = None, status: ConversationStatus | None = None) -> list[Conversation]: ...

    @abstractmethod
    def update_conversation(self, conversation: Conversation) -> Conversation: ...

    @abstractmethod
    def end_conversation(self, conversation_id: int) -> Conversation: ...

    # Message operations
    @abstractmethod
    def add_message(self, message: Message) -> Message: ...

    @abstractmethod
    def get_messages(self, conversation_id: int, limit: int | None = None, offset: int = 0) -> list[Message]: ...

    @abstractmethod
    def get_message_count(self, conversation_id: int) -> int: ...

    # Memory operations
    @abstractmethod
    def create_memory(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def get_memory(self, memory_id: int) -> Memory | None: ...

    @abstractmethod
    def get_memories(
        self,
        topic_id: int | None = None,
        memory_type: MemoryType | None = None,
        resolution: ResolutionLevel | None = None,
        status: MemoryStatus | None = None,
        is_current: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]: ...

    @abstractmethod
    def update_memory(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def delete_memory(self, memory_id: int) -> bool: ...

    # Memory version operations
    @abstractmethod
    def add_memory_version(self, version: MemoryVersion) -> MemoryVersion: ...

    @abstractmethod
    def get_memory_versions(self, memory_id: int) -> list[MemoryVersion]: ...

    @abstractmethod
    def get_memory_version(self, memory_id: int, resolution: ResolutionLevel) -> MemoryVersion | None: ...

    # Decision operations
    @abstractmethod
    def create_decision(self, decision: Decision) -> Decision: ...

    @abstractmethod
    def get_decisions(self, topic_id: int, current_only: bool = True) -> list[Decision]: ...

    @abstractmethod
    def update_decision(self, decision: Decision) -> Decision: ...

    # Association operations
    @abstractmethod
    def create_association(self, association: Association) -> Association: ...

    @abstractmethod
    def get_associations(self, memory_id: int, association_type: AssociationType | None = None) -> list[Association]: ...

    @abstractmethod
    def get_related_memories(self, memory_id: int, min_strength: float = 0.3) -> list[tuple[Memory, Association]]: ...

    # Recall event operations
    @abstractmethod
    def log_recall(self, recall: RecallEvent) -> RecallEvent: ...

    @abstractmethod
    def get_recent_recalls(self, topic_id: int | None = None, limit: int = 50) -> list[RecallEvent]: ...

    # Compression event operations
    @abstractmethod
    def log_compression(self, event: CompressionEvent) -> CompressionEvent: ...

    @abstractmethod
    def get_compression_history(self, memory_id: int) -> list[CompressionEvent]: ...

    # Context IR operations
    @abstractmethod
    def add_context_ir(self, ir: ContextIR) -> ContextIR: ...

    @abstractmethod
    def get_context_ir(self, conversation_id: int) -> list[ContextIR]: ...

    # Token usage operations
    @abstractmethod
    def log_token_usage(self, usage: TokenUsage) -> TokenUsage: ...

    @abstractmethod
    def get_token_usage(self, conversation_id: int | None = None, operation: str | None = None) -> list[TokenUsage]: ...

    # Health/management
    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...


class TopicClassifier(Protocol):
    """Interface for classifying conversations/messages into topics."""

    @abstractmethod
    def classify(self, text: str, existing_topics: list[Topic]) -> tuple[Topic | None, float]:
        """Returns (topic, confidence) or (None, 0.0) if no match."""
        ...

    @abstractmethod
    def suggest_new_topic(self, text: str, project_name: str) -> Topic: ...

    @abstractmethod
    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[tuple[str, float]]: ...


class Compressor(Protocol):
    """Interface for compressing memories at different resolutions."""

    @abstractmethod
    def compress_light(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Light compression: almost original, just remove redundancy."""
        ...

    @abstractmethod
    def compress_episode(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Episode compression: 'what happened in this conversation'."""
        ...

    @abstractmethod
    def compress_semantic(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Semantic compression: 'what was decided/resulted'."""
        ...

    @abstractmethod
    def compress_long_term(self, content: str, metadata: dict) -> tuple[str, dict]:
        """Long-term compression: abstract summary for long-term retention."""
        ...

    @abstractmethod
    def get_compression_ratio(self, original: str, compressed: str) -> float: ...


class RecallEngine(Protocol):
    """Interface for progressive recall of memories."""

    @abstractmethod
    def recall(
        self,
        query: str,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], int]:  # (memories, total_tokens)
        ...

    @abstractmethod
    def expand_resolution(self, memory: Memory, target_resolution: ResolutionLevel) -> Memory | None: ...

    @abstractmethod
    def get_memory_provenance(self, memory: Memory) -> list[Memory]: ...

    @abstractmethod
    def search_by_keywords(self, keywords: list[str], topic_id: int | None = None, limit: int = 20) -> list[Memory]: ...

    @abstractmethod
    def search_by_time_range(
        self,
        start: datetime,
        end: datetime,
        topic_id: int | None = None,
        limit: int = 20
    ) -> list[Memory]: ...

    @abstractmethod
    def search_associations(self, memory: Memory, max_depth: int = 2, min_strength: float = 0.3) -> list[Memory]: ...


class ContextBuilder(Protocol):
    """Interface for building optimized context for LLM."""

    @abstractmethod
    def build_context(
        self,
        query: str,
        topic_id: int | None = None,
        max_tokens: int = 8000,
        current_memories: list[Memory] | None = None,
    ) -> str: ...

    @abstractmethod
    def optimize_context(
        self,
        context_parts: list[tuple[str, int, float]],  # (content, tokens, priority)
        max_tokens: int,
    ) -> str: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def get_context_stats(self) -> dict[str, int]: ...


class MemoryCompiler(Protocol):
    """Incremental compiler that processes conversations into memories."""

    @abstractmethod
    def process_message(
        self,
        message: Message,
        conversation: Conversation,
        current_memories: list[Memory],
    ) -> list[Memory]: ...

    @abstractmethod
    def compile_conversation(self, conversation: Conversation) -> list[Memory]: ...

    @abstractmethod
    def consolidate_topic(self, topic_id: int) -> list[Memory]: ...

    @abstractmethod
    def run_consolidation_cycle(self) -> dict[str, int]: ...


class ContextIRCompiler(Protocol):
    """Compiler from natural language to Context IR."""

    @abstractmethod
    def compile_to_ir(self, conversation: Conversation) -> list[ContextIR]: ...

    @abstractmethod
    def decompile_from_ir(self, ir_units: list[ContextIR]) -> str: ...

    @abstractmethod
    def optimize_ir(self, ir_units: list[ContextIR], max_units: int) -> list[ContextIR]: ...


class VectorSearchEngine(Protocol):
    """Interface for vector search engines."""

    @abstractmethod
    def add_memory(self, memory: Memory) -> int: ...

    @abstractmethod
    def search(
        self,
        query: str,
        topic_id: int | None = None,
        memory_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[Any]: ...

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        topic_id: int | None = None,
        k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[Any]: ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]: ...
