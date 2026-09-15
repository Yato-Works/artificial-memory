"""Artificial Memory / Context Runtime Package."""

__version__ = "0.1.0"
__author__ = "Artificial Memory Project"

from artificial_memory.core.interfaces import (
    Compressor,
    ContextBuilder,
    ContextIRCompiler,
    MemoryCompiler,
    MemoryStore,
    RecallEngine,
    TopicClassifier,
)
from artificial_memory.core.models import (
    Association,
    AssociationType,
    CompressionEvent,
    CompressionMethod,
    ContextIR,
    Conversation,
    ConversationStatus,
    Decision,
    IRType,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Message,
    MessageRole,
    Project,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
    TokenUsage,
    Topic,
)

__all__ = [
    # Models
    "Project", "Topic", "Conversation", "Message", "Memory",
    "MemoryVersion", "Decision", "Association", "RecallEvent",
    "CompressionEvent", "ContextIR", "TokenUsage",
    # Enums
    "MemoryType", "MemoryStatus", "ResolutionLevel", "RecallLevel",
    "AssociationType", "ConversationStatus", "MessageRole", "IRType",
    "CompressionMethod",
    # Interfaces
    "MemoryStore", "TopicClassifier", "Compressor", "RecallEngine",
    "ContextBuilder", "MemoryCompiler", "ContextIRCompiler",
]
