# Memory module init

from artificial_memory.memory.association import AssociationEngine, create_association_engine
from artificial_memory.memory.compiler import IncrementalMemoryCompiler, create_memory_compiler
from artificial_memory.memory.confidence import (
    ConfidenceEngine,
    ConfidenceLevel,
    ConfidenceScore,
    create_confidence_engine,
)
from artificial_memory.memory.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationScheduler,
    create_consolidation_engine,
    create_consolidation_scheduler,
)
from artificial_memory.memory.file_writer import ConversationFileWriter, MemoryFileWriter
from artificial_memory.memory.style import (
    ConversationStyle,
    StyleEngine,
    StyleProfile,
    create_style_engine,
)
from artificial_memory.memory.temporal import (
    TemporalEngine,
    TemporalQuery,
    TemporalState,
    create_temporal_engine,
)

# Heavy modules (faiss / sentence-transformers / torch) are exported lazily
# so that importing `artificial_memory.memory.<light_module>` stays fast.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "VectorSearchEngine": ("artificial_memory.memory.vector_search", "VectorSearchEngine"),
    "VectorSearchResult": ("artificial_memory.memory.vector_search", "VectorSearchResult"),
    "create_vector_search_engine": (
        "artificial_memory.memory.vector_search",
        "create_vector_search_engine",
    ),
}


def __getattr__(name: str):
    mapping = _LAZY_EXPORTS.get(name)
    if mapping is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = mapping
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)

__all__ = [
    "IncrementalMemoryCompiler",
    "create_memory_compiler",
    "ConsolidationEngine",
    "ConsolidationScheduler",
    "ConsolidationConfig",
    "create_consolidation_engine",
    "create_consolidation_scheduler",
    "MemoryFileWriter",
    "ConversationFileWriter",
    "AssociationEngine",
    "create_association_engine",
    "TemporalEngine",
    "TemporalState",
    "TemporalQuery",
    "create_temporal_engine",
    "ConfidenceEngine",
    "ConfidenceScore",
    "ConfidenceLevel",
    "create_confidence_engine",
    "StyleEngine",
    "ConversationStyle",
    "StyleProfile",
    "create_style_engine",
    "VectorSearchEngine",
    "VectorSearchResult",
    "create_vector_search_engine",
]
