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
from artificial_memory.memory.contradiction_edges import (
    ContradictionEdge,
    ContradictionEdgeManager,
    create_contradiction_edge_manager,
)
from artificial_memory.memory.evidence import (
    EvidenceItem,
    EvidenceRankConfig,
    EvidenceRanker,
    create_evidence_ranker,
)
from artificial_memory.memory.evolution import (
    EvolutionEvent,
    EvolutionOperationType,
    MemoryEvolutionEngine,
    create_memory_evolution_engine,
)
from artificial_memory.memory.evolution_policy import (
    EvolutionGovernor,
    EvolutionPolicyConfig,
    EvolutionProposal,
    PolicyDecision,
    PolicyVerdict,
    create_evolution_governor,
)
from artificial_memory.memory.file_writer import ConversationFileWriter, MemoryFileWriter
from artificial_memory.memory.graph import (
    GraphNode,
    GraphTraversalConfig,
    MemoryGraph,
    create_memory_graph,
)
from artificial_memory.memory.lifecycle_transitions import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    LifecycleTransitionEngine,
    create_lifecycle_transition_engine,
)
from artificial_memory.memory.provenance import (
    ProvenanceChain,
    ProvenanceChainBuilder,
    ProvenanceNode,
    create_provenance_chain_builder,
)
from artificial_memory.memory.reconstruction import (
    ConflictContext,
    MemoryReconstructor,
    ProvenanceTrace,
    ReconstructedEvidencePackage,
    ReconstructionConfig,
    create_memory_reconstructor,
)
from artificial_memory.memory.reflection import (
    BackgroundReflector,
    ReflectionConfig,
    ReflectionFinding,
    ReflectionKind,
    ReflectionReport,
    ReflectionTrigger,
    create_background_reflector,
)
from artificial_memory.memory.state_signals import (
    MemoryStateVector,
    StateSignalCalculator,
    StateSignalConfig,
    create_state_signal_calculator,
)
from artificial_memory.memory.style import (
    ConversationStyle,
    StyleEngine,
    StyleProfile,
    create_style_engine,
)
from artificial_memory.memory.supersession import (
    SupersessionManager,
    create_supersession_manager,
)
from artificial_memory.memory.temporal import (
    TemporalEngine,
    TemporalQuery,
    TemporalState,
    create_temporal_engine,
)
from artificial_memory.memory.temporal_validity import (
    TemporalValidityManager,
    ValidityInterval,
    create_temporal_validity_manager,
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
    "MemoryEvolutionEngine",
    "EvolutionEvent",
    "EvolutionOperationType",
    "create_memory_evolution_engine",
    "EvolutionGovernor",
    "EvolutionPolicyConfig",
    "EvolutionProposal",
    "PolicyDecision",
    "PolicyVerdict",
    "create_evolution_governor",
    "MemoryStateVector",
    "StateSignalCalculator",
    "StateSignalConfig",
    "create_state_signal_calculator",
    "TemporalValidityManager",
    "ValidityInterval",
    "create_temporal_validity_manager",
    "SupersessionManager",
    "create_supersession_manager",
    "ContradictionEdge",
    "ContradictionEdgeManager",
    "create_contradiction_edge_manager",
    "LifecycleTransitionEngine",
    "InvalidTransitionError",
    "ALLOWED_TRANSITIONS",
    "create_lifecycle_transition_engine",
    "ProvenanceChainBuilder",
    "ProvenanceChain",
    "ProvenanceNode",
    "create_provenance_chain_builder",
    "MemoryGraph",
    "GraphNode",
    "GraphTraversalConfig",
    "create_memory_graph",
    "EvidenceRanker",
    "EvidenceItem",
    "EvidenceRankConfig",
    "create_evidence_ranker",
    "MemoryReconstructor",
    "BackgroundReflector",
    "ReflectionConfig",
    "ReflectionFinding",
    "ReflectionKind",
    "ReflectionReport",
    "ReflectionTrigger",
    "create_background_reflector",
    "ReconstructedEvidencePackage",
    "ReconstructionConfig",
    "ConflictContext",
    "ProvenanceTrace",
    "create_memory_reconstructor",
]
