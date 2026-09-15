from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artificial_memory.core.interfaces import MemoryStore, VectorSearchEngine

from artificial_memory.auth import AuthService
from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.context.builder import EnhancedContextBuilder
from artificial_memory.context.ir_compiler import create_ir_compiler
from artificial_memory.core.adapters import create_memory_ir_adapter
from artificial_memory.core.interfaces import ContextIRCompiler
from artificial_memory.core.ir import (
    BudgetAllocation,
    ContextIR,
    MemoryIR,
    PriorityTier,
)
from artificial_memory.core.models import (
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    RecallLevel,
    ResolutionLevel,
)
from artificial_memory.governance.audit import (
    create_audit_logger,
)
from artificial_memory.governance.engine import (
    create_governance_engine,
)
from artificial_memory.governance.retention import (
    create_retention_policy_engine,
)
from artificial_memory.governance.tenancy import (
    create_tenancy_manager,
)
from artificial_memory.governance.trust import (
    create_trust_policy_engine,
)
from artificial_memory.llm.manager import LLMManager, OllamaProvider
from artificial_memory.memory.association import AssociationEngine
from artificial_memory.memory.belief import Belief, BeliefConflict, BeliefEngine
from artificial_memory.memory.compiler import CompilerPipelineWrapper, IncrementalMemoryCompiler
from artificial_memory.memory.confidence import ConfidenceEngine
from artificial_memory.memory.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationScheduler,
)
from artificial_memory.memory.contradiction import (
    ContradictionDetector,
    ContradictionPair,
)
from artificial_memory.memory.counterfactual import (
    CounterfactualEngine,
    CounterfactualOperation,
    CounterfactualResult,
    CounterfactualScenario,
)
from artificial_memory.memory.debugger import (
    CompilationTrace,
    ContextTrace,
    MemoryDebugger,
    RecallTrace,
)
from artificial_memory.memory.dependency import (
    Dependency,
    DependencyGraph,
    DependencyType,
    ImpactAnalysis,
)

# Phase 3: Memory Evolution & Belief
from artificial_memory.memory.evolution import MemoryEvolutionEngine
from artificial_memory.memory.exchange import (
    create_memory_exchange_protocol,
)

# Phase 7: Multi-Agent & Enterprise
from artificial_memory.memory.federation import (
    FederationConfig,
    create_federated_memory_engine,
)
from artificial_memory.memory.healing import (
    HealingResult,
    MemoryHealer,
)

# Phase 4: Memory Integrity & Debugging
from artificial_memory.memory.integrity import (
    IntegrityMetrics,
    IntegrityReport,
)
from artificial_memory.memory.stale import StaleMemoryDetector, StalenessAssessment
from artificial_memory.memory.style import StyleEngine
from artificial_memory.memory.temporal import TemporalEngine
from artificial_memory.memory.validation import (
    CompressionValidationReport,
    CompressionValidator,
)
from artificial_memory.observability.metrics import (
    create_metrics_collector,
)
from artificial_memory.recall.adaptive import AdaptiveRecallEngine, RecallBudget, UtilityScore
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.recall.human_recall import HumanRecallEngine
from artificial_memory.research.ablation.framework import (
    AblationConfig,
    AblationExperimentResult,
)
from artificial_memory.research.benchmarks.am_benchmarks import (
    BenchmarkCategory,
    BenchmarkResult,
)
from artificial_memory.research.experiments.manifest import ExperimentManifest
from artificial_memory.research.redteam.suite import (
    AttackConfig,
    AttackResult,
)
from artificial_memory.storage.conversation_logger import ConversationManager
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore

# Phase 5: Temporal & Debugging Research
from artificial_memory.temporal import (
    BeliefEvolution,
    BeliefHistoryEngine,
    BeliefTimeline,
    ChangeSimulation,
    ContextReconstructionConfig,
    ContextReconstructor,
    DecisionTracer,
    DecisionTraceReport,
    DependencyCycle,
    HistoricalBelief,
    ImpactAnalyzer,
    ImpactPrediction,
    ReconstructedContext,
    TemporalQuery,
    TemporalQueryType,
    TemporalState,
    TimeTravelEngine,
    TimeTravelResult,
)
from artificial_memory.topic.classifier import RuleBasedTopicClassifier


@dataclass
class RuntimeConfig:
    """Configuration for ArtificialMemoryRuntime.

    Security (P0-1): no secrets are hard-coded.jwt_secret, private_key and
    public_key must be provided via configuration or environment variables.
    If omitted, ephemeral values are generated at startup (fine for local
    development / research; NOT suitable for production or persistence
    across restarts).
    """
    database_path: str = "memory.db"
    use_postgres: bool = False
    database_url: str | None = None
    memory_files_path: str = "memory_files"
    vector_index_path: str = "vector_index"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    jwt_secret: str | None = None
    default_provider: str = "ollama"
    max_context_tokens: int = 8000
    enable_auth: bool = False

    # Phase 7: Multi-Agent & Enterprise
    node_id: str = "local"
    node_name: str = "local-node"
    private_key: str | None = None
    public_key: str | None = None
    enable_federation: bool = False
    enable_governance: bool = True
    enable_audit: bool = True
    enable_retention: bool = True
    enable_metrics: bool = True
    enable_multitenancy: bool = False
    audit_log_dir: str = "audit_logs"
    max_audit_file_mb: int = 100
    max_audit_files: int = 100
    metrics_output_dir: str = "metrics"

    def __post_init__(self) -> None:
        import secrets as _secrets
        import warnings as _warnings

        if self.jwt_secret is None:
            if self.enable_auth:
                _warnings.warn(
                    "RuntimeConfig.jwt_secret is not set while auth is enabled; "
                    "generating an ephemeral secret. Sessions will be invalidated "
                    "on restart. Set jwt_secret explicitly for production.",
                    stacklevel=2,
                )
            self.jwt_secret = _secrets.token_urlsafe(32)

        if self.private_key is None or self.public_key is None:
            if self.enable_federation:
                _warnings.warn(
                    "RuntimeConfig.private_key/public_key are not set while "
                    "federation is enabled; generating ephemeral keys. Set them "
                    "explicitly for production federation.",
                    stacklevel=2,
                )
            self.private_key = self.private_key or _secrets.token_urlsafe(32)
            self.public_key = self.public_key or _secrets.token_urlsafe(32)


@dataclass
class ChatResult:
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    context_used: bool
    topic_id: int | None
    context_ir: ContextIR | None = None


@dataclass
class RecallResult:
    query: str
    level: RecallLevel
    memories_retrieved: int
    tokens: int
    memories: list[MemoryIR]
    context_ir: ContextIR | None = None


@dataclass
class TimelineResult:
    topic: str
    events: list[dict[str, Any]]


@dataclass
class RecallExplanation:
    query: str
    mode: str
    resolution: str
    confidence: float
    memories_retrieved: int
    reasoning: str
    adaptations: list[dict[str, Any]]


@dataclass
class MemoryInspection:
    memory_id: int
    memory_ir: MemoryIR
    provenance_chain: list[MemoryIR]
    source_conversation: Conversation | None
    source_messages: list[Any]
    exact_source_message: Any | None


class ArtificialMemoryRuntime:
    """Unified facade for Artificial Memory Cognitive Runtime."""

    # Instance variable type annotations
    config: RuntimeConfig
    store: MemoryStore
    conversation_manager: ConversationManager
    topic_classifier: RuleBasedTopicClassifier
    compressor: RuleBasedCompressor
    recall_engine: BasicRecallEngine
    context_builder: EnhancedContextBuilder
    memory_compiler: IncrementalMemoryCompiler
    pipeline_wrapper: CompilerPipelineWrapper
    association_engine: AssociationEngine
    temporal_engine: TemporalEngine
    confidence_engine: ConfidenceEngine
    style_engine: StyleEngine
    human_recall_engine: HumanRecallEngine
    vector_search_engine: VectorSearchEngine
    ir_compiler: ContextIRCompiler
    memory_ir_adapter: Any
    consolidation_engine: ConsolidationEngine
    consolidation_scheduler: ConsolidationScheduler
    evolution_engine: MemoryEvolutionEngine
    belief_engine: BeliefEngine
    contradiction_detector: ContradictionDetector
    dependency_graph: DependencyGraph
    counterfactual_engine: CounterfactualEngine
    integrity_metrics: IntegrityMetrics
    stale_detector: StaleMemoryDetector
    compression_validator: CompressionValidator
    healer: MemoryHealer
    debugger: MemoryDebugger
    time_travel_engine: TimeTravelEngine
    belief_history_engine: BeliefHistoryEngine
    context_reconstructor: ContextReconstructor
    decision_tracer: DecisionTracer
    impact_analyzer: ImpactAnalyzer
    federated_engine: Any
    exchange_protocol: Any
    trust_engine: Any
    tenancy_manager: Any
    governance_engine: Any
    audit_logger: Any
    retention_engine: Any
    metrics_collector: Any
    auth_service: Any | None
    llm_manager: LLMManager

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        self._initialize_components()

    def _initialize_components(self) -> None:
        # Storage
        if self.config.use_postgres:
            from artificial_memory.storage.postgres_store import create_postgres_store
            url = self.config.database_url or os.environ.get(
                "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/artificial_memory"
            )
            self.store = create_postgres_store(url)
        else:
            self.store = SQLiteMemoryStore(self.config.database_path)

        # Core managers
        self.conversation_manager = ConversationManager(self.store)
        self.topic_classifier = RuleBasedTopicClassifier(self.store)
        self.compressor = RuleBasedCompressor()
        self.recall_engine = BasicRecallEngine(self.store)
        self.context_builder = EnhancedContextBuilder(
            self.store, self.recall_engine,
            budget_ratios={
                PriorityTier.HIGH: 0.50,
                PriorityTier.MEDIUM: 0.35,
                PriorityTier.LOW: 0.15,
            }
        )
        self.memory_compiler = IncrementalMemoryCompiler(
            self.store, self.compressor, self.topic_classifier
        )
        self.pipeline_wrapper = CompilerPipelineWrapper(
            self.store, self.compressor, self.topic_classifier
        )

        # Advanced engines
        self.association_engine = AssociationEngine(self.store)
        self.temporal_engine = TemporalEngine(self.store)
        self.confidence_engine = ConfidenceEngine(self.store, self.recall_engine)
        self.style_engine = StyleEngine(self.store)
        self.human_recall_engine = HumanRecallEngine(self.store, self.recall_engine)
        if self.config.use_postgres:
            try:
                from artificial_memory.memory.pgvector_search import create_pgvector_search_engine
                from artificial_memory.storage.postgres_store import PostgresMemoryStore
                if isinstance(self.store, PostgresMemoryStore):
                    self.vector_search_engine = create_pgvector_search_engine(self.store)
                else:
                    from artificial_memory.memory.vector_search import (
                        VectorSearchEngine as FAISSVectorSearchEngine,
                    )
                    self.vector_search_engine = FAISSVectorSearchEngine(
                        self.store, model_name=self.config.embedding_model,
                        index_path=Path(self.config.vector_index_path)
                    )
            except ImportError:
                from artificial_memory.memory.vector_search import (
                    VectorSearchEngine as FAISSVectorSearchEngine,
                )
                self.vector_search_engine = FAISSVectorSearchEngine(
                    self.store, model_name=self.config.embedding_model,
                    index_path=Path(self.config.vector_index_path)
                )
        else:
            from artificial_memory.memory.vector_search import (
                VectorSearchEngine as FAISSVectorSearchEngine,
            )
            self.vector_search_engine = FAISSVectorSearchEngine(
                self.store, model_name=self.config.embedding_model,
                index_path=Path(self.config.vector_index_path)
            )

        # IR components
        self.ir_compiler = create_ir_compiler()
        self.memory_ir_adapter = create_memory_ir_adapter(self.store)

        # Consolidation
        self.consolidation_engine = ConsolidationEngine(
            self.store, self.compressor, ConsolidationConfig()
        )
        self.consolidation_scheduler = ConsolidationScheduler(self.consolidation_engine)

        # Phase 3: Memory Evolution & Belief
        self.evolution_engine = MemoryEvolutionEngine(self.store, self.compressor)
        self.belief_engine = BeliefEngine(self.store)
        self.contradiction_detector = ContradictionDetector(self.store)
        self.dependency_graph = DependencyGraph(self.store)
        self.counterfactual_engine = CounterfactualEngine(
            self.store, self.recall_engine, self.context_builder, self.evolution_engine
        )

        # Phase 4: Memory Integrity & Debugging
        self.integrity_metrics = IntegrityMetrics(self.store, self.compressor)
        self.stale_detector = StaleMemoryDetector(self.store, self.recall_engine)
        self.compression_validator = CompressionValidator(self.store, self.compressor)
        self.healer = MemoryHealer(
            self.store, self.compressor, self.pipeline_wrapper.pipeline,
            self.belief_engine, self.evolution_engine, self.contradiction_detector
        )

        # Adaptive Recall (needed for debugger)
        self._adaptive_recall_engine = AdaptiveRecallEngine(self.store, self.recall_engine)

        self.debugger = MemoryDebugger(
            self.store, self.recall_engine, self.context_builder,
            self.pipeline_wrapper.pipeline, self.evolution_engine, self.belief_engine,
            self.contradiction_detector, self.dependency_graph, self.counterfactual_engine,
            self._adaptive_recall_engine, self.integrity_metrics, self.stale_detector, self.healer
        )

        # Phase 5: Temporal & Debugging Research
        self.time_travel_engine = TimeTravelEngine(
            self.store, self.belief_engine, self.recall_engine, self.context_builder
        )
        self.belief_history_engine = BeliefHistoryEngine(
            self.store, self.belief_engine, self.time_travel_engine
        )
        self.context_reconstructor = ContextReconstructor(
            self.store, self.recall_engine, self.context_builder
        )
        self.decision_tracer = DecisionTracer(self.store, self.belief_engine)
        self.impact_analyzer = ImpactAnalyzer(
            self.store, self.dependency_graph, self.evolution_engine, self.belief_engine
        )

        # Phase 7: Multi-Agent & Enterprise
        # Federation
        federation_config = FederationConfig(
            node_id=self.config.node_id if hasattr(self.config, 'node_id') else "local",
            node_name=self.config.node_name if hasattr(self.config, 'node_name') else "local-node",
        )
        self.federated_engine = create_federated_memory_engine(
            self.store, federation_config, self.config.node_id,
            self.config.private_key or "", self.config.public_key or "",
        )
        self.exchange_protocol = create_memory_exchange_protocol(self.federated_engine)

        # Trust & Governance
        self.trust_engine = create_trust_policy_engine(self.store)
        self.tenancy_manager = create_tenancy_manager(self.store)
        self.governance_engine = create_governance_engine(self.store, self.trust_engine, self.tenancy_manager)
        self.audit_logger = create_audit_logger(self.store)
        self.retention_engine = create_retention_policy_engine(self.store, self.compressor, self.consolidation_engine)

        # Observability
        self.metrics_collector = create_metrics_collector()

        # Research components (import at runtime)
        from artificial_memory.research.ablation.framework import AblationFramework
        from artificial_memory.research.benchmarks.am_benchmarks import AMBenchmarkSuite
        from artificial_memory.research.experiments.runner import ExperimentRunner
        from artificial_memory.research.redteam.suite import RedTeamSuite

        self.am_benchmark_suite = AMBenchmarkSuite(
            self.store, self.recall_engine, self.context_builder, self.compressor,
            time_travel_engine=self.time_travel_engine,
            belief_engine=self.belief_engine,
            evolution_engine=self.evolution_engine,
            contradiction_detector=self.contradiction_detector,
            dependency_graph=self.dependency_graph,
            adaptive_recall=self._adaptive_recall_engine,
        )
        self.ablation_framework = AblationFramework(
            self.store, self.recall_engine, self.context_builder, self.compressor
        )
        self.redteam_suite = RedTeamSuite(
            self.store, self.recall_engine, self.context_builder, self.compressor,
            belief_engine=self.belief_engine,
            evolution_engine=self.evolution_engine,
            contradiction_detector=self.contradiction_detector,
        )
        self.experiment_runner = ExperimentRunner()

        # Auth (optional)
        self.auth_service = (
            AuthService(self.store, jwt_secret=self.config.jwt_secret)
            if self.config.enable_auth else None
        )

        # LLM Manager
        self.llm_manager = LLMManager(self.store, self.recall_engine, self.context_builder)
        self.llm_manager.add_provider("ollama", OllamaProvider(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model
        ))
        if self.config.openai_api_key:
            from artificial_memory.llm.manager import OpenAIProvider
            self.llm_manager.add_provider("openai", OpenAIProvider(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model
            ))

    # ==================== Public API ====================

    async def chat(
        self,
        message: str,
        topic: str,
        conversation_id: int | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        use_context: bool = True,
        recall_level: int = 2,
    ) -> ChatResult:
        """Chat with LLM using full Context Runtime."""
        result = await self.llm_manager.chat(
            user_message=message,
            topic_path=topic,
            conversation_id=conversation_id,
            provider_name=provider or self.config.default_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            use_context=use_context,
            recall_level=recall_level,
        )

        # Build ContextIR for inspection/debugging
        context_ir = None
        topic_id = result.get("topic_id")
        if use_context and topic_id:
            current_memories = self.store.get_memories(
                topic_id=topic_id,
                is_current=True,
                status=MemoryStatus.ACTIVE
            )
            self.context_builder.build_context(
                message, topic_id, self.config.max_context_tokens, current_memories
            )
            stats = self.context_builder.get_context_stats()
            budget = BudgetAllocation(
                high=stats.budget_allocation.high if stats.budget_allocation else 0,
                medium=stats.budget_allocation.medium if stats.budget_allocation else 0,
                low=stats.budget_allocation.low if stats.budget_allocation else 0,
            )
            system_prompt = self.context_builder.build_system_prompt(topic)
            context_ir = ContextIR(
                query=message,
                budget=budget,
                parts=[],  # Would be populated from builder internals
                stats=stats,
                system_prompt=system_prompt,
            )

        return ChatResult(
            response=result["response"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            model=result["model"],
            context_used=result["context_used"],
            topic_id=result.get("topic_id"),
            context_ir=context_ir,
        )

    async def remember(
        self,
        content: str,
        topic: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        resolution: ResolutionLevel = ResolutionLevel.SEMANTIC,
        importance: float = 0.7,
        confidence: float = 0.8,
    ) -> MemoryIR:
        """Explicitly store a memory."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)

        memory = Memory(
            topic_id=topic_obj.id,
            memory_type=memory_type,
            content=content,
            resolution=resolution,
            importance=importance,
            confidence=confidence,
            status=MemoryStatus.ACTIVE,
            valid_from=__import__('datetime').datetime.now(),
            is_current=True,
        )
        memory = self.store.create_memory(memory)

        # Create compressed versions
        self.memory_compiler._create_compressed_versions(memory)

        # Convert to IR
        return self.memory_ir_adapter.to_ir(memory)

    async def recall(
        self,
        query: str,
        topic: str,
        level: int = 2,
        max_tokens: int = 4000,
    ) -> RecallResult:
        """Recall memories with full provenance."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        recall_level = RecallLevel(level)

        memories, tokens = self.recall_engine.recall(
            query, topic_obj.id, recall_level, max_tokens
        )

        # Convert to IR
        memory_irs = self.memory_ir_adapter.batch_to_ir(memories)

        # Build context for provenance
        context_ir = self._build_context_ir(query, topic_obj.id, max_tokens)

        return RecallResult(
            query=query,
            level=recall_level,
            memories_retrieved=len(memories),
            tokens=tokens,
            memories=memory_irs,
            context_ir=context_ir,
        )

    async def expand(
        self,
        memory_id: int,
        target_resolution: ResolutionLevel,
    ) -> MemoryIR:
        """Expand memory to higher resolution (lower number)."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        expanded = self.recall_engine.expand_resolution(memory, target_resolution)
        if not expanded:
            raise ValueError(f"No version available at {target_resolution.name}")

        # Convert to IR with expanded content
        ir = self.memory_ir_adapter.to_ir(memory)
        ir.semantic_content.content = expanded.content
        ir.resolution = target_resolution

        return ir

    async def trace(self, memory_id: int) -> MemoryInspection:
        """Full provenance trace for a memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        ir = self.memory_ir_adapter.to_ir(memory)
        provenance_chain = self.recall_engine.get_memory_provenance(memory)
        provenance_irs = [self.memory_ir_adapter.to_ir(m) for m in provenance_chain]

        full_prov = self.recall_engine.get_full_provenance(memory)

        return MemoryInspection(
            memory_id=memory_id,
            memory_ir=ir,
            provenance_chain=provenance_irs,
            source_conversation=full_prov.get("source_conversation"),
            source_messages=full_prov.get("source_messages", []),
            exact_source_message=full_prov.get("exact_source_message"),
        )

    async def timeline(self, topic: str, limit: int = 100) -> TimelineResult:
        """Get timeline for a topic."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        memories = self.store.get_memories(
            topic_id=topic_obj.id,
            memory_type=MemoryType.TIMELINE,
            status=MemoryStatus.ACTIVE
        )
        memories.sort(key=lambda m: m.valid_from or m.created_at)

        events = []
        for mem in memories[:limit]:
            date_str = (mem.valid_from or mem.created_at).strftime('%Y-%m-%d')
            events.append({
                "date": date_str,
                "content": mem.content,
                "memory_id": mem.id,
            })

        return TimelineResult(topic=topic, events=events)

    async def explain(self, query: str, topic: str) -> RecallExplanation:
        """Explain how recall would work for a query."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        explanation = self.human_recall_engine.get_recall_explanation(query, topic_obj.id)

        return RecallExplanation(
            query=explanation["query"],
            mode=explanation["mode"],
            resolution=explanation["resolution"],
            confidence=explanation["confidence"],
            memories_retrieved=explanation["memories_retrieved"],
            reasoning=explanation["reasoning"],
            adaptations=explanation["adaptations"],
        )

    def inspect(self, memory_id: int) -> MemoryInspection:
        """Debug inspection of a memory (sync version)."""
        import asyncio
        return asyncio.run(self.trace(memory_id))

    def _build_context_ir(self, query: str, topic_id: int, max_tokens: int) -> ContextIR:
        """Build ContextIR from context builder."""
        current_memories = self.store.get_memories(
            topic_id=topic_id,
            is_current=True,
            status=MemoryStatus.ACTIVE
        )
        self.context_builder.build_context(
            query, topic_id, max_tokens, current_memories
        )
        stats = self.context_builder.get_context_stats()

        budget = BudgetAllocation(
            high=stats.budget_allocation.high if stats.budget_allocation else 0,
            medium=stats.budget_allocation.medium if stats.budget_allocation else 0,
            low=stats.budget_allocation.low if stats.budget_allocation else 0,
        )

        return ContextIR(
            query=query,
            budget=budget,
            parts=[],
            stats=stats,
            system_prompt=self.context_builder.build_system_prompt("General"),
        )

    def close(self) -> None:
        """Clean up resources."""
        self.store.close()

    def __enter__(self) -> ArtificialMemoryRuntime:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== Phase 3: Memory Evolution ====================

    async def revise_memory(
        self,
        memory_id: int,
        new_evidence: str,
        evidence_source: str = "conversation",
        confidence: float = 0.8,
        supersede: bool = False,
    ) -> MemoryIR:
        """Revise a memory with new evidence."""
        memory = self.evolution_engine.revise_memory(
            memory_id, new_evidence, evidence_source, confidence, supersede
        )
        return self.memory_ir_adapter.to_ir(memory)

    async def merge_memories(
        self,
        memory_ids: list[int],
        strategy: str = "combine",
        new_type: MemoryType | None = None,
    ) -> MemoryIR:
        """Merge multiple memories into one."""
        merged = self.evolution_engine.merge_memories(memory_ids, strategy, new_type)
        return self.memory_ir_adapter.to_ir(merged)

    async def split_memory(
        self,
        memory_id: int,
        split_points: list[str],
    ) -> list[MemoryIR]:
        """Split a memory into multiple memories."""
        memories = self.evolution_engine.split_memory(memory_id, split_points)
        return [self.memory_ir_adapter.to_ir(m) for m in memories]

    async def reactivate_memory(self, memory_id: int) -> MemoryIR:
        """Reactivate an archived memory."""
        memory = self.evolution_engine.reactivate_memory(memory_id)
        return self.memory_ir_adapter.to_ir(memory)

    async def heal_memory(
        self,
        memory_id: int,
        issue_type: str,
        repair_action: str,
    ) -> MemoryIR:
        """Heal a memory from integrity issues."""
        memory = self.evolution_engine.heal_memory(memory_id, issue_type, repair_action)
        return self.memory_ir_adapter.to_ir(memory)

    # ==================== Phase 3: Belief Management ====================

    async def create_belief(
        self,
        proposition: str,
        supporting_memory_ids: list[int],
        confidence: float | None = None,
    ) -> Belief:
        """Create or update a belief from evidence."""
        return self.belief_engine.create_or_update_belief(proposition, supporting_memory_ids, confidence)

    async def get_belief(self, belief_id: int) -> Belief | None:
        return self.belief_engine.get_belief(belief_id)

    async def get_beliefs_for_topic(self, topic_id: int) -> list[Belief]:
        return self.belief_engine.get_beliefs_by_topic(topic_id)

    async def get_belief_at(self, timestamp: datetime) -> list[Belief]:
        """Time travel: get beliefs at a specific timestamp."""
        return self.belief_engine.get_belief_at(timestamp)

    async def supersede_belief(
        self,
        old_belief_id: int,
        new_proposition: str,
        supporting_memory_ids: list[int],
    ) -> Belief:
        """Explicitly supersede a belief."""
        return self.belief_engine.supersede_belief(old_belief_id, new_proposition, supporting_memory_ids)

    async def detect_contradictions(self, memory_ids: list[int]) -> list[BeliefConflict]:
        return self.belief_engine.detect_contradictions(memory_ids)

    async def resolve_conflict(self, conflict_id: int, resolution: str) -> bool:
        return self.belief_engine.resolve_conflict(conflict_id, resolution)

    async def get_belief_provenance(self, belief_id: int) -> dict[str, Any]:
        return self.belief_engine.get_belief_provenance(belief_id)

    # ==================== Phase 3: Contradiction Detection ====================

    async def find_contradictions(self, new_memory_id: int) -> list[ContradictionPair]:
        """Find all contradictions for a new memory."""
        new_mem = self.store.get_memory(new_memory_id)
        if not new_mem:
            raise ValueError(f"Memory {new_memory_id} not found")

        existing = self.store.get_memories(topic_id=new_mem.topic_id, limit=100)
        return self.contradiction_detector.detect_contradictions(new_mem, existing)

    async def resolve_contradiction(self, contradiction: ContradictionPair, resolution: str) -> None:
        self.contradiction_detector.resolve_contradiction(contradiction, resolution)

    # ==================== Phase 3: Dependency Graph ====================

    async def add_dependency(
        self,
        source_id: int,
        target_id: int,
        dep_type: DependencyType,
        strength: float = 0.5,
    ) -> Dependency:
        return self.dependency_graph.add_dependency(source_id, target_id, dep_type, strength)

    async def get_dependencies(self, memory_id: int) -> list[Dependency]:
        return self.dependency_graph.get_dependencies(memory_id)

    async def get_dependents(self, memory_id: int) -> list[Dependency]:
        return self.dependency_graph.get_dependents(memory_id)

    async def analyze_impact(self, memory_id: int) -> ImpactAnalysis:
        """Analyze what would be affected if this memory changed."""
        return self.dependency_graph.analyze_impact(memory_id)

    async def propagate_invalidation(self, memory_id: int, reason: str) -> list[int]:
        """Propagate invalidation from a memory to its dependents."""
        return self.dependency_graph.propagate_invalidation(memory_id, reason)

    async def check_consistency(self, memory_id: int) -> list[dict[str, Any]]:
        return self.dependency_graph.check_consistency(memory_id)

    # ==================== Phase 3: Counterfactual Analysis ====================

    async def evaluate_counterfactual(
        self,
        operation: CounterfactualOperation,
        target_memory_id: int,
        parameters: dict[str, Any] | None = None,
        query: str = "",
    ) -> CounterfactualResult:
        """Evaluate a counterfactual scenario."""
        scenario = CounterfactualScenario(
            id=f"cf_{operation.value}_{target_memory_id}_{datetime.now().timestamp()}",
            operation=operation,
            target_memory_id=target_memory_id,
            parameters=parameters or {},
            description=f"{operation.value} on memory {target_memory_id}",
        )
        return self.counterfactual_engine.evaluate_scenario(scenario)

    async def analyze_influence(
        self,
        query: str,
        topic_id: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Analyze memory influence scores for a query."""
        return self.counterfactual_engine.run_influence_analysis(query, topic_id, top_k)

    # ==================== Phase 3: Adaptive Recall ====================

    async def adaptive_recall(
        self,
        query: str,
        topic: str,
        max_tokens: int = 4000,
        weights: dict[str, float] | None = None,
    ) -> tuple[list[Memory], int, list[UtilityScore]]:
        """Utility-based adaptive recall."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        budget = RecallBudget(max_tokens=max_tokens)
        if weights:
            budget.weights = weights
        return self._adaptive_recall_engine.adaptive_recall(query, topic_obj.id, budget, max_tokens)

    async def explain_recall_selection(
        self,
        query: str,
        topic: str,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Explain why certain memories were selected for recall."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        budget = RecallBudget(max_tokens=max_tokens)
        return self._adaptive_recall_engine.explain_selection(query, topic_obj.id, budget)

    # ==================== Phase 4: Memory Integrity ====================

    async def check_integrity(self, memory_id: int) -> IntegrityReport:
        """Run full integrity check on a memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return self.integrity_metrics.compute_overall_integrity(memory)

    async def scan_integrity(self, topic_id: int, limit: int = 100) -> list[IntegrityReport]:
        """Scan all memories in a topic for integrity issues."""
        return self.integrity_metrics.scan_topic(topic_id, limit)

    async def validate_compression(
        self,
        memory_id: int,
        target_resolution: ResolutionLevel,
    ) -> CompressionValidationReport:
        """Validate compression quality for a specific resolution."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return self.compression_validator.validate_compression(memory, target_resolution)

    async def validate_all_versions(self, memory_id: int) -> list[CompressionValidationReport]:
        """Validate all compressed versions of a memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return self.compression_validator.validate_all_versions(memory)

    async def detect_stale_memories(
        self,
        topic_id: int,
        limit: int = 100,
    ) -> list[StalenessAssessment]:
        """Detect stale memories in a topic."""
        return self.stale_detector.scan_topic(topic_id, limit)

    async def get_stale_summary(self, topic_id: int) -> dict[str, Any]:
        """Get summary of stale memories in a topic."""
        return self.stale_detector.get_stale_summary(topic_id)

    # ==================== Phase 4: Memory Healing ====================

    async def auto_heal_memory(self, memory_id: int) -> list[HealingResult]:
        """Automatically diagnose and heal all issues for a memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        report = self.healer.diagnose(memory)
        if not report.has_critical and not report.has_warnings:
            return []

        plan = self.healer.create_healing_plan(report)
        return self.healer.execute_plan(plan)

    async def heal_topic(self, topic_id: int, max_issues: int = 50) -> list[HealingResult]:
        """Heal all issues in a topic."""
        return self.healer.heal_topic(topic_id, max_issues)

    # ==================== Phase 4: Memory Debugger ====================

    async def trace_recall(
        self,
        query: str,
        topic: str | None = None,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> RecallTrace:
        """Execute recall with full debugging trace."""
        topic_id = topic_id
        if topic and not topic_id:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.debugger.trace_recall(query, topic_id, level, max_tokens)

    async def explain_recall_trace(self, trace: RecallTrace) -> dict[str, Any]:
        """Generate human-readable explanation from recall trace."""
        return self.debugger.explain_recall(trace)

    async def trace_context_build(
        self,
        query: str,
        topic_id: int | None = None,
        max_tokens: int = 8000,
        current_memories: list[Memory] | None = None,
    ) -> ContextTrace:
        """Execute context building with full tracing."""
        return self.debugger.trace_context_build(query, topic_id, max_tokens, current_memories)

    async def explain_context_trace(self, trace: ContextTrace) -> dict[str, Any]:
        """Explain why context was built this way."""
        return self.debugger.explain_context(trace)

    async def trace_compilation(
        self,
        conversation_id: int,
    ) -> CompilationTrace:
        """Execute memory compilation with full tracing."""
        conv = self.store.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        messages = self.store.get_messages(conversation_id)
        return self.debugger.trace_compilation(conv, messages)

    async def explain_compilation_trace(self, trace: CompilationTrace) -> dict[str, Any]:
        """Explain compilation decisions."""
        return self.debugger.explain_compilation(trace)

    async def inspect_memory(self, memory_id: int) -> dict[str, Any]:
        """Complete inspection of a memory with all diagnostics."""
        return self.debugger.inspect_memory(memory_id)

    async def get_debug_history(
        self,
        trace_type: str = "all",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get recent debug traces."""
        return self.debugger.get_recent_traces(trace_type, limit)

    # ==================== Phase 4: Adaptive Recall with Explanation ====================

    async def adaptive_recall_with_explanation(
        self,
        query: str,
        topic: str,
        max_tokens: int = 4000,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Adaptive recall with full explanation of selection."""
        topic_obj = self.conversation_manager.get_or_create_topic(topic)
        budget = RecallBudget(max_tokens=max_tokens)
        if weights:
            budget.weights = weights

        selected, tokens, scored = self._adaptive_recall_engine.adaptive_recall(
            query, topic_obj.id, budget, max_tokens
        )

        explanation = self._adaptive_recall_engine.explain_selection(query, topic_obj.id, budget)

        return {
            "selected": [
                {"id": m.id, "type": m.memory_type.value, "content": m.content[:200]}
                for m in selected
            ],
            "tokens": tokens,
            "scored_count": len(scored),
            "explanation": explanation,
        }

    # ==================== Phase 5: Temporal & Time Travel ====================

    async def travel_to(
        self,
        query_type: TemporalQueryType,
        timestamp: datetime | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        topic: str | None = None,
        include_archived: bool = False,
    ) -> TimeTravelResult:
        """Execute a time travel query."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        query = TemporalQuery(
            query_type=query_type,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            topic_id=topic_id,
            include_archived=include_archived,
        )
        return self.time_travel_engine.travel_to(query)

    async def get_state_at(
        self,
        timestamp: datetime,
        topic: str | None = None,
        include_archived: bool = False,
    ) -> TemporalState:
        """Get memory state at a specific timestamp."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        query = TemporalQuery(
            query_type=TemporalQueryType.STATE_AT,
            timestamp=timestamp,
            topic_id=topic_id,
            include_archived=include_archived,
        )
        result = self.time_travel_engine.travel_to(query)
        return result.state

    async def get_beliefs_at(
        self,
        timestamp: datetime,
        topic: str | None = None,
    ) -> list[HistoricalBelief]:
        """Get beliefs as they existed at a specific timestamp (Time Travel)."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.belief_history_engine.get_beliefs_at(timestamp, topic_id)

    async def get_belief_evolution(
        self,
        belief_id: int,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> BeliefEvolution | None:
        """Get the full evolution history of a belief."""
        return self.belief_history_engine.get_belief_evolution(belief_id, start, end)

    async def get_belief_timeline(
        self,
        topic: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> BeliefTimeline:
        """Get timeline of all beliefs for a topic."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.belief_history_engine.get_belief_timeline(topic_id, start, end)

    async def compare_beliefs(
        self,
        timestamp_a: datetime,
        timestamp_b: datetime,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Compare beliefs at two different timestamps."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.belief_history_engine.compare_beliefs(timestamp_a, timestamp_b, topic_id)

    # ==================== Phase 5: Context Reconstruction ====================

    async def reconstruct_context_at(
        self,
        timestamp: datetime,
        query: str,
        topic: str | None = None,
        config: ContextReconstructionConfig | None = None,
    ) -> ReconstructedContext:
        """Reconstruct the context that would have been built at a timestamp."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.context_reconstructor.reconstruct_context_at(
            timestamp, query, topic_id, config
        )

    async def get_context_at_decision_point(
        self,
        decision_memory_id: int,
        query: str = "decision context",
    ) -> ReconstructedContext | None:
        """Get the context that existed when a decision was made."""
        return self.context_reconstructor.get_context_at_decision_point(
            decision_memory_id, query
        )

    async def compare_contexts(
        self,
        timestamp_a: datetime,
        timestamp_b: datetime,
        query: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Compare contexts at two different timestamps."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.context_reconstructor.compare_contexts(
            timestamp_a, timestamp_b, query, topic_id
        )

    # ==================== Phase 5: Decision Traces ====================

    async def trace_decision(self, decision_id: int) -> DecisionTraceReport:
        """Generate a complete trace report for a decision."""
        return self.decision_tracer.trace_decision(decision_id)

    async def get_all_decision_traces(self, topic: str | None = None) -> list[DecisionTraceReport]:
        """Get traces for all decisions in a topic."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.decision_tracer.get_all_decision_traces(topic_id)

    async def compare_decisions(
        self,
        decision_id_a: int,
        decision_id_b: int,
    ) -> dict[str, Any]:
        """Compare two decisions."""
        return self.decision_tracer.compare_decisions(decision_id_a, decision_id_b)

    # ==================== Phase 5: Impact Analysis ====================

    async def predict_impact(
        self,
        change_type: str,
        target_memory_id: int,
        parameters: dict[str, Any] | None = None,
    ) -> ImpactPrediction:
        """Predict the impact of a proposed memory change."""
        simulation = ChangeSimulation(
            change_type=change_type,
            target_memory_id=target_memory_id,
            proposed_changes=parameters or {},
        )
        return self.impact_analyzer.predict_impact(simulation)

    async def detect_circular_dependencies(self) -> list[DependencyCycle]:
        """Detect all circular dependencies in the memory graph."""
        return self.impact_analyzer.detect_circular_dependencies()

    async def find_single_points_of_failure(self) -> list[dict[str, Any]]:
        """Find memories that are single points of failure."""
        return self.impact_analyzer.find_single_points_of_failure()

    async def get_safe_modification_strategy(self, memory_id: int) -> dict[str, Any]:
        """Get a safe strategy for modifying a memory."""
        return self.impact_analyzer.get_safe_modification_strategy(memory_id)

    async def simulate_scenario(
        self,
        change_type: str,
        target_memory_id: int,
        parameters: dict[str, Any] | None = None,
    ) -> ImpactPrediction:
        """Run a simulation of a proposed change."""
        return self.impact_analyzer.simulate_scenario(change_type, target_memory_id, parameters or {})

    # ==================== Phase 6: Research Platform ====================

    # --- Benchmarks ---
    async def run_benchmark(
        self,
        case_name: str,
        topic: str | None = None,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a specific benchmark case."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.am_benchmark_suite.run_case(
            case_name, self.store, self.recall_engine,
            self.context_builder, self.compressor, topic_id or 1, seed
        )

    async def run_benchmark_suite(
        self,
        topic: str | None = None,
        categories: list[BenchmarkCategory] | None = None,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run a full benchmark suite."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.am_benchmark_suite.run_suite(
            self.store, self.recall_engine, self.context_builder,
            self.compressor, topic_id or 1, categories, seed
        )

    async def run_benchmarks_by_category(
        self,
        category: BenchmarkCategory,
        topic: str | None = None,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks in a specific category."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.am_benchmark_suite.run_by_category(
            self.store, self.recall_engine, self.context_builder,
            self.compressor, category, topic_id or 1, seed
        )

    async def get_benchmark_report(self, output_path: str | None = None) -> str:
        """Generate a benchmark report."""
        return self.am_benchmark_suite.generate_report(output_path)

    # --- Ablation Studies ---
    async def register_ablation(self, config: AblationConfig):
        """Register an ablation study."""
        self.ablation_framework.register_ablation(config)

    async def run_ablation(
        self,
        ablation_name: str,
        test_queries: list[str],
        topic: str | None = None,
    ) -> AblationExperimentResult:
        """Run an ablation study."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.ablation_framework.run_ablation(
            ablation_name, self.recall_engine, test_queries, topic_id
        )

    async def get_ablation_results(self, ablation_name: str) -> AblationExperimentResult | None:
        """Get results of an ablation study."""
        return self.ablation_framework.results.get(ablation_name)

    async def get_ablation_report(self, output_path: str | None = None) -> str:
        """Generate an ablation study report."""
        return self.ablation_framework.generate_report(output_path)

    # --- Red-Team Suite ---
    async def register_attack(self, config: AttackConfig):
        """Register a red-team attack."""
        self.redteam_suite.register_attack(config)

    async def run_attack(
        self,
        attack_name: str,
        test_queries: list[str],
        topic: str | None = None,
    ) -> AttackResult:
        """Execute a red-team attack."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.redteam_suite.run_attack(attack_name, test_queries, topic_id)

    async def run_redteam_suite(
        self,
        test_queries: list[str],
        topic: str | None = None,
        attack_names: list[str] | None = None,
    ) -> list[AttackResult]:
        """Run a suite of red-team attacks."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.redteam_suite.run_attack_suite(test_queries, topic_id, attack_names)

    async def run_full_redteam(
        self,
        test_queries: list[str],
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Run full red-team evaluation."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.redteam_suite.run_full_suite(test_queries, topic_id)

    async def get_redteam_report(self, output_path: str | None = None) -> str:
        """Generate and save a red-team report."""
        return self.redteam_suite.generate_report(output_path)

    # --- AM-Specific Benchmarks ---
    async def run_am_benchmark(
        self,
        case_name: str,
        topic: str | None = None,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a specific AM benchmark case."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.am_benchmark_suite.run_case(
            case_name, self.store, self.recall_engine,
            self.context_builder, self.compressor, topic_id or 1, seed
        )

    async def run_am_benchmark_suite(
        self,
        topic: str | None = None,
        categories: list[BenchmarkCategory] | None = None,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run the full AM benchmark suite."""
        topic_id = None
        if topic:
            topic_obj = self.conversation_manager.get_or_create_topic(topic)
            topic_id = topic_obj.id

        return self.am_benchmark_suite.run_suite(
            self.store, self.recall_engine, self.context_builder,
            self.compressor, topic_id or 1, categories, seed
        )

    async def get_am_benchmark_report(self, output_path: str | None = None) -> str:
        """Generate an AM benchmark report."""
        return self.am_benchmark_suite.generate_report(output_path)

    # --- Experiment Framework ---
    async def run_experiment(
        self,
        manifest: ExperimentManifest,
        runner_fn: Callable[[dict[str, Any]], dict[str, float]],
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an experiment with the given manifest."""
        return self.experiment_runner.run(manifest, runner_fn, seed, extra_params)

    async def load_experiment(self, run_id: str) -> dict[str, Any] | None:
        """Load an experiment run by ID."""
        return self.experiment_runner.load_run(run_id)

    async def list_experiments(self, status: str | None = None) -> list[dict[str, Any]]:
        """List experiment runs."""
        return self.experiment_runner.list_runs(status)

    async def get_experiment(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific experiment run."""
        return self.experiment_runner.get_run(run_id)

    async def compare_experiments(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two experiment runs."""
        return self.experiment_runner.compare_runs(run_id_a, run_id_b)


def create_runtime(config: RuntimeConfig | None = None) -> ArtificialMemoryRuntime:
    """Factory function to create runtime."""
    return ArtificialMemoryRuntime(config)
