"""Tests for Phase 4: Memory Integrity, Stale Detection, Validation, Healing, Debugger."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.context.builder import EnhancedContextBuilder
from artificial_memory.core.models import (
    Conversation,
    Memory,
    MemoryType,
    Project,
    RecallLevel,
    ResolutionLevel,
    Topic,
)
from artificial_memory.memory.compiler import CompilerPipelineWrapper
from artificial_memory.memory.debugger import MemoryDebugger
from artificial_memory.memory.healing import (
    HealingAction,
    HealingActionType,
    HealingResult,
    MemoryHealer,
)
from artificial_memory.memory.integrity import (
    IntegrityMetrics,
    IntegrityReport,
)
from artificial_memory.memory.stale import StaleMemoryDetector, StalenessAssessment, StalenessReason
from artificial_memory.memory.validation import (
    CompressionValidationReport,
    CompressionValidator,
    ValidationStatus,
)
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore
from artificial_memory.topic.classifier import RuleBasedTopicClassifier


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    import gc
    import time
    gc.collect()
    time.sleep(0.1)
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass


@pytest.fixture
def store(temp_db):
    s = SQLiteMemoryStore(temp_db)
    yield s
    s.close()


@pytest.fixture
def compressor():
    return RuleBasedCompressor()


@pytest.fixture
def recall_engine(store):
    return BasicRecallEngine(store)


@pytest.fixture
def context_builder(store, recall_engine):
    return EnhancedContextBuilder(store, recall_engine)


@pytest.fixture
def pipeline_wrapper(store, compressor):
    classifier = RuleBasedTopicClassifier(store)
    return CompilerPipelineWrapper(store, compressor, classifier)


@pytest.fixture
def sample_data(store):
    """Create sample data for testing."""
    project = Project(name="test-project")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test-project/Test")
    topic = store.create_topic(topic)

    conv = Conversation(
        topic_id=topic.id,
        title="Test Conversation",
        started_at=datetime.now(),
    )
    conv = store.create_conversation(conv)

    # Create memories of different types and resolutions
    mem1 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.DECISION,
        content="Decision: We adopted PostgreSQL for horizontal scaling. Reason: Multi-region support.",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        source_conversation_id=conv.id,
    )
    mem1 = store.create_memory(mem1)

    mem2 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.SEMANTIC,
        content="PostgreSQL provides horizontal scaling and ACID compliance for the project.",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.8,
        confidence=0.9,
        source_conversation_id=conv.id,
    )
    mem2 = store.create_memory(mem2)

    mem3 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.EPISODE,
        content="Team discussed database options for 30 minutes. Decided on PostgreSQL.",
        resolution=ResolutionLevel.EPISODE,
        importance=0.7,
        confidence=0.8,
        source_conversation_id=conv.id,
    )
    mem3 = store.create_memory(mem3)

    # Add compressed versions
    from artificial_memory.core.models import MemoryVersion
    for mem in [mem1, mem2, mem3]:
        for res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE, ResolutionLevel.LONG_TERM]:
            if res.value > mem.resolution.value:
                ver = MemoryVersion(
                    memory_id=mem.id,
                    resolution=res,
                    content=f"Compressed {res.name}: {mem.content[:50]}",
                    compression_ratio=2.0,
                    created_at=datetime.now(),
                    source='test',
                )
                store.add_memory_version(ver)

    return {
        "project": project,
        "topic": topic,
        "conversation": conv,
        "memories": [mem1, mem2, mem3],
    }


class TestIntegrityMetrics:
    """Test Integrity Metrics."""

    def test_semantic_preservation(self, store, compressor, sample_data):
        """Test semantic preservation score."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        score = metrics.semantic_preservation_score(mem)

        assert 0 <= score <= 1

    def test_temporal_consistency(self, store, compressor, sample_data):
        """Test temporal consistency score."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        score = metrics.temporal_consistency_score(mem)

        assert 0 <= score <= 1

    def test_provenance_integrity(self, store, compressor, sample_data):
        """Test provenance integrity score."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        score = metrics.provenance_integrity_score(mem)

        assert 0 <= score <= 1

    def test_compression_quality(self, store, compressor, sample_data):
        """Test compression quality score."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        score = metrics.compression_quality_score(mem)

        assert 0 <= score <= 1

    def test_contradiction_consistency(self, store, compressor, sample_data):
        """Test contradiction consistency score."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        score = metrics.contradiction_consistency_score(mem)

        assert 0 <= score <= 1

    def test_overall_integrity_report(self, store, compressor, sample_data):
        """Test full integrity report generation."""
        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        report = metrics.compute_overall_integrity(mem)

        assert isinstance(report, IntegrityReport)
        assert report.memory_id == mem.id
        assert 0 <= report.overall_score <= 1
        assert isinstance(report.issues, list)
        assert isinstance(report.metrics, dict)

    def test_scan_topic(self, store, compressor, sample_data):
        """Test scanning a topic for integrity issues."""
        metrics = IntegrityMetrics(store, compressor)
        topic = sample_data["topic"]

        reports = metrics.scan_topic(topic.id)

        assert len(reports) >= 3
        for r in reports:
            assert isinstance(r, IntegrityReport)


class TestStaleMemoryDetector:
    """Test Stale Memory Detector."""

    def test_assess_fresh_memory(self, store, recall_engine, sample_data):
        """Test assessment of a fresh memory."""
        detector = StaleMemoryDetector(store, recall_engine)
        mem = sample_data["memories"][0]

        assessment = detector.assess_memory(mem)

        assert isinstance(assessment, StalenessAssessment)
        assert 0 <= assessment.staleness_score <= 1
        assert isinstance(assessment.is_stale, bool)
        assert assessment.recommended_action in ["keep", "monitor", "review_and_update", "archive_or_delete"]

    def test_stale_due_to_time(self, store, recall_engine, sample_data):
        """Test staleness detection for time-expired memory."""
        detector = StaleMemoryDetector(store, recall_engine, {
            "staleness_threshold": 0.5,
        })
        mem = sample_data["memories"][0]

        # Set valid_until to past
        mem.valid_until = datetime.now() - timedelta(days=10)
        store.update_memory(mem)

        assessment = detector.assess_memory(mem)

        assert assessment.is_stale
        assert any(s.reason == StalenessReason.TIME_EXPIRED for s in assessment.signals)

    def test_stale_due_to_inactivity(self, store, recall_engine, sample_data):
        """Test staleness detection for inactive memory."""
        detector = StaleMemoryDetector(store, recall_engine, {
            "max_inactive_days": 1,
            "staleness_threshold": 0.5,
        })
        mem = sample_data["memories"][0]

        # Set last_accessed to old
        mem.last_accessed = datetime.now() - timedelta(days=30)
        store.update_memory(mem)

        assessment = detector.assess_memory(mem)

        assert any(s.reason == StalenessReason.NO_RECENT_ACCESS for s in assessment.signals)

    def test_scan_topic(self, store, recall_engine, sample_data):
        """Test scanning a topic for stale memories."""
        detector = StaleMemoryDetector(store, recall_engine)
        topic = sample_data["topic"]

        assessments = detector.scan_topic(topic.id, include_keep=True)

        assert len(assessments) >= 3

    def test_get_stale_summary(self, store, recall_engine, sample_data):
        """Test stale summary generation."""
        detector = StaleMemoryDetector(store, recall_engine)
        topic = sample_data["topic"]

        summary = detector.get_stale_summary(topic.id)

        assert "topic_id" in summary
        assert "total_memories" in summary
        assert "stale_count" in summary
        assert "stale_percentage" in summary


class TestCompressionValidator:
    """Test Compression Validator."""

    def test_validate_compression(self, store, compressor, recall_engine, context_builder, sample_data):
        """Test validation of a single compression."""
        validator = CompressionValidator(store, compressor)
        mem = sample_data["memories"][0]

        report = validator.validate_compression(mem, ResolutionLevel.LIGHT)

        assert isinstance(report, CompressionValidationReport)
        assert report.memory_id == mem.id
        assert report.original_resolution == mem.resolution
        assert report.target_resolution == ResolutionLevel.LIGHT
        assert report.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]

    def test_validate_all_versions(self, store, compressor, recall_engine, context_builder, sample_data):
        """Test validation of all compressed versions."""
        validator = CompressionValidator(store, compressor)
        mem = sample_data["memories"][0]

        reports = validator.validate_all_versions(mem)

        assert len(reports) >= 1
        for r in reports:
            assert isinstance(r, CompressionValidationReport)

    def test_rule_min_compression_ratio(self, store, compressor):
        """Test minimum compression ratio rule."""
        validator = CompressionValidator(store, compressor)

        result = validator._check_min_compression_ratio(1.5)

        assert result.status == ValidationStatus.PASSED
        assert result.rule == "min_compression_ratio"

    def test_rule_content_not_empty(self, store, compressor):
        """Test content not empty rule."""
        validator = CompressionValidator(store, compressor)

        result_pass = validator._check_content_not_empty("Some content")
        result_fail = validator._check_content_not_empty("")

        assert result_pass.status == ValidationStatus.PASSED
        assert result_fail.status == ValidationStatus.FAILED


class TestMemoryHealer:
    """Test Memory Healer."""

    def test_heal_semantic_drift(self, store, compressor, pipeline_wrapper, sample_data):
        """Test healing semantic drift via recompilation."""
        from artificial_memory.memory.belief import BeliefEngine
        from artificial_memory.memory.contradiction import ContradictionDetector
        from artificial_memory.memory.evolution import MemoryEvolutionEngine

        belief_engine = BeliefEngine(store)
        evolution_engine = MemoryEvolutionEngine(store, compressor)
        contradiction_detector = ContradictionDetector(store)

        healer = MemoryHealer(store, compressor, pipeline_wrapper.pipeline,
                             belief_engine, evolution_engine, contradiction_detector)

        mem = sample_data["memories"][0]

        action = HealingAction(
            action_type=HealingActionType.RECOMPILE,
            target_memory_id=mem.id,
            parameters={"reason": "semantic_drift"},
            priority=10,
            reason="Test semantic drift healing",
        )

        result = healer.execute_action(action)

        assert isinstance(result, HealingResult)
        assert result.action.action_type == HealingActionType.RECOMPILE

    def test_heal_recompress(self, store, compressor, pipeline_wrapper, sample_data):
        """Test healing via recompression."""
        from artificial_memory.memory.belief import BeliefEngine
        from artificial_memory.memory.contradiction import ContradictionDetector
        from artificial_memory.memory.evolution import MemoryEvolutionEngine

        belief_engine = BeliefEngine(store)
        evolution_engine = MemoryEvolutionEngine(store, compressor)
        contradiction_detector = ContradictionDetector(store)

        healer = MemoryHealer(store, compressor, pipeline_wrapper.pipeline,
                             belief_engine, evolution_engine, contradiction_detector)

        mem = sample_data["memories"][0]

        action = HealingAction(
            action_type=HealingActionType.RECOMPRESS,
            target_memory_id=mem.id,
            parameters={"reason": "compression_artifact"},
            priority=8,
            reason="Test recompression",
        )

        result = healer.execute_action(action)

        assert isinstance(result, HealingResult)
        assert result.action.action_type == HealingActionType.RECOMPRESS

    def test_heal_restore_version(self, store, compressor, pipeline_wrapper, sample_data):
        """Test healing via version restoration."""
        from artificial_memory.memory.belief import BeliefEngine
        from artificial_memory.memory.contradiction import ContradictionDetector
        from artificial_memory.memory.evolution import MemoryEvolutionEngine

        belief_engine = BeliefEngine(store)
        evolution_engine = MemoryEvolutionEngine(store, compressor)
        contradiction_detector = ContradictionDetector(store)

        healer = MemoryHealer(store, compressor, pipeline_wrapper.pipeline,
                             belief_engine, evolution_engine, contradiction_detector)

        mem = sample_data["memories"][0]

        action = HealingAction(
            action_type=HealingActionType.RESTORE_VERSION,
            target_memory_id=mem.id,
            parameters={"target_resolution": "SEMANTIC"},
            priority=6,
            reason="Test version restore",
        )

        result = healer.execute_action(action)

        assert isinstance(result, HealingResult)
        assert result.action.action_type == HealingActionType.RESTORE_VERSION

    def test_create_healing_plan(self, store, compressor, pipeline_wrapper, sample_data):
        """Test creating a healing plan from integrity report."""
        from artificial_memory.memory.belief import BeliefEngine
        from artificial_memory.memory.contradiction import ContradictionDetector
        from artificial_memory.memory.evolution import MemoryEvolutionEngine
        from artificial_memory.memory.integrity import IntegrityMetrics

        belief_engine = BeliefEngine(store)
        evolution_engine = MemoryEvolutionEngine(store, compressor)
        contradiction_detector = ContradictionDetector(store)

        healer = MemoryHealer(store, compressor, pipeline_wrapper.pipeline,
                             belief_engine, evolution_engine, contradiction_detector)

        metrics = IntegrityMetrics(store, compressor)
        mem = sample_data["memories"][0]

        report = metrics.compute_overall_integrity(mem)
        plan = healer.create_healing_plan(report)

        assert isinstance(plan, type(plan))
        assert plan.memory_id == mem.id
        assert isinstance(plan.actions, list)


class TestMemoryDebugger:
    """Test Memory Debugger."""

    def test_trace_recall(self, store, recall_engine, context_builder, sample_data):
        """Test tracing a recall operation."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        topic = sample_data["topic"]

        trace = debugger.trace_recall("database choice", topic.id, RecallLevel.LONG_TERM_SUMMARY, 4000)

        assert isinstance(trace, type(trace))
        assert trace.query == "database choice"
        assert trace.recall_level == RecallLevel.LONG_TERM_SUMMARY
        assert trace.topic_id == topic.id
        assert trace.end_time is not None
        assert trace.duration_ms >= 0

    def test_explain_recall(self, store, recall_engine, context_builder, sample_data):
        """Test explaining a recall trace."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        topic = sample_data["topic"]

        trace = debugger.trace_recall("database choice", topic.id, RecallLevel.LONG_TERM_SUMMARY, 4000)
        explanation = debugger.explain_recall(trace)

        assert explanation["trace_id"] == trace.trace_id
        assert explanation["query"] == "database choice"
        assert "why_selected" in explanation

    def test_trace_context_build(self, store, recall_engine, context_builder, sample_data):
        """Test tracing context build."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        topic = sample_data["topic"]

        trace = debugger.trace_context_build("database", topic.id, 4000)

        assert isinstance(trace, type(trace))
        assert trace.query == "database"
        assert trace.topic_id == topic.id
        assert trace.end_time is not None

    def test_explain_context(self, store, recall_engine, context_builder, sample_data):
        """Test explaining context trace."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        topic = sample_data["topic"]

        trace = debugger.trace_context_build("database", topic.id, 4000)
        explanation = debugger.explain_context(trace)

        assert "trace_id" in explanation
        assert "final_tokens" in explanation
        assert "selected_parts" in explanation

    def test_inspect_memory(self, store, recall_engine, context_builder, sample_data):
        """Test full memory inspection."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        mem = sample_data["memories"][0]

        inspection = debugger.inspect_memory(mem.id)

        assert "memory" in inspection
        assert inspection["memory"]["id"] == mem.id
        assert "provenance" in inspection
        assert "versions" in inspection
        assert "associations" in inspection
        assert "compression_history" in inspection

    def test_get_debug_history(self, store, recall_engine, context_builder, sample_data):
        """Test getting debug history."""
        debugger = MemoryDebugger(store, recall_engine, context_builder)
        topic = sample_data["topic"]

        # Generate some traces
        debugger.trace_recall("test query", topic.id, RecallLevel.LONG_TERM_SUMMARY)
        debugger.trace_context_build("test context", topic.id, 4000)

        history = debugger.get_recent_traces("all", limit=5)

        assert "recall" in history
        assert "context" in history
        assert "compilation" in history


class TestRuntimeFacadePhase4:
    """Test Phase 4 features through Runtime Facade."""

    @pytest.mark.asyncio
    async def test_check_integrity(self, temp_db):
        """Test integrity check through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        # Create memory
        mem_ir = await runtime.remember(
            "PostgreSQL chosen for database with horizontal scaling",
            "Projects/Test/DB",
        )

        # Check integrity
        report = await runtime.check_integrity(mem_ir.identity.memory_id)

        assert isinstance(report, IntegrityReport)
        assert report.memory_id == mem_ir.identity.memory_id
        runtime.close()

    @pytest.mark.asyncio
    async def test_validate_compression(self, temp_db):
        """Test compression validation through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        mem_ir = await runtime.remember(
            "PostgreSQL chosen for database",
            "Projects/Test/DB",
        )

        report = await runtime.validate_compression(
            mem_ir.identity.memory_id,
            ResolutionLevel.LIGHT
        )

        assert isinstance(report, CompressionValidationReport)
        assert report.memory_id == mem_ir.identity.memory_id
        runtime.close()

    @pytest.mark.asyncio
    async def test_detect_stale_memories(self, temp_db):
        """Test stale memory detection through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        # Create old memory
        mem_ir = await runtime.remember(
            "Old information from last year",
            "Projects/Test/Stale",
        )

        # Manually set to stale
        mem = runtime.store.get_memory(mem_ir.identity.memory_id)
        mem.valid_until = datetime.now() - timedelta(days=30)
        mem.last_accessed = datetime.now() - timedelta(days=60)
        runtime.store.update_memory(mem)

        assessments = await runtime.detect_stale_memories(
            runtime.store.get_topic_by_path("Projects/Test/Stale").id
        )

        assert len(assessments) >= 1
        assert any(a.is_stale for a in assessments)
        runtime.close()

    @pytest.mark.asyncio
    async def test_auto_heal_memory(self, temp_db):
        """Test auto-healing through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        mem_ir = await runtime.remember(
            "Test memory for healing",
            "Projects/Test/Heal",
        )

        results = await runtime.auto_heal_memory(mem_ir.identity.memory_id)

        assert isinstance(results, list)
        runtime.close()

    @pytest.mark.asyncio
    async def test_trace_recall(self, temp_db):
        """Test recall tracing through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        await runtime.remember(
            "Test content for tracing",
            "Projects/Test/Trace",
        )

        trace = await runtime.trace_recall("test query", "Projects/Test/Trace")

        assert hasattr(trace, 'trace_id')
        assert trace.query == "test query"
        runtime.close()

    @pytest.mark.asyncio
    async def test_inspect_memory(self, temp_db):
        """Test memory inspection through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        mem_ir = await runtime.remember(
            "Test content for inspection",
            "Projects/Test/Inspect",
        )

        inspection = await runtime.inspect_memory(mem_ir.identity.memory_id)

        assert "memory" in inspection
        assert inspection["memory"]["id"] == mem_ir.identity.memory_id
        runtime.close()

    @pytest.mark.asyncio
    async def test_adaptive_recall_with_explanation(self, temp_db):
        """Test adaptive recall with explanation through facade."""
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        await runtime.remember(
            "PostgreSQL chosen for horizontal scaling",
            "Projects/Test/Adaptive",
        )

        result = await runtime.adaptive_recall_with_explanation(
            "database scaling", "Projects/Test/Adaptive"
        )

        assert "selected" in result
        assert "explanation" in result
        assert "tokens" in result
        runtime.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
