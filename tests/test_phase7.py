"""Tests for Phase 7: Multi-Agent & Enterprise features."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from artificial_memory.core.models import (
    Conversation,
    Memory,
    MemoryType,
    Project,
    ResolutionLevel,
    Topic,
)
from artificial_memory.governance.audit import (
    AuditEventType,
    AuditSeverity,
    create_audit_logger,
)
from artificial_memory.governance.engine import (
    create_governance_engine,
)
from artificial_memory.governance.retention import (
    create_retention_policy_engine,
)
from artificial_memory.governance.tenancy import (
    TenantConfig,
    create_tenancy_manager,
)
from artificial_memory.governance.trust import (
    TrustLevel,
    create_trust_policy_engine,
)
from artificial_memory.memory.exchange import (
    create_memory_exchange_protocol,
)
from artificial_memory.memory.federation import (
    ExchangeStatus,
    FederationConfig,
    FederationPeer,
    FederationRole,
    create_federated_memory_engine,
)
from artificial_memory.observability.metrics import (
    create_metrics_collector,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


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
    from artificial_memory.compression.compressor import RuleBasedCompressor
    return RuleBasedCompressor()


@pytest.fixture
def recall_engine(store):
    from artificial_memory.recall.engine import BasicRecallEngine
    return BasicRecallEngine(store)


@pytest.fixture
def context_builder(store, recall_engine):
    from artificial_memory.context.builder import EnhancedContextBuilder
    return EnhancedContextBuilder(store, recall_engine)


@pytest.fixture
def sample_data(store):
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

    mem1 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.SEMANTIC,
        content="PostgreSQL was chosen as the database for horizontal scaling",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        source_conversation_id=conv.id,
    )
    mem1 = store.create_memory(mem1)

    return {
        "project": project,
        "topic": topic,
        "conversation": conv,
        "memories": [mem1],
    }


class TestFederatedMemoryEngine:
    """Test Federated Memory Engine."""

    def test_create_federated_engine(self, store, compressor, sample_data):
        config = FederationConfig(
            node_id="node-1",
            node_name="Test Node",
            role="source",
        )

        engine = create_federated_memory_engine(
            store, config, "node-1", "private_key", "public_key"
        )

        assert engine is not None
        assert engine.config.node_id == "node-1"

    def test_register_peer(self, store, compressor, sample_data):
        config = FederationConfig(node_id="node-1", node_name="Test Node")
        engine = create_federated_memory_engine(store, config, "node-1", "key1", "key2")

        peer = FederationPeer(
            node_id="node-2",
            node_name="Remote Node",
            role=FederationRole.TARGET,
            trust_level=TrustLevel.HIGH,
            endpoint="http://node-2:8080",
            public_key="pubkey2",
        )
        engine.register_peer(peer)

        assert "node-2" in engine.peers
        assert engine.peers["node-2"].node_name == "Remote Node"

    def test_exchange_request_response(self, store, compressor, sample_data):
        config = FederationConfig(node_id="node-1", node_name="Test Node")
        engine = create_federated_memory_engine(store, config, "node-1", "key1", "key2")

        mem = sample_data["memories"][0]

        request = engine.create_exchange_request(
            target_node_id="node-2",
            memory_ids=[mem.id],
            purpose="sharing decision",
        )

        assert request.request_id is not None
        assert request.source_node_id == "node-1"
        assert request.target_node_id == "node-2"
        assert request.memory_ids == [mem.id]

        response = engine.process_exchange_request(request)

        assert response.request_id == request.request_id
        assert response.responder_node_id == "node-1"
        assert response.status in [ExchangeStatus.APPROVED, ExchangeStatus.REJECTED]


class TestMemoryExchangeProtocol:
    """Test Memory Exchange Protocol."""

    def test_create_protocol(self, store, compressor, sample_data):
        config = FederationConfig(node_id="node-1", node_name="Test Node")
        engine = create_federated_memory_engine(store, config, "node-1", "key1", "key2")

        protocol = create_memory_exchange_protocol(engine)

        assert protocol is not None
        assert protocol.federated_engine == engine


class TestTrustPolicyEngine:
    """Test Trust Policy Engine."""

    def test_create_trust_engine(self, store, sample_data):
        engine = create_trust_policy_engine(store)

        assert engine is not None
        assert len(engine.policies) >= 2  # default policies

    def test_set_node_trust_level(self, store, sample_data):
        engine = create_trust_policy_engine(store)

        engine.set_node_trust_level("node-1", "high")
        assert engine.get_node_trust_level("node-1") == "high"

        # Default trust level
        assert engine.get_node_trust_level("unknown-node") == "medium"

    def test_evaluate_trust(self, store, sample_data):
        engine = create_trust_policy_engine(store)
        engine.set_node_trust_level("node-1", "high")

        result = engine.evaluate_trust("node-1", [])

        assert result.allowed is True
        assert result.trust_level == "high"


class TestTenancyManager:
    """Test Tenancy Manager."""

    def test_create_tenant(self, store):
        manager = create_tenancy_manager(store)

        tenant = manager.create_tenant(
            tenant_id="tenant-1",
            name="Test Tenant",
            display_name="Test Tenant Display",
        )

        assert tenant.tenant_id == "tenant-1"
        assert tenant.name == "Test Tenant"
        assert tenant.is_active is True

    def test_tenant_limits(self, store):
        manager = create_tenancy_manager(store)
        tenant = manager.create_tenant("tenant-1", "Test", "Test", TenantConfig(
            tenant_id="tenant-1",
            name="Test",
            display_name="Test",
            max_memories=10,
        ))

        can_create, reason = tenant.can_create_memory("semantic", 1)
        assert can_create is True

        # Fill up to limit
        for i in range(10):
            assert tenant.can_create_memory("semantic", 1)[0] is True
            tenant.increment_memory_count(1)

        # Should fail at limit
        can_create, reason = tenant.can_create_memory("semantic", 1)
        assert can_create is False


class TestGovernanceEngine:
    """Test Governance Engine."""

    def test_create_governance_engine(self, store):
        engine = create_governance_engine(store)

        assert engine is not None
        assert len(engine.rules) >= 8  # default rules

    def test_evaluate_policy(self, store, sample_data):
        engine = create_governance_engine(store)
        mem = sample_data["memories"][0]

        violations = engine.evaluate({
            "memory_id": mem.id,
            "topic_id": mem.topic_id,
            "content": mem.content,
            "confidence": mem.confidence,
            "tokens": 100,
        })

        # Should not have violations for valid memory
        assert isinstance(violations, list)

    def test_policy_violation_detection(self, store):
        engine = create_governance_engine(store)

        # Test PII detection
        violations = engine.evaluate({
            "content": "My SSN is 123-45-6789",
            "memory_id": 1,
        })

        assert len(violations) > 0
        assert any(v.rule_id == "pii_detection" for v in violations)


class TestAuditLogger:
    """Test Audit Logger."""

    def test_log_event(self, store, sample_data):
        logger = create_audit_logger(store)

        event_id = logger.log_event(
            event_type=AuditEventType.MEMORY_CREATED,
            severity=AuditSeverity.INFO,
            user_id="user-1",
            tenant_id="tenant-1",
            action="create_memory",
            description="Created new memory",
        )

        assert event_id is not None
        assert len(event_id) > 0

    def test_query_events(self, store):
        logger = create_audit_logger(store)

        logger.log_event(
            event_type=AuditEventType.MEMORY_CREATED,
            severity=AuditSeverity.INFO,
            user_id="user-1",
        )

        events = logger.query_events(event_types=[AuditEventType.MEMORY_CREATED], limit=10)

        assert len(events) >= 1


class TestRetentionPolicyEngine:
    """Test Retention Policy Engine."""

    def test_create_retention_engine(self, store, compressor, sample_data):
        from artificial_memory.memory.consolidation import ConsolidationConfig, ConsolidationEngine

        consolidation = ConsolidationEngine(store, compressor, ConsolidationConfig())

        engine = create_retention_policy_engine(store, compressor, consolidation)

        assert engine is not None
        assert len(engine.policies) >= 4  # default policies

    def test_evaluate_memory(self, store, compressor, sample_data):
        from artificial_memory.memory.consolidation import ConsolidationConfig, ConsolidationEngine

        consolidation = ConsolidationEngine(store, compressor, ConsolidationConfig())

        engine = create_retention_policy_engine(store, compressor, consolidation)
        mem = sample_data["memories"][0]

        rules = engine.evaluate_memory(mem)

        assert isinstance(rules, list)


class TestMetricsCollector:
    """Test Metrics Collector."""

    def test_create_collector(self):
        collector = create_metrics_collector()

        assert collector is not None

    def test_record_metrics(self):
        collector = create_metrics_collector()

        collector.increment_counter("test_counter", 5, labels={"env": "test"})
        collector.set_gauge("test_gauge", 42, labels={"env": "test"})
        collector.observe_histogram("test_histogram", 100, labels={"type": "test"})

        assert collector.get_counter("test_counter", labels={"env": "test"}) == 5
        assert collector.get_gauge("test_gauge", labels={"env": "test"}) == 42

        stats = collector.get_histogram_stats("test_histogram", labels={"type": "test"})
        assert stats["count"] == 1
        assert stats["mean"] == 100

    def test_prometheus_export(self):
        collector = create_metrics_collector()

        collector.increment_counter("test_metric", 10)

        prom = collector.export_prometheus()
        assert "test_metric" in prom
        assert "counter" in prom


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
