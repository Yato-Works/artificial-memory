from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from artificial_memory.core.ir import MemoryIR
from artificial_memory.core.models import Memory


class FederationRole(StrEnum):
    """Role of a node in the federation."""
    SOURCE = "source"          # Provides memories
    TARGET = "target"          # Receives memories
    BROKER = "broker"          # Facilitates exchange
    COORDINATOR = "coordinator" # Manages federation topology


class TrustLevel(StrEnum):
    """Trust levels for memory exchange."""
    UNTRUSTED = "untrusted"      # No trust - verify everything
    LOW = "low"                  # Minimal trust - verify claims
    MEDIUM = "medium"            # Standard trust - spot check
    HIGH = "high"                # High trust - accept with provenance
    IMPLICIT = "implicit"        # Full trust - no verification needed


class ExchangeStatus(StrEnum):
    """Status of a memory exchange."""
    PENDING = "pending"
    VERIFYING = "verifying"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FederationConfig:
    """Configuration for a federation participant."""
    node_id: str
    node_name: str
    role: FederationRole = FederationRole.SOURCE
    trust_level: TrustLevel = TrustLevel.MEDIUM
    allowed_memory_types: list[str] = field(default_factory=lambda: ["semantic", "decision", "episode"])
    max_memories_per_exchange: int = 100
    required_provenance: bool = True
    require_encryption: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPackage:
    """A package of memories for exchange."""
    package_id: str
    source_node_id: str
    target_node_id: str
    memories: list[MemoryIR]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    trust_level: TrustLevel = TrustLevel.MEDIUM
    provenance_chain: list[str] = field(default_factory=list)  # Node IDs in chain
    encryption_key_id: str | None = None
    digital_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeRequest:
    """Request to exchange memories."""
    request_id: str
    source_node_id: str
    target_node_id: str
    memory_ids: list[int]
    topic_filter: str | None = None
    trust_level: TrustLevel = TrustLevel.MEDIUM
    purpose: str = ""
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: ExchangeStatus = ExchangeStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeResponse:
    """Response to an exchange request."""
    request_id: str
    responder_node_id: str
    status: ExchangeStatus
    approved_memory_ids: list[int] = field(default_factory=list)
    rejected_memory_ids: list[int] = field(default_factory=list)
    reason: str = ""
    responded_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederationPeer:
    """A peer in the federation."""
    node_id: str
    node_name: str
    role: FederationRole
    trust_level: TrustLevel
    endpoint: str  # API endpoint
    public_key: str  # For encryption/signature verification
    capabilities: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class FederatedMemoryEngine:
    """Manages federated memory exchange between agents/nodes.

    Handles:
    - Memory exchange with trust policies
    - Provenance tracking across federation
    - Trust-based verification
    - Encryption and digital signatures
    - Federation peer management
    """

    def __init__(
        self,
        store,
        config: FederationConfig,
        local_node_id: str,
        local_private_key: str,
        local_public_key: str,
    ):
        self.store = store
        self.config = config
        self.local_node_id = local_node_id
        self.local_private_key = local_private_key
        self.local_public_key = local_public_key

        self.peers: dict[str, FederationPeer] = {}
        self.pending_requests: dict[str, ExchangeRequest] = {}
        self.exchange_history: list[dict[str, Any]] = []
        self.memory_packages: dict[str, MemoryPackage] = {}

        # Local trust policies
        self.trust_policies: dict[str, TrustLevel] = {}  # node_id -> trust_level

    def register_peer(self, peer: FederationPeer) -> None:
        """Register a federation peer."""
        self.peers[peer.node_id] = peer

    def unregister_peer(self, node_id: str) -> bool:
        """Unregister a federation peer."""
        if node_id in self.peers:
            del self.peers[node_id]
            return True
        return False

    def set_trust_policy(self, node_id: str, trust_level: TrustLevel) -> None:
        """Set trust level for a specific node."""
        self.trust_policies[node_id] = trust_level

    def get_effective_trust_level(self, node_id: str) -> TrustLevel:
        """Get effective trust level for a node (policy overrides peer config)."""
        return self.trust_policies.get(node_id, self.peers.get(node_id, FederationPeer(node_id, "", FederationRole.SOURCE, TrustLevel.UNTRUSTED, "", "")).trust_level)

    def create_exchange_request(
        self,
        target_node_id: str,
        memory_ids: list[int],
        topic_filter: str | None = None,
        trust_level: TrustLevel = TrustLevel.MEDIUM,
        purpose: str = "",
        expires_in_hours: int = 24,
    ) -> ExchangeRequest:
        """Create an exchange request to another node."""
        request_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)

        request = ExchangeRequest(
            request_id=request_id,
            source_node_id=self.local_node_id,
            target_node_id=target_node_id,
            memory_ids=memory_ids,
            topic_filter=topic_filter,
            trust_level=trust_level,
            purpose=purpose,
            expires_at=expires_at,
        )

        self.pending_requests[request_id] = request
        return request

    def process_exchange_request(self, request: ExchangeRequest) -> ExchangeResponse:
        """Process an incoming exchange request."""
        # Check if we trust the source
        trust_level = self.get_effective_trust_level(request.source_node_id)

        if trust_level == TrustLevel.UNTRUSTED:
            return ExchangeResponse(
                request_id=request.request_id,
                responder_node_id=self.local_node_id,
                status=ExchangeStatus.REJECTED,
                reason="Source node is untrusted",
            )

        # Verify request hasn't expired
        if request.expires_at and request.expires_at < datetime.now():
            return ExchangeResponse(
                request_id=request.request_id,
                responder_node_id=self.local_node_id,
                status=ExchangeStatus.REJECTED,
                reason="Request has expired",
            )

        # Get requested memories
        memories = []
        for mem_id in request.memory_ids:
            mem = self.store.get_memory(mem_id)
            if mem:
                memories.append(mem)

        # Filter by topic if specified
        if request.topic_filter:
            memories = [m for m in memories if self._matches_topic(m, request.topic_filter)]

        # Filter by trust level - don't share high-trust memories with low-trust nodes
        filtered_memories = self._filter_by_trust(memories, trust_level)

        approved_ids = [m.id for m in filtered_memories]
        rejected_ids = [m.id for m in request.memory_ids if m.id not in approved_ids]

        status = ExchangeStatus.APPROVED if approved_ids else ExchangeStatus.REJECTED
        reason = "" if approved_ids else "No memories matched trust level or topic filter"

        response = ExchangeResponse(
            request_id=request.request_id,
            responder_node_id=self.local_node_id,
            status=status,
            approved_memory_ids=approved_ids,
            rejected_memory_ids=rejected_ids,
            reason=reason,
        )

        # Log exchange
        self.exchange_history.append({
            "request_id": request.request_id,
            "source": request.source_node_id,
            "target": self.local_node_id,
            "status": status.value,
            "memory_count": len(approved_ids),
            "timestamp": datetime.now().isoformat(),
        })

        return response

    def create_memory_package(
        self,
        target_node_id: str,
        memory_ids: list[int],
        trust_level: TrustLevel = TrustLevel.MEDIUM,
    ) -> MemoryPackage:
        """Create a memory package for exchange."""
        package_id = str(uuid.uuid4())

        memories = []
        for mem_id in memory_ids:
            mem = self.store.get_memory(mem_id)
            if mem:
                mem_ir = self._memory_to_ir(mem)
                memories.append(mem_ir)

        package = MemoryPackage(
            package_id=package_id,
            source_node_id=self.local_node_id,
            target_node_id=target_node_id,
            memories=memories,
            trust_level=trust_level,
        )

        self.memory_packages[package_id] = package
        return package

    def verify_package(self, package: MemoryPackage, sender_public_key: str) -> bool:
        """Verify a received memory package."""
        # Verify digital signature
        if package.digital_signature:
            # In production, verify using sender_public_key
            pass

        # Verify provenance chain
        if package.provenance_chain:
            # Check if we trust all nodes in chain
            for node_id in package.provenance_chain:
                trust = self.get_effective_trust_level(node_id)
                if trust == TrustLevel.UNTRUSTED:
                    return False

        # Check expiration
        if package.expires_at and package.expires_at < datetime.now():
            return False

        return True

    def import_package(self, package: MemoryPackage, sender_public_key: str) -> list[MemoryIR]:
        """Import a verified memory package."""
        if not self.verify_package(package, sender_public_key):
            raise ValueError("Package verification failed")

        imported = []
        for mem_ir in package.memories:
            # Convert to local memory
            legacy_mem, versions, associations, compressions = self._ir_to_memory(mem_ir)
            stored = self.store.create_memory(legacy_mem)

            # Store versions
            for v in versions:
                v.memory_id = stored.id
                self.store.add_memory_version(v)

            # Store associations
            for a in associations:
                a.source_memory_id = stored.id
                self.store.create_association(a)

            imported.append(mem_ir)

        return imported

    def get_exchange_history(
        self,
        node_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get exchange history."""
        history = self.exchange_history
        if node_id:
            history = [h for h in history if h.get("source") == node_id or h.get("target") == node_id]
        return history[-limit:]

    def _matches_topic(self, memory: Memory, topic_filter: str) -> bool:
        """Check if memory matches topic filter."""
        # In practice, would check topic hierarchy
        return True  # Simplified

    def _filter_by_trust(self, memories: list[Memory], trust_level: TrustLevel) -> list[Memory]:
        """Filter memories based on trust level."""
        if trust_level == TrustLevel.IMPLICIT:
            return memories

        filtered = []
        for mem in memories:
            # Don't share high-importance memories with low-trust nodes
            if trust_level == TrustLevel.LOW and mem.importance > 0.7:
                continue
            if trust_level == TrustLevel.LOW and mem.confidence < 0.8:
                continue
            filtered.append(mem)
        return filtered

    def _memory_to_ir(self, memory: Memory) -> MemoryIR:
        """Convert legacy Memory to MemoryIR."""
        import hashlib

        from artificial_memory.core.ir import (
            AccessRecord,
            ConfidenceProfile,
            LifecycleState,
            MemoryIdentity,
            MemoryIR,
            ProvenanceChain,
            ProvenanceLink,
            SemanticContent,
            TemporalScope,
        )

        return MemoryIR(
            identity=MemoryIdentity(
                memory_id=memory.id,
                version=1,
                content_hash=hashlib.sha256(memory.content.encode()).hexdigest()[:16],
            ),
            type=memory.memory_type,
            resolution=memory.resolution,
            semantic_content=SemanticContent(content=memory.content),
            source=ProvenanceChain(
                source=ProvenanceLink(
                    conversation_id=memory.source_conversation_id,
                    message_id=memory.source_message_id,
                    timestamp=memory.created_at,
                )
            ),
            temporal_scope=TemporalScope(
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            ),
            confidence=ConfidenceProfile(
                memory_confidence=memory.confidence,
                retrieval_confidence=memory.importance,
                temporal_confidence=1.0,
                source_confidence=0.9,
                overall=memory.confidence,
            ),
            importance=memory.importance,
            dependencies=[],
            relations=[],
            compression_history=[],
            access_history=AccessRecord(),
            lifecycle_state=LifecycleState.HOT,
        )

    def _ir_to_memory(self, ir) -> tuple:
        """Convert MemoryIR back to legacy Memory."""
        # Simplified - in practice would use the adapter
        return None, [], [], []


def create_federated_memory_engine(
    store,
    config: FederationConfig,
    local_node_id: str,
    local_private_key: str,
    local_public_key: str,
) -> FederatedMemoryEngine:
    return FederatedMemoryEngine(store, config, local_node_id, local_private_key, local_public_key)
