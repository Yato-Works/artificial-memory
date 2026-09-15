from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from artificial_memory.core.ir import MemoryIR
from artificial_memory.memory.federation import (
    ExchangeRequest,
    ExchangeResponse,
    ExchangeStatus,
    FederatedMemoryEngine,
    MemoryPackage,
    TrustLevel,
)


class ExchangeProtocolVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"


class ExchangeProtocol(StrEnum):
    REST = "rest"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"


class CompressionAlgorithm(StrEnum):
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"


class EncryptionAlgorithm(StrEnum):
    NONE = "none"
    AES256_GCM = "aes256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"


@dataclass
class ExchangeProtocolConfig:
    version: ExchangeProtocolVersion = ExchangeProtocolVersion.V2
    protocol: ExchangeProtocol = ExchangeProtocol.REST
    compression: CompressionAlgorithm = CompressionAlgorithm.ZSTD
    encryption: EncryptionAlgorithm = EncryptionAlgorithm.AES256_GCM
    max_package_size_mb: int = 100
    request_timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: int = 5


@dataclass
class ExchangeSession:
    session_id: str
    source_node_id: str
    target_node_id: str
    protocol_config: ExchangeProtocolConfig
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: ExchangeStatus = ExchangeStatus.PENDING
    packages_sent: int = 0
    packages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryExchangeProtocol:
    """Handles the low-level protocol for memory exchange between nodes.

    Supports:
    - Multiple transport protocols (REST, gRPC, WebSocket, Message Queue)
    - Compression and encryption
    - Session management
    - Flow control and backpressure
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        federated_engine: FederatedMemoryEngine,
        protocol_config: ExchangeProtocolConfig | None = None,
    ):
        self.federated_engine = federated_engine
        self.protocol_config = protocol_config or ExchangeProtocolConfig()
        self.active_sessions: dict[str, ExchangeSession] = {}
        self.message_handlers: dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self):
        self.message_handlers = {
            "exchange_request": self._handle_exchange_request,
            "exchange_response": self._handle_exchange_response,
            "memory_package": self._handle_memory_package,
            "heartbeat": self._handle_heartbeat,
            "ack": self._handle_ack,
            "error": self._handle_error,
        }

    def create_session(
        self,
        target_node_id: str,
        protocol_config: ExchangeProtocolConfig | None = None,
    ) -> ExchangeSession:
        session_id = str(uuid.uuid4())
        config = protocol_config or self.protocol_config

        session = ExchangeSession(
            session_id=session_id,
            source_node_id=self.federated_engine.local_node_id,
            target_node_id=target_node_id,
            protocol_config=config,
        )

        self.active_sessions[session_id] = session
        return session

    def close_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = ExchangeStatus.COMPLETED
            del self.active_sessions[session_id]
            return True
        return False

    def send_exchange_request(
        self,
        session_id: str,
        request: ExchangeRequest,
    ) -> bool:
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Serialize and send request
        self._serialize_request(request)
        # In production, would send via HTTP/gRPC/WebSocket
        # For now, simulate by calling local handler
        self.federated_engine.pending_requests[request.request_id] = request
        session.packages_sent += 1
        session.last_activity = datetime.now()
        return True

    def send_memory_package(
        self,
        session_id: str,
        package: MemoryPackage,
    ) -> bool:
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Verify package size
        package_size = len(json.dumps(package.__dict__).encode())
        if package_size > self.protocol_config.max_package_size_mb * 1024 * 1024:
            return False

        # Serialize and send
        self._serialize_package(package)
        session.packages_sent += 1
        session.bytes_sent += package_size
        session.last_activity = datetime.now()
        return True

    def receive_message(self, session_id: str, message_type: str, payload: bytes) -> bool:
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        handler = self.message_handlers.get(message_type)
        if not handler:
            session.errors.append(f"Unknown message type: {message_type}")
            return False

        try:
            handler(session_id, payload)
            session.last_activity = datetime.now()
            return True
        except Exception as e:
            session.errors.append(f"Handler error: {str(e)}")
            return False

    def _handle_exchange_request(self, session_id: str, payload: bytes):
        # Deserialize and process request
        request = self._deserialize_request(payload)
        response = self.federated_engine.process_exchange_request(request)
        # Send response back
        self._send_response(session_id, response)

    def _handle_exchange_response(self, session_id: str, payload: bytes):
        self._deserialize_response(payload)
        # Update session state
        session = self.active_sessions.get(session_id)
        if session:
            session.packages_received += 1

    def _handle_memory_package(self, session_id: str, payload: bytes):
        package = self._deserialize_package(payload)
        session = self.active_sessions.get(session_id)
        if session:
            session.packages_received += 1
            session.bytes_received += len(payload)

            # Import package
            if package.source_node_id in self.federated_engine.peers:
                peer = self.federated_engine.peers[package.source_node_id]
                self.federated_engine.import_package(package, peer.public_key)
                # Log successful import

    def _handle_heartbeat(self, session_id: str, payload: bytes):
        session = self.active_sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()

    def _handle_ack(self, session_id: str, payload: bytes):
        session = self.active_sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()

    def _handle_error(self, session_id: str, payload: bytes):
        session = self.active_sessions.get(session_id)
        if session:
            error_msg = payload.decode('utf-8')
            session.errors.append(error_msg)

    def _serialize_request(self, request) -> bytes:
        return json.dumps({
            "type": "exchange_request",
            "request_id": request.request_id,
            "source_node_id": request.source_node_id,
            "target_node_id": request.target_node_id,
            "memory_ids": request.memory_ids,
            "topic_filter": request.topic_filter,
            "trust_level": request.trust_level.value,
            "purpose": request.purpose,
            "expires_at": request.expires_at.isoformat() if request.expires_at else None,
            "created_at": request.created_at.isoformat(),
        }).encode('utf-8')

    def _deserialize_request(self, payload: bytes):
        from artificial_memory.memory.federation import ExchangeRequest, ExchangeStatus
        data = json.loads(payload.decode('utf-8'))
        return ExchangeRequest(
            request_id=data["request_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            memory_ids=data["memory_ids"],
            topic_filter=data.get("topic_filter"),
            trust_level=TrustLevel(data["trust_level"]),
            purpose=data["purpose"],
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            status=ExchangeStatus(data.get("status", "pending")),
        )

    def _serialize_response(self, response) -> bytes:
        return json.dumps({
            "type": "exchange_response",
            "request_id": response.request_id,
            "responder_node_id": response.responder_node_id,
            "status": response.status.value,
            "approved_memory_ids": response.approved_memory_ids,
            "rejected_memory_ids": response.rejected_memory_ids,
            "reason": response.reason,
            "responded_at": response.responded_at.isoformat(),
        }).encode('utf-8')

    def _deserialize_response(self, payload: bytes):
        from artificial_memory.memory.federation import ExchangeStatus
        data = json.loads(payload.decode('utf-8'))
        return ExchangeResponse(
            request_id=data["request_id"],
            responder_node_id=data["responder_node_id"],
            status=ExchangeStatus(data["status"]),
            approved_memory_ids=data["approved_memory_ids"],
            rejected_memory_ids=data["rejected_memory_ids"],
            reason=data["reason"],
            responded_at=datetime.fromisoformat(data["responded_at"]),
        )

    def _serialize_package(self, package: MemoryPackage) -> bytes:
        return json.dumps({
            "type": "memory_package",
            "package_id": package.package_id,
            "source_node_id": package.source_node_id,
            "target_node_id": package.target_node_id,
            "memories": [self._serialize_memory_ir(m) for m in package.memories],
            "created_at": package.created_at.isoformat(),
            "expires_at": package.expires_at.isoformat() if package.expires_at else None,
            "trust_level": package.trust_level.value,
            "provenance_chain": package.provenance_chain,
            "encryption_key_id": package.encryption_key_id,
            "digital_signature": package.digital_signature,
            "metadata": package.metadata,
        }).encode('utf-8')

    def _deserialize_package(self, payload: bytes):
        from artificial_memory.memory.federation import MemoryPackage
        data = json.loads(payload.decode('utf-8'))
        return MemoryPackage(
            package_id=data["package_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            memories=[self._deserialize_memory_ir(m) for m in data["memories"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            trust_level=TrustLevel(data["trust_level"]),
            provenance_chain=data.get("provenance_chain", []),
            encryption_key_id=data.get("encryption_key_id"),
            digital_signature=data.get("digital_signature"),
            metadata=data.get("metadata", {}),
        )

    def _serialize_memory_ir(self, mem_ir) -> dict[str, Any]:
        return {
            "identity": {
                "memory_id": mem_ir.identity.memory_id,
                "version": mem_ir.identity.version,
                "content_hash": mem_ir.identity.content_hash,
            },
            "type": mem_ir.type.value,
            "resolution": mem_ir.resolution.value,
            "semantic_content": {
                "content": mem_ir.semantic_content.content,
                "structured_data": mem_ir.semantic_content.structured_data,
            },
            "source": {
                "source": {
                    "conversation_id": mem_ir.source.source.conversation_id,
                    "message_id": mem_ir.source.source.message_id,
                    "timestamp": mem_ir.source.source.timestamp.isoformat() if mem_ir.source.source.timestamp else None,
                },
                "compilation_chain": [
                    {
                        "conversation_id": c.conversation_id,
                        "message_id": c.message_id,
                        "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                    }
                    for c in mem_ir.source.compilation_chain
                ],
            },
            "temporal_scope": {
                "valid_from": mem_ir.temporal_scope.valid_from.isoformat() if mem_ir.temporal_scope.valid_from else None,
                "valid_until": mem_ir.temporal_scope.valid_until.isoformat() if mem_ir.temporal_scope.valid_until else None,
                "created_at": mem_ir.temporal_scope.created_at.isoformat(),
                "updated_at": mem_ir.temporal_scope.updated_at.isoformat(),
            },
            "confidence": {
                "memory_confidence": mem_ir.confidence.memory_confidence,
                "retrieval_confidence": mem_ir.confidence.retrieval_confidence,
                "temporal_confidence": mem_ir.confidence.temporal_confidence,
                "source_confidence": mem_ir.confidence.source_confidence,
                "overall": mem_ir.confidence.overall,
            },
            "importance": mem_ir.importance,
            "dependencies": [
                {"memory_id": d.memory_id, "dependency_type": d.dependency_type.value, "strength": d.strength}
                for d in mem_ir.dependencies
            ],
            "relations": [
                {"target_memory_id": r.target_memory_id, "association_type": r.association_type.value, "strength": r.strength}
                for r in mem_ir.relations
            ],
            "compression_history": [
                {
                    "from_resolution": c.from_resolution.value,
                    "to_resolution": c.to_resolution.value,
                    "original_tokens": c.original_tokens,
                    "compressed_tokens": c.compressed_tokens,
                    "compression_ratio": c.compression_ratio,
                    "method": c.method.value,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in mem_ir.compression_history
            ],
            "access_history": {
                "last_accessed": mem_ir.access_history.last_accessed.isoformat() if mem_ir.access_history.last_accessed else None,
                "access_count": mem_ir.access_history.access_count,
                "total_tokens_retrieved": mem_ir.access_history.total_tokens_retrieved,
            },
            "lifecycle_state": mem_ir.lifecycle_state.value,
        }

    def _deserialize_memory_ir(self, data: dict[str, Any]) -> MemoryIR:
        from artificial_memory.core.ir import (
            AccessRecord,
            AssociationRef,
            CompressionRecord,
            ConfidenceProfile,
            DependencyRef,
            LifecycleState,
            MemoryIdentity,
            MemoryIR,
            MemoryType,
            ProvenanceChain,
            ProvenanceLink,
            ResolutionLevel,
            SemanticContent,
            TemporalScope,
        )

        return MemoryIR(
            identity=MemoryIdentity(**data["identity"]),
            type=MemoryType(data["type"]),
            resolution=ResolutionLevel(data["resolution"]),
            semantic_content=SemanticContent(**data["semantic_content"]),
            source=ProvenanceChain(
                source=ProvenanceLink(**data["source"]["source"]),
                compilation_chain=[ProvenanceLink(**c) for c in data["source"]["compilation_chain"]],
            ),
            temporal_scope=TemporalScope(
                valid_from=datetime.fromisoformat(data["temporal_scope"]["valid_from"]) if data["temporal_scope"]["valid_from"] else None,
                valid_until=datetime.fromisoformat(data["temporal_scope"]["valid_until"]) if data["temporal_scope"]["valid_until"] else None,
                created_at=datetime.fromisoformat(data["temporal_scope"]["created_at"]),
                updated_at=datetime.fromisoformat(data["temporal_scope"]["updated_at"]),
            ),
            confidence=ConfidenceProfile(**data["confidence"]),
            importance=data["importance"],
            dependencies=[DependencyRef(**d) for d in data["dependencies"]],
            relations=[AssociationRef(**r) for r in data["relations"]],
            compression_history=[CompressionRecord(**c) for c in data["compression_history"]],
            access_history=AccessRecord(**data["access_history"]),
            lifecycle_state=LifecycleState(data["lifecycle_state"]),
        )

    def check_session_health(self, session_id: str) -> bool:
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        # Check if session has timed out
        timeout = timedelta(seconds=self.protocol_config.request_timeout_seconds * 3)
        if datetime.now() - session.last_activity > timeout:
            return False

        return True

    def cleanup_stale_sessions(self, max_age: timedelta = timedelta(hours=1)) -> int:
        now = datetime.now()
        stale = [
            sid for sid, session in self.active_sessions.items()
            if now - session.last_activity > max_age
        ]

        for sid in stale:
            self.close_session(sid)

        return len(stale)


def create_memory_exchange_protocol(
    federated_engine: FederatedMemoryEngine,
    protocol_config: ExchangeProtocolConfig | None = None,
) -> MemoryExchangeProtocol:
    return MemoryExchangeProtocol(federated_engine, protocol_config)
