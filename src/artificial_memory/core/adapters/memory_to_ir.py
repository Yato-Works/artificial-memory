from __future__ import annotations

import hashlib
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.ir import (
    AccessRecord,
    AssociationRef,
    CompressionRecord,
    ConfidenceProfile,
    DependencyRef,
    LifecycleState,
    MemoryIdentity,
    MemoryIR,
    ProvenanceChain,
    ProvenanceLink,
    SemanticContent,
    TemporalScope,
)
from artificial_memory.core.ir import AssociationType as IRAssociationType
from artificial_memory.core.ir import CompressionMethod as IRCompressionMethod
from artificial_memory.core.ir import MemoryStatus as IRMemoryStatus
from artificial_memory.core.ir import MemoryType as IRMemoryType
from artificial_memory.core.ir import ResolutionLevel as IRResolutionLevel
from artificial_memory.core.models import Association, CompressionEvent, Memory, MemoryVersion
from artificial_memory.core.models import AssociationType as LegacyAssociationType
from artificial_memory.core.models import CompressionMethod as LegacyCompressionMethod
from artificial_memory.core.models import MemoryStatus as LegacyMemoryStatus
from artificial_memory.core.models import MemoryType as LegacyMemoryType
from artificial_memory.core.models import ResolutionLevel as LegacyResolutionLevel


def _legacy_to_ir_memory_type(legacy: LegacyMemoryType) -> IRMemoryType:
    return IRMemoryType(legacy.value)


def _legacy_to_ir_resolution_level(legacy: LegacyResolutionLevel) -> IRResolutionLevel:
    return IRResolutionLevel(legacy.value)


def _legacy_to_ir_association_type(legacy: LegacyAssociationType) -> IRAssociationType:
    return IRAssociationType(legacy.value)


def _legacy_to_ir_compression_method(legacy: LegacyCompressionMethod) -> IRCompressionMethod:
    return IRCompressionMethod(legacy.value)


def _ir_to_legacy_memory_type(ir: IRMemoryType) -> LegacyMemoryType:
    return LegacyMemoryType(ir.value)


def _ir_to_legacy_resolution_level(ir: IRResolutionLevel) -> LegacyResolutionLevel:
    return LegacyResolutionLevel(ir.value)


def _ir_to_legacy_association_type(ir: IRAssociationType) -> LegacyAssociationType:
    return LegacyAssociationType(ir.value)


def _ir_to_legacy_compression_method(ir: IRCompressionMethod) -> LegacyCompressionMethod:
    return LegacyCompressionMethod(ir.value)


def _legacy_to_ir_memory_status(legacy: LegacyMemoryStatus) -> IRMemoryStatus:
    return IRMemoryStatus(legacy.value)


def _ir_to_legacy_memory_status(ir: LifecycleState) -> LegacyMemoryStatus:
    mapping = {
        LifecycleState.HOT: LegacyMemoryStatus.ACTIVE,
        LifecycleState.WARM: LegacyMemoryStatus.ACTIVE,
        LifecycleState.COLD: LegacyMemoryStatus.COMPRESSED,
        LifecycleState.ARCHIVED: LegacyMemoryStatus.ARCHIVED,
        LifecycleState.DEEP_ARCHIVED: LegacyMemoryStatus.DEEP_ARCHIVED,
    }
    return mapping.get(ir, LegacyMemoryStatus.ACTIVE)


class MemoryToIRAdapter:
    """Convert legacy Memory to MemoryIR (lossless)."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def to_ir(self, memory: Memory) -> MemoryIR:
        memory_id = memory.id
        assert memory_id is not None, "Memory must have an ID"
        versions = self.store.get_memory_versions(memory_id)
        associations = self.store.get_associations(memory_id)
        compression_events = self.store.get_compression_history(memory_id)

        identity = self._build_identity(memory, versions)
        semantic_content = self._build_semantic_content(memory)
        source = self._build_provenance_chain(memory)
        temporal_scope = self._build_temporal_scope(memory)
        confidence = self._build_confidence_profile(memory)
        dependencies = self._build_dependencies(associations)
        relations = self._build_relations(associations)
        compression_history = self._build_compression_history(compression_events, versions)
        access_history = self._build_access_record(memory)
        lifecycle_state = self._map_status_to_lifecycle(memory.status)
        belief_ref = None

        return MemoryIR(
            identity=identity,
            type=_legacy_to_ir_memory_type(memory.memory_type),
            resolution=_legacy_to_ir_resolution_level(memory.resolution),
            semantic_content=semantic_content,
            source=source,
            temporal_scope=temporal_scope,
            confidence=confidence,
            importance=memory.importance,
            dependencies=dependencies,
            relations=relations,
            compression_history=compression_history,
            access_history=access_history,
            lifecycle_state=lifecycle_state,
            belief_ref=belief_ref,
        )

    def batch_to_ir(self, memories: list[Memory]) -> list[MemoryIR]:
        return [self.to_ir(m) for m in memories]

    def _build_identity(self, memory: Memory, versions: list[MemoryVersion]) -> MemoryIdentity:
        memory_id = memory.id
        assert memory_id is not None
        content_hash = hashlib.sha256(memory.content.encode()).hexdigest()[:16]
        return MemoryIdentity(
            memory_id=memory_id,
            version=len(versions) + 1,
            content_hash=content_hash,
        )

    def _build_semantic_content(self, memory: Memory) -> SemanticContent:
        return SemanticContent(content=memory.content, structured_data=None)

    def _build_provenance_chain(self, memory: Memory) -> ProvenanceChain:
        source = ProvenanceLink(
            conversation_id=memory.source_conversation_id,
            message_id=memory.source_message_id,
            timestamp=memory.created_at,
        )

        return ProvenanceChain(source=source, compilation_chain=[])

    def _build_temporal_scope(self, memory: Memory) -> TemporalScope:
        return TemporalScope(
            valid_from=memory.valid_from,
            valid_until=memory.valid_until,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    def _build_confidence_profile(self, memory: Memory) -> ConfidenceProfile:
        return ConfidenceProfile(
            memory_confidence=memory.confidence,
            retrieval_confidence=memory.importance,
            temporal_confidence=self._compute_temporal_confidence(memory),
            source_confidence=0.9,
            overall=memory.confidence,
        )

    def _compute_temporal_confidence(self, memory: Memory) -> float:
        days_old = (datetime.now() - memory.created_at).days
        return max(0.5, 1.0 - days_old / 365 * 0.3)

    def _build_dependencies(self, associations: list[Association]) -> list[DependencyRef]:
        deps = []
        for assoc in associations:
            if assoc.association_type in (LegacyAssociationType.CAUSES, LegacyAssociationType.FOLLOWS, LegacyAssociationType.ELABORATES):
                deps.append(DependencyRef(
                    memory_id=assoc.target_memory_id if assoc.source_memory_id != assoc.target_memory_id else assoc.source_memory_id,
                    dependency_type=_legacy_to_ir_association_type(assoc.association_type),
                    strength=assoc.strength,
                ))
        return deps

    def _build_relations(self, associations: list[Association]) -> list[AssociationRef]:
        relations = []
        for assoc in associations:
            relations.append(AssociationRef(
                target_memory_id=assoc.target_memory_id if assoc.source_memory_id != assoc.target_memory_id else assoc.source_memory_id,
                association_type=_legacy_to_ir_association_type(assoc.association_type),
                strength=assoc.strength,
            ))
        return relations

    def _build_compression_history(
        self,
        events: list[CompressionEvent],
        versions: list[MemoryVersion]
    ) -> list[CompressionRecord]:
        records = []
        for event in events:
            records.append(CompressionRecord(
                from_resolution=_legacy_to_ir_resolution_level(event.from_resolution),
                to_resolution=_legacy_to_ir_resolution_level(event.to_resolution),
                original_tokens=event.original_tokens,
                compressed_tokens=event.compressed_tokens,
                compression_ratio=event.compression_ratio,
                method=_legacy_to_ir_compression_method(event.method),
                timestamp=event.created_at,
            ))
        return records

    def _build_access_record(self, memory: Memory) -> AccessRecord:
        return AccessRecord(
            last_accessed=memory.last_accessed,
            access_count=memory.access_count,
            total_tokens_retrieved=0,
        )

    def _map_status_to_lifecycle(self, status: LegacyMemoryStatus) -> LifecycleState:
        mapping = {
            LegacyMemoryStatus.ACTIVE: LifecycleState.HOT,
            LegacyMemoryStatus.DORMANT: LifecycleState.WARM,
            LegacyMemoryStatus.COMPRESSED: LifecycleState.COLD,
            LegacyMemoryStatus.ARCHIVED: LifecycleState.ARCHIVED,
            LegacyMemoryStatus.DEEP_ARCHIVED: LifecycleState.DEEP_ARCHIVED,
        }
        return mapping.get(status, LifecycleState.HOT)


class IRToMemoryAdapter:
    """Convert MemoryIR back to legacy Memory (lossless)."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def to_legacy(self, ir: MemoryIR) -> tuple[Memory, list[MemoryVersion], list[Association], list[CompressionEvent]]:
        memory = self._to_memory(ir)
        versions = self._to_versions(ir)
        associations = self._to_associations(ir)
        compression_events = self._to_compression_events(ir)

        return memory, versions, associations, compression_events

    def _to_memory(self, ir: MemoryIR) -> Memory:
        legacy_dict = ir.to_legacy_dict()
        # Convert IR enums to legacy enums
        legacy_dict["memory_type"] = _ir_to_legacy_memory_type(ir.type)
        legacy_dict["resolution"] = _ir_to_legacy_resolution_level(ir.resolution)
        legacy_dict["status"] = _ir_to_legacy_memory_status(ir.lifecycle_state)
        return Memory(**legacy_dict)

    def _to_versions(self, ir: MemoryIR) -> list[MemoryVersion]:
        versions = []
        for record in ir.compression_history:
            version = self.store.get_memory_version(ir.identity.memory_id, _ir_to_legacy_resolution_level(record.to_resolution))
            if version:
                versions.append(version)
        return versions

    def _to_associations(self, ir: MemoryIR) -> list[Association]:
        associations = []
        for rel in ir.relations:
            associations.append(Association(
                source_memory_id=ir.identity.memory_id,
                target_memory_id=rel.target_memory_id,
                association_type=_ir_to_legacy_association_type(rel.association_type),
                strength=rel.strength,
                created_at=datetime.now(),
            ))
        return associations

    def _to_compression_events(self, ir: MemoryIR) -> list[CompressionEvent]:
        events = []
        for record in ir.compression_history:
            events.append(CompressionEvent(
                source_memory_id=ir.identity.memory_id,
                target_memory_id=None,
                from_resolution=_ir_to_legacy_resolution_level(record.from_resolution),
                to_resolution=_ir_to_legacy_resolution_level(record.to_resolution),
                original_tokens=record.original_tokens,
                compressed_tokens=record.compressed_tokens,
                compression_ratio=record.compression_ratio,
                method=_ir_to_legacy_compression_method(record.method),
                created_at=record.timestamp,
            ))
        return events


def create_memory_ir_adapter(store: MemoryStore) -> MemoryToIRAdapter:
    return MemoryToIRAdapter(store)


def create_ir_memory_adapter(store: MemoryStore) -> IRToMemoryAdapter:
    return IRToMemoryAdapter(store)
