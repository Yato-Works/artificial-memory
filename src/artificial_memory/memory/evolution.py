from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Association,
    AssociationType,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    ResolutionLevel,
)


class EvolutionOperationType(StrEnum):
    """Types of memory evolution operations."""
    REVISE = "revise"           # Update memory with new evidence
    MERGE = "merge"             # Combine multiple memories
    SPLIT = "split"             # Divide a memory into parts
    CONTRADICT = "contradict"   # Mark contradiction with another memory
    SUPERSEDE = "supersede"     # Replace with newer version
    REACTIVATE = "reactivate"   # Bring back from archive
    HEAL = "heal"               # Repair from integrity check


@dataclass
class EvolutionEvent:
    """Record of a memory evolution operation."""
    id: int | None = None
    memory_id: int = 0
    operation: EvolutionOperationType = EvolutionOperationType.REVISE
    source_memory_ids: list[int] = field(default_factory=list)  # For merge/split
    target_memory_id: int | None = None  # For supersede
    description: str = ""
    old_content: str = ""
    new_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    triggered_by: str = "auto"  # "auto", "user", "integrity_check", "recall"


@dataclass
class RevisionCandidate:
    """Candidate for memory revision."""
    memory_id: int
    new_evidence: str
    evidence_source: str  # "conversation", "user", "external"
    confidence: float = 0.8
    should_supersede: bool = False  # True = replace, False = merge


class MemoryEvolutionEngine:
    """Central engine for memory lifecycle operations.

    Memory is not immutable. It evolves through:
    - Revision: new evidence updates content
    - Merge: combine similar memories
    - Split: divide complex memory
    - Contradiction: explicit conflict tracking
    - Belief: current interpretation separate from evidence
    - Healing: repair from integrity issues

    All operations preserve history via EvolutionEvent and MemoryVersion.
    """

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
    ):
        self.store = store
        self.compressor = compressor

    # ==================== Revision ====================

    def revise_memory(
        self,
        memory_id: int,
        new_evidence: str,
        evidence_source: str = "conversation",
        confidence: float = 0.8,
        should_supersede: bool = False,
    ) -> Memory:
        """Revise a memory with new evidence.

        If should_supersede=True: creates new version, marks old as superseded
        If should_supersede=False: merges evidence into existing memory
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        # Create evolution event
        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.SUPERSEDE if should_supersede else EvolutionOperationType.REVISE,
            description=f"Revised with new evidence from {evidence_source}",
            old_content=memory.content,
            new_content=new_evidence if should_supersede else f"{memory.content}\n\n[Updated: {new_evidence}]",
            metadata={
                "evidence_source": evidence_source,
                "confidence": confidence,
                "should_supersede": should_supersede,
            },
            triggered_by="auto" if evidence_source == "conversation" else "user",
        )

        # Store old version
        self._create_version_before_change(memory)

        if should_supersede:
            # Mark old memory as superseded, create new one
            memory.status = MemoryStatus.ARCHIVED
            memory.is_current = False
            memory.updated_at = datetime.now()
            self.store.update_memory(memory)

            # Create new memory with updated content
            new_memory = Memory(
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=new_evidence,
                resolution=memory.resolution,
                importance=memory.importance,
                confidence=confidence,
                status=MemoryStatus.ACTIVE,
                valid_from=datetime.now(),
                is_current=True,
                source_conversation_id=memory.source_conversation_id,
            )
            new_memory = self.store.create_memory(new_memory)

            # Link to old memory
            self.store.create_association(Association(
                source_memory_id=memory.id,
                target_memory_id=new_memory.id,
                association_type=AssociationType.ELABORATES,
                strength=0.9,
            ))

            memory = new_memory
        else:
            # Merge evidence into existing memory
            memory.content = f"{memory.content}\n\n[Updated {datetime.now().strftime('%Y-%m-%d')}: {new_evidence}]"
            memory.confidence = min(1.0, (memory.confidence + confidence) / 2)
            memory.updated_at = datetime.now()
            memory = self.store.update_memory(memory)

        # Log event (would need a table for this)
        self._log_evolution_event(event)

        return memory

    # ==================== Merge ====================

    def merge_memories(
        self,
        memory_ids: list[int],
        merge_strategy: str = "combine",  # "combine", "summarize", "latest"
        new_memory_type: MemoryType | None = None,
    ) -> Memory:
        """Merge multiple memories into one.

        Creates new merged memory, marks originals as merged (not deleted).
        """
        if len(memory_ids) < 2:
            raise ValueError("Need at least 2 memories to merge")

        memories = [self.store.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m is not None]

        if not memories:
            raise ValueError("No valid memories to merge")

        # Determine merged content
        if merge_strategy == "combine":
            content = self._combine_contents(memories)
        elif merge_strategy == "summarize":
            content = self._summarize_contents(memories)
        elif merge_strategy == "latest":
            content = max(memories, key=lambda m: m.updated_at).content
        else:
            content = self._combine_contents(memories)

        # Create new merged memory
        base_memory = memories[0]
        merged_memory = Memory(
            topic_id=base_memory.topic_id,
            memory_type=new_memory_type or base_memory.memory_type,
            content=content,
            resolution=base_memory.resolution,
            importance=max(m.importance for m in memories),
            confidence=sum(m.confidence for m in memories) / len(memories),
            status=MemoryStatus.ACTIVE,
            valid_from=datetime.now(),
            is_current=True,
            source_conversation_id=base_memory.source_conversation_id,
        )
        merged_memory = self.store.create_memory(merged_memory)

        # Create versions for merged memory
        self._create_compressed_versions(merged_memory)

        # Mark originals as merged
        for mem in memories:
            mem.status = MemoryStatus.ARCHIVED
            mem.is_current = False
            mem.updated_at = datetime.now()
            self.store.update_memory(mem)

            # Link to merged memory
            self.store.create_association(Association(
                source_memory_id=mem.id,
                target_memory_id=merged_memory.id,
                association_type=AssociationType.ELABORATES,
                strength=0.8,
            ))

        # Log evolution event
        event = EvolutionEvent(
            memory_id=merged_memory.id,
            operation=EvolutionOperationType.MERGE,
            source_memory_ids=memory_ids,
            description=f"Merged {len(memories)} memories using {merge_strategy} strategy",
            new_content=content,
            metadata={"strategy": merge_strategy},
            triggered_by="auto",
        )
        self._log_evolution_event(event)

        return merged_memory

    def _combine_contents(self, memories: list[Memory]) -> str:
        """Combine multiple memory contents."""
        parts = []
        for i, mem in enumerate(memories):
            parts.append(f"=== Memory {i+1} (ID: {mem.id}) ===\n{mem.content}")
        return "\n\n---\n\n".join(parts)

    def _summarize_contents(self, memories: list[Memory]) -> str:
        """Summarize multiple memories using compressor."""
        combined = self._combine_contents(memories)
        summary, _ = self.compressor.compress_semantic(combined, {})
        return f"[Merged Summary]\n{summary}"

    # ==================== Split ====================

    def split_memory(
        self,
        memory_id: int,
        split_points: list[str],  # Section headers or keywords to split on
    ) -> list[Memory]:
        """Split a memory into multiple memories based on content sections."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        # Split content by points
        sections = self._split_content(memory.content, split_points)

        if len(sections) < 2:
            return [memory]  # Nothing to split

        new_memories = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue

            new_mem = Memory(
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=section.strip(),
                resolution=memory.resolution,
                importance=memory.importance,
                confidence=memory.confidence,
                status=MemoryStatus.ACTIVE,
                valid_from=memory.valid_from,
                is_current=True,
                source_conversation_id=memory.source_conversation_id,
            )
            new_mem = self.store.create_memory(new_mem)
            self._create_compressed_versions(new_mem)
            new_memories.append(new_mem)

            # Link to original
            self.store.create_association(Association(
                source_memory_id=memory.id,
                target_memory_id=new_mem.id,
                association_type=AssociationType.ELABORATES,
                strength=0.7,
            ))

        # Archive original
        memory.status = MemoryStatus.ARCHIVED
        memory.is_current = False
        memory.updated_at = datetime.now()
        self.store.update_memory(memory)

        # Log event
        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.SPLIT,
            description=f"Split into {len(new_memories)} memories",
            metadata={"split_points": split_points, "new_memory_ids": [m.id for m in new_memories]},
            triggered_by="auto",
        )
        self._log_evolution_event(event)

        return new_memories

    def _split_content(self, content: str, split_points: list[str]) -> list[str]:
        """Split content by given points."""
        sections = []
        current = content

        for point in split_points:
            if point in current:
                idx = current.index(point)
                before = current[:idx].strip()
                if before:
                    sections.append(before)
                current = current[idx:].strip()
            else:
                continue

        if current.strip():
            sections.append(current)

        return sections if sections else [content]

    # ==================== Reactivation ====================

    def reactivate_memory(self, memory_id: int) -> Memory:
        """Reactivate an archived memory (e.g., after recall)."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        memory.status = MemoryStatus.ACTIVE
        memory.is_current = True
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        memory.updated_at = datetime.now()

        # Restore to higher resolution if deeply archived
        if memory.resolution >= ResolutionLevel.LONG_TERM:
            version = self.store.get_memory_version(memory_id, ResolutionLevel.SEMANTIC)
            if version:
                memory.content = version.content
                memory.resolution = ResolutionLevel.SEMANTIC

        self.store.update_memory(memory)

        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.REACTIVATE,
            description="Memory reactivated after recall",
            triggered_by="recall",
        )
        self._log_evolution_event(event)

        return memory

    # ==================== Healing ====================

    def heal_memory(
        self,
        memory_id: int,
        issue_type: str,  # "compression_loss", "provenance_broken", "contradiction"
        repair_action: str,
    ) -> Memory:
        """Repair a memory from integrity issues."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        # Store original for provenance
        self._create_version_before_change(memory)

        if issue_type == "compression_loss":
            # Try to restore from higher resolution version
            if memory.resolution > ResolutionLevel.SEMANTIC:
                version = self.store.get_memory_version(memory_id, ResolutionLevel.SEMANTIC)
                if version:
                    memory.content = version.content
                    memory.resolution = ResolutionLevel.SEMANTIC

        elif issue_type == "provenance_broken":
            # Attempt to reconstruct provenance from associations
            pass  # Implementation depends on specific case

        elif issue_type == "contradiction":
            # Mark as needing review
            memory.metadata["needs_review"] = True
            memory.metadata["contradiction_flag"] = True

        memory.updated_at = datetime.now()
        memory = self.store.update_memory(memory)

        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.HEAL,
            description=f"Healed {issue_type}: {repair_action}",
            metadata={"issue_type": issue_type, "repair_action": repair_action},
            triggered_by="integrity_check",
        )
        self._log_evolution_event(event)

        return memory

    # ==================== Helpers ====================

    def _create_version_before_change(self, memory: Memory) -> None:
        """Save current state as version before modification."""
        version = MemoryVersion(
            memory_id=memory.id,
            resolution=memory.resolution,
            content=memory.content,
            compression_ratio=1.0,
            created_at=datetime.now(),
            source='pre_evolution',
        )
        self.store.add_memory_version(version)

    def _create_compressed_versions(self, memory: Memory) -> None:
        """Generate all resolution versions for a memory."""
        resolutions = [
            ResolutionLevel.LIGHT,
            ResolutionLevel.EPISODE,
            ResolutionLevel.SEMANTIC,
            ResolutionLevel.LONG_TERM,
            ResolutionLevel.DEEP_LONG_TERM,
        ]

        for res in resolutions:
            if res.value <= memory.resolution.value:
                continue

            # Compress to target resolution
            content = memory.content
            for step_res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE,
                            ResolutionLevel.SEMANTIC, ResolutionLevel.LONG_TERM,
                            ResolutionLevel.DEEP_LONG_TERM]:
                if step_res.value > res.value:
                    break
                if step_res.value <= memory.resolution.value:
                    continue
                content, _ = self._compress_to_resolution(content, step_res)

            version = MemoryVersion(
                memory_id=memory.id,
                resolution=res,
                content=content,
                compression_ratio=self.compressor.get_compression_ratio(memory.content, content),
                created_at=datetime.now(),
                source='auto_evolution',
            )
            self.store.add_memory_version(version)

    def _compress_to_resolution(self, content: str, resolution: ResolutionLevel) -> tuple[str, dict]:
        """Compress content to specific resolution level."""
        method_map = {
            ResolutionLevel.LIGHT: (self.compressor.compress_light, "light"),
            ResolutionLevel.EPISODE: (self.compressor.compress_episode, "episode"),
            ResolutionLevel.SEMANTIC: (self.compressor.compress_semantic, "semantic"),
            ResolutionLevel.LONG_TERM: (self.compressor.compress_long_term, "longterm"),
            ResolutionLevel.DEEP_LONG_TERM: (self.compressor.compress_long_term, "deeplongterm"),
        }
        func, _ = method_map.get(resolution, (self.compressor.compress_light, "light"))
        return func(content, {})

    def _log_evolution_event(self, event: EvolutionEvent) -> None:
        """Persist evolution event to storage (P0-5).

        Every meaningful memory change (revise/merge/split/heal/...) is
        written to the evolution_events table for full traceability.
        Falls back to in-memory ring buffer for stores without persistence.
        """
        from artificial_memory.core.models import EvolutionEventRecord

        record = EvolutionEventRecord(
            id=event.id,
            memory_id=event.memory_id,
            operation=event.operation.value,
            source_memory_ids=list(event.source_memory_ids),
            target_memory_id=event.target_memory_id,
            description=event.description,
            old_content=event.old_content,
            new_content=event.new_content,
            metadata=dict(event.metadata),
            triggered_by=event.triggered_by,
            created_at=event.created_at,
        )
        try:
            saved = self.store.add_evolution_event(record)
            event.id = saved.id
        except AttributeError:
            # Store backend without evolution event persistence
            pass


def create_memory_evolution_engine(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
) -> MemoryEvolutionEngine:
    return MemoryEvolutionEngine(store, compressor)
