from __future__ import annotations

import time
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import (
    Memory,
    MemoryStatus,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
)


class BasicRecallEngine:
    """Basic recall engine for MVP - levels 0-2."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def recall(
        self,
        query: str,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], int]:
        """Recall memories based on query and level."""
        start_time = time.time()

        # Get candidate memories based on level
        candidates = self._get_candidates(query, topic_id, level)

        # Rank by relevance (simple scoring for MVP)
        ranked = self._rank_memories(query, candidates)

        # Select within token budget
        selected, total_tokens = self._select_within_budget(ranked, max_tokens)

        # Update access stats
        for memory in selected:
            memory.touch()
            self.store.update_memory(memory)

        # Log recall event
        latency_ms = int((time.time() - start_time) * 1000)
        recall_event = RecallEvent(
            query=query,
            topic_id=topic_id,
            recall_level=level,
            memories_retrieved=len(selected),
            tokens_returned=total_tokens,
            latency_ms=latency_ms,
        )
        self.store.log_recall(recall_event)

        return selected, total_tokens

    def _get_candidates(
        self,
        query: str,
        topic_id: int | None,
        level: RecallLevel
    ) -> list[Memory]:
        """Get candidate memories based on recall level."""
        candidates = []

        if level == RecallLevel.CURRENT_ONLY:
            # Level 0: Current state only
            memories = self.store.get_memories(
                topic_id=topic_id,
                is_current=True,
                status=MemoryStatus.ACTIVE,
                limit=50
            )
            candidates.extend(memories)

        elif level == RecallLevel.LONG_TERM_SUMMARY:
            # Level 1: Long-term summary (semantic + longterm resolutions)
            for res in [ResolutionLevel.SEMANTIC, ResolutionLevel.LONG_TERM, ResolutionLevel.DEEP_LONG_TERM]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=30
                )
                candidates.extend(memories)

        elif level == RecallLevel.EPISODE:
            # Level 2: Episode memories
            memories = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.EPISODE,
                status=MemoryStatus.ACTIVE,
                limit=40
            )
            candidates.extend(memories)
            # Also include semantic for context
            semantic = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.SEMANTIC,
                status=MemoryStatus.ACTIVE,
                limit=20
            )
            candidates.extend(semantic)

        elif level == RecallLevel.LIGHT_COMPRESSION:
            # Level 3: Light compression
            memories = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.LIGHT,
                status=MemoryStatus.ACTIVE,
                limit=30
            )
            candidates.extend(memories)
            # Include episode and semantic
            for res in [ResolutionLevel.EPISODE, ResolutionLevel.SEMANTIC]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=15
                )
                candidates.extend(memories)

        elif level == RecallLevel.RAW:
            # Level 4: Raw conversation (would need conversation store)
            # For MVP, return light + episode
            for res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE, ResolutionLevel.SEMANTIC]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=20
                )
                candidates.extend(memories)

        # Deduplicate by memory ID
        seen = set()
        unique = []
        for m in candidates:
            if m.id not in seen:
                seen.add(m.id)
                unique.append(m)

        return unique

    def _rank_memories(self, query: str, memories: list[Memory]) -> list[Memory]:
        """Simple relevance ranking for MVP."""
        query_words = set(query.lower().split())

        scored = []
        for memory in memories:
            score = 0.0

            # Content overlap
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)
            score += overlap * 0.5

            # Importance boost
            score += memory.importance * 0.3

            # Confidence boost
            score += memory.confidence * 0.2

            # Recency boost (more recent = higher)
            if memory.updated_at:
                days_old = (datetime.now() - memory.updated_at).days
                recency_score = max(0, 1.0 - days_old / 365)
                score += recency_score * 0.2

            # Access count boost (frequently accessed = more relevant)
            if memory.access_count > 0:
                score += min(memory.access_count * 0.05, 0.3)

            scored.append((memory, score))

        scored.sort(key=lambda x: -x[1])
        return [m for m, _ in scored]

    def _select_within_budget(
        self,
        ranked_memories: list[Memory],
        max_tokens: int
    ) -> tuple[list[Memory], int]:
        """Select memories within token budget."""
        # Estimate tokens (rough: 1 token ≈ 4 chars for Japanese, 3-4 for English)
        def estimate_tokens(text: str) -> int:
            return max(1, len(text) // 3)

        selected = []
        total_tokens = 0

        for memory in ranked_memories:
            mem_tokens = estimate_tokens(memory.content)
            if total_tokens + mem_tokens <= max_tokens:
                selected.append(memory)
                total_tokens += mem_tokens
            else:
                break

        return selected, total_tokens

    def expand_resolution(self, memory: Memory, target_resolution: ResolutionLevel) -> Memory | None:
        """Expand memory to higher resolution (lower resolution number)."""
        if target_resolution.value >= memory.resolution.value:
            return None  # Already at or below target resolution

        # Special case: expanding to RAW returns original content
        if target_resolution == ResolutionLevel.RAW:
            expanded = Memory(
                id=memory.id,
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=memory.content,  # Original content is RAW
                resolution=ResolutionLevel.RAW,
                importance=memory.importance,
                confidence=memory.confidence,
                status=memory.status,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                is_current=memory.is_current,
                source_conversation_id=memory.source_conversation_id,
                source_message_id=memory.source_message_id,
                created_at=memory.created_at,
                updated_at=datetime.now(),
            )
            return expanded

        # Check if version exists
        version = self.store.get_memory_version(memory.id, target_resolution)
        if version:
            # Create expanded memory from version
            expanded = Memory(
                id=memory.id,
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=version.content,
                resolution=target_resolution,
                importance=memory.importance,
                confidence=memory.confidence,
                status=memory.status,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                is_current=memory.is_current,
                source_conversation_id=memory.source_conversation_id,
                source_message_id=memory.source_message_id,
                created_at=memory.created_at,
                updated_at=datetime.now(),
            )
            return expanded

        return None

    def get_memory_provenance(self, memory: Memory) -> list[Memory]:
        """Get provenance chain: memory -> episode -> raw."""
        chain = [memory]

        # Get versions at lower resolutions (higher detail)
        for res in ResolutionLevel:
            if res.value < memory.resolution.value:
                version = self.store.get_memory_version(memory.id, res)
                if version:
                    chain.append(Memory(
                        id=memory.id,
                        topic_id=memory.topic_id,
                        memory_type=memory.memory_type,
                        content=version.content,
                        resolution=res,
                        importance=memory.importance,
                        confidence=memory.confidence,
                        status=memory.status,
                        valid_from=memory.valid_from,
                        valid_until=memory.valid_until,
                        is_current=memory.is_current,
                        source_conversation_id=memory.source_conversation_id,
                        source_message_id=memory.source_message_id,
                        created_at=memory.created_at,
                        updated_at=datetime.now(),
                    ))

        return chain

    def get_full_provenance(self, memory: Memory) -> dict:
        """Get full provenance including source conversation and messages."""
        result = {
            "memory": memory,
            "versions": self.get_memory_provenance(memory),
            "source_conversation": None,
            "source_messages": [],
        }

        # Get source conversation
        if memory.source_conversation_id:
            conv = self.store.get_conversation(memory.source_conversation_id)
            result["source_conversation"] = conv

            # Get source messages
            if conv:
                messages = self.store.get_messages(conv.id)
                result["source_messages"] = messages

        # If specific source message, highlight it
        if memory.source_message_id:
            for msg in result["source_messages"]:
                if msg.id == memory.source_message_id:
                    result["exact_source_message"] = msg
                    break

        return result

    def search_by_keywords(self, keywords: list[str], topic_id: int | None = None, limit: int = 20) -> list[Memory]:
        """Search memories by keywords."""
        all_memories = self.store.get_memories(topic_id=topic_id, limit=200)

        keyword_set = set(k.lower() for k in keywords)
        scored = []

        for memory in all_memories:
            content_lower = memory.content.lower()
            matches = sum(1 for kw in keyword_set if kw in content_lower)
            if matches > 0:
                score = matches * 0.5 + memory.importance * 0.3
                scored.append((memory, score))

        scored.sort(key=lambda x: -x[1])
        return [m for m, _ in scored[:limit]]

    def search_by_time_range(
        self,
        start: datetime,
        end: datetime,
        topic_id: int | None = None,
        limit: int = 20
    ) -> list[Memory]:
        """Search memories by time range."""
        all_memories = self.store.get_memories(topic_id=topic_id, limit=200)

        filtered = [
            m for m in all_memories
            if m.created_at >= start and m.created_at <= end
        ]

        filtered.sort(key=lambda m: m.created_at, reverse=True)
        return filtered[:limit]

    def search_associations(self, memory: Memory, max_depth: int = 2, min_strength: float = 0.3) -> list[Memory]:
        """Search associated memories via association graph."""
        visited = set()
        results = []
        current_level = [memory]

        for depth in range(max_depth):
            next_level = []
            for mem in current_level:
                if mem.id in visited:
                    continue
                visited.add(mem.id)

                related = self.store.get_related_memories(mem.id, min_strength)
                for related_mem, assoc in related:
                    if related_mem.id not in visited:
                        results.append(related_mem)
                        next_level.append(related_mem)

            current_level = next_level
            if not current_level:
                break

        return results


def create_recall_engine(store: MemoryStore) -> RecallEngine:
    """Factory function to create recall engine."""
    return BasicRecallEngine(store)
