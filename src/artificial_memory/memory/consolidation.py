from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    CompressionEvent,
    CompressionMethod,
    Memory,
    MemoryStatus,
    ResolutionLevel,
)


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation behavior."""
    # Time thresholds (days)
    active_to_dormant_days: int = 7
    dormant_to_compressed_days: int = 30
    compressed_to_archived_days: int = 90
    archived_to_deep_archived_days: int = 365

    # Importance thresholds
    high_importance: float = 0.7
    low_importance: float = 0.3

    # Access thresholds
    recent_access_days: int = 14
    min_access_count_for_active: int = 2

    # Resolution rules
    compress_raw_to_light: bool = True
    compress_light_to_episode: bool = True
    compress_episode_to_semantic: bool = True
    compress_semantic_to_longterm: bool = True


class ConsolidationEngine:
    """Handles memory lifecycle transitions and compression based on consolidation rules."""

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
        config: ConsolidationConfig | None = None,
    ):
        self.store = store
        self.compressor = compressor
        self.config = config or ConsolidationConfig()

    def evaluate_memory(self, memory: Memory) -> MemoryStatus:
        """Evaluate what status a memory should have based on rules."""
        now = datetime.now()
        age_days = (now - memory.created_at).days
        # Use created_at as fallback for last_accessed
        last_accessed = memory.last_accessed if memory.last_accessed else memory.created_at
        days_since_access = (now - last_accessed).days

        # High importance + recently accessed -> ACTIVE
        if memory.importance >= self.config.high_importance and days_since_access <= self.config.recent_access_days:
            return MemoryStatus.ACTIVE

        # High importance + not accessed recently -> DORMANT
        if memory.importance >= self.config.high_importance:
            return MemoryStatus.DORMANT

        # Low importance + old -> DEEP_ARCHIVED
        if memory.importance <= self.config.low_importance and age_days > self.config.archived_to_deep_archived_days:
            return MemoryStatus.DEEP_ARCHIVED

        # Low importance + medium age -> ARCHIVED
        if memory.importance <= self.config.low_importance and age_days > self.config.compressed_to_archived_days:
            return MemoryStatus.ARCHIVED

        # Medium importance, check age-based transitions
        if age_days > self.config.archived_to_deep_archived_days:
            return MemoryStatus.DEEP_ARCHIVED
        elif age_days > self.config.compressed_to_archived_days:
            return MemoryStatus.ARCHIVED
        elif age_days > self.config.dormant_to_compressed_days:
            return MemoryStatus.COMPRESSED
        elif age_days > self.config.active_to_dormant_days:
            return MemoryStatus.DORMANT

        return MemoryStatus.ACTIVE

    def get_target_resolution(self, memory: Memory, new_status: MemoryStatus) -> ResolutionLevel:
        """Determine target resolution based on new status."""
        current_res = memory.resolution

        # Resolution progression based on status
        if new_status == MemoryStatus.DORMANT and current_res == ResolutionLevel.RAW:
            return ResolutionLevel.LIGHT
        elif new_status == MemoryStatus.COMPRESSED:
            if current_res == ResolutionLevel.RAW:
                return ResolutionLevel.LIGHT
            elif current_res == ResolutionLevel.LIGHT:
                return ResolutionLevel.EPISODE
            elif current_res == ResolutionLevel.EPISODE:
                return ResolutionLevel.SEMANTIC
        elif new_status == MemoryStatus.ARCHIVED:
            if current_res <= ResolutionLevel.EPISODE:
                return ResolutionLevel.SEMANTIC
            elif current_res == ResolutionLevel.SEMANTIC:
                return ResolutionLevel.LONG_TERM
        elif new_status == MemoryStatus.DEEP_ARCHIVED:
            return ResolutionLevel.DEEP_LONG_TERM

        return current_res

    def compress_memory(self, memory: Memory, target_resolution: ResolutionLevel) -> Memory:
        """Compress memory to target resolution."""
        # Higher enum value = more compressed (less detail)
        if target_resolution.value <= memory.resolution.value:
            return memory  # Already at or beyond target compression

        content = memory.content

        # Map resolution to compression method
        method_map = {
            ResolutionLevel.LIGHT: CompressionMethod.LIGHT,
            ResolutionLevel.EPISODE: CompressionMethod.EPISODE,
            ResolutionLevel.SEMANTIC: CompressionMethod.SEMANTIC,
            ResolutionLevel.LONG_TERM: CompressionMethod.LONG_TERM,
            ResolutionLevel.DEEP_LONG_TERM: CompressionMethod.DEEP_LONG_TERM,
        }
        method = method_map.get(target_resolution, CompressionMethod.LIGHT)

        # Apply progressive compression
        if target_resolution == ResolutionLevel.LIGHT and memory.resolution == ResolutionLevel.RAW:
            content, meta = self.compressor.compress_light(content, {})
        elif target_resolution == ResolutionLevel.EPISODE:
            if memory.resolution == ResolutionLevel.RAW:
                content, _ = self.compressor.compress_light(content, {})
            content, meta = self.compressor.compress_episode(content, {})
        elif target_resolution == ResolutionLevel.SEMANTIC:
            if memory.resolution <= ResolutionLevel.LIGHT:
                content, _ = self.compressor.compress_episode(content, {})
            content, meta = self.compressor.compress_semantic(content, {})
        elif target_resolution == ResolutionLevel.LONG_TERM:
            if memory.resolution <= ResolutionLevel.EPISODE:
                content, _ = self.compressor.compress_semantic(content, {})
            content, meta = self.compressor.compress_long_term(content, {})
        elif target_resolution == ResolutionLevel.DEEP_LONG_TERM:
            if memory.resolution <= ResolutionLevel.SEMANTIC:
                content, _ = self.compressor.compress_long_term(content, {})
            content, meta = self.compressor.compress_long_term(content, {})
        else:
            content, meta = self.compressor.compress_light(content, {})

        # Create compressed version
        from artificial_memory.core.models import MemoryVersion
        version = MemoryVersion(
            memory_id=memory.id,
            resolution=target_resolution,
            content=content,
            compression_ratio=meta.get('compression_ratio'),
            created_at=datetime.now(),
            source='auto_consolidation',
        )
        self.store.add_memory_version(version)

        # Log compression event
        event = CompressionEvent(
            source_memory_id=memory.id,
            target_memory_id=None,
            from_resolution=memory.resolution,
            to_resolution=target_resolution,
            original_tokens=meta.get('original_tokens', self.compressor.count_tokens(memory.content)),
            compressed_tokens=meta.get('compressed_tokens', self.compressor.count_tokens(content)),
            compression_ratio=meta.get('compression_ratio', 1.0),
            method=method,
            created_at=datetime.now(),
        )
        self.store.log_compression(event)

        # Update memory
        memory.content = content
        memory.resolution = target_resolution
        memory.status = self._resolution_to_status(target_resolution)
        memory.updated_at = datetime.now()

        return memory

    def _resolution_to_status(self, resolution: ResolutionLevel) -> MemoryStatus:
        """Map resolution to appropriate status."""
        mapping = {
            ResolutionLevel.RAW: MemoryStatus.ACTIVE,
            ResolutionLevel.LIGHT: MemoryStatus.DORMANT,
            ResolutionLevel.EPISODE: MemoryStatus.COMPRESSED,
            ResolutionLevel.SEMANTIC: MemoryStatus.COMPRESSED,
            ResolutionLevel.LONG_TERM: MemoryStatus.ARCHIVED,
            ResolutionLevel.DEEP_LONG_TERM: MemoryStatus.DEEP_ARCHIVED,
        }
        return mapping.get(resolution, MemoryStatus.ACTIVE)

    def consolidate_topic(self, topic_id: int) -> dict[str, int]:
        """Run full consolidation for a topic."""
        stats = {
            "evaluated": 0,
            "status_changed": 0,
            "compressed": 0,
            "archived": 0,
            "deep_archived": 0,
        }

        # Get all memories for topic (not just active)
        memories = self.store.get_memories(topic_id=topic_id, limit=1000)

        for memory in memories:
            stats["evaluated"] += 1

            # Evaluate target status
            target_status = self.evaluate_memory(memory)

            if target_status != memory.status:
                stats["status_changed"] += 1

                # Determine target resolution
                target_resolution = self.get_target_resolution(memory, target_status)

                # Compress if target resolution represents more compression (higher enum value)
                if target_resolution.value > memory.resolution.value:
                    # Need to compress
                    memory = self.compress_memory(memory, target_resolution)
                    stats["compressed"] += 1
                else:
                    # Just status change
                    memory.status = target_status
                    memory.updated_at = datetime.now()

                self.store.update_memory(memory)

                if target_status == MemoryStatus.ARCHIVED:
                    stats["archived"] += 1
                elif target_status == MemoryStatus.DEEP_ARCHIVED:
                    stats["deep_archived"] += 1

        return stats

    def run_consolidation_cycle(self) -> dict[str, int]:
        """Run consolidation across all topics."""
        topics = self.store.list_topics()
        total_stats = {
            "topics_processed": 0,
            "evaluated": 0,
            "status_changed": 0,
            "compressed": 0,
            "archived": 0,
            "deep_archived": 0,
        }

        for topic in topics:
            stats = self.consolidate_topic(topic.id)
            total_stats["topics_processed"] += 1
            for k in ["evaluated", "status_changed", "compressed", "archived", "deep_archived"]:
                total_stats[k] += stats.get(k, 0)

        return total_stats

    def re_activate_memory(self, memory_id: int) -> Memory | None:
        """Reactivate a memory (e.g., after recall)."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            return None

        memory.status = MemoryStatus.ACTIVE
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        memory.updated_at = datetime.now()

        # If deeply archived, restore to higher resolution
        if memory.resolution >= ResolutionLevel.LONG_TERM:
            # Restore to semantic level
            version = self.store.get_memory_version(memory_id, ResolutionLevel.SEMANTIC)
            if version:
                memory.content = version.content
                memory.resolution = ResolutionLevel.SEMANTIC

        self.store.update_memory(memory)
        return memory


class ConsolidationScheduler:
    """Scheduler for running consolidation periodically."""

    def __init__(self, engine: ConsolidationEngine):
        self.engine = engine
        self.last_run: datetime | None = None
        self.interval_hours: int = 24

    def should_run(self) -> bool:
        """Check if consolidation should run."""
        if self.last_run is None:
            return True
        return datetime.now() - self.last_run > timedelta(hours=self.interval_hours)

    def run(self) -> dict[str, int]:
        """Run consolidation cycle."""
        stats = self.engine.run_consolidation_cycle()
        self.last_run = datetime.now()
        return stats

    def force_run(self) -> dict[str, int]:
        """Force run consolidation regardless of schedule."""
        return self.run()


def create_consolidation_engine(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
    config: ConsolidationConfig | None = None,
) -> ConsolidationEngine:
    """Factory function to create consolidation engine."""
    return ConsolidationEngine(store, compressor, config)


def create_consolidation_scheduler(engine: ConsolidationEngine) -> ConsolidationScheduler:
    """Factory function to create consolidation scheduler."""
    return ConsolidationScheduler(engine)
