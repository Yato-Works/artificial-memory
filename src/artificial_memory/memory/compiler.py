from __future__ import annotations

from datetime import datetime

from artificial_memory.compiler import create_compiler_pipeline
from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.adapters import create_memory_ir_adapter
from artificial_memory.core.interfaces import MemoryCompiler, MemoryStore
from artificial_memory.core.ir import MemoryIR
from artificial_memory.core.models import (
    CompressionEvent,
    CompressionMethod,
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Message,
    ResolutionLevel,
)
from artificial_memory.topic.classifier import RuleBasedTopicClassifier


class IncrementalMemoryCompiler:
    """Incremental compiler that processes messages into memories (legacy)."""

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
        topic_classifier: RuleBasedTopicClassifier,
    ):
        self.store = store
        self.compressor = compressor
        self.topic_classifier = topic_classifier

    def process_message(
        self,
        message: Message,
        conversation: Conversation,
        current_memories: list[Memory],
    ) -> list[Memory]:
        """Process a single message and update memories incrementally."""
        new_memories = []

        # For MVP: Extract key information from assistant messages
        if message.role.value == "assistant":
            # Check for decisions
            if self._contains_decision(message.content):
                decision_memory = self._create_decision_memory(message, conversation)
                if decision_memory:
                    new_memories.append(decision_memory)

            # Check for important statements
            if self._contains_important_info(message.content):
                semantic_memory = self._create_semantic_memory(message, conversation)
                if semantic_memory:
                    new_memories.append(semantic_memory)

        # Save new memories
        for mem in new_memories:
            self.store.create_memory(mem)
            # Create light compression version
            self._create_compressed_versions(mem)

        return new_memories

    def _contains_decision(self, content: str) -> bool:
        """Check if content contains a decision."""
        decision_markers = [
            '決めた', '決定', '採用', '却下', '選択', '選んだ', '決断',
            'decided', 'adopted', 'rejected', 'selected', 'chose'
        ]
        content_lower = content.lower()
        return any(marker in content_lower for marker in decision_markers)

    def _contains_important_info(self, content: str) -> bool:
        """Check if content contains important information."""
        important_markers = [
            '重要', '大事', 'クリティカル', '必須', 'マスト',
            '理由', 'なぜなら', 'なぜ', '懸念', 'リスク',
            'アーキテクチャ', '設計', '方針',
            'important', 'critical', 'must', 'reason', 'architecture', 'design'
        ]
        content_lower = content.lower()
        return any(marker in content_lower for marker in important_markers)

    def _create_decision_memory(self, message: Message, conversation: Conversation) -> Memory | None:
        """Create a decision memory from message."""
        lines = message.content.split('\n')
        decision_lines = [line for line in lines if any(m in line.lower() for m in ['決めた', '決定', '採用', '却下', '選択'])]
        decision_text = '\n'.join(decision_lines) if decision_lines else message.content[:500]

        return Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.DECISION,
            content=decision_text,
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.85,
            confidence=0.8,
            status=MemoryStatus.ACTIVE,
            valid_from=datetime.now(),
            is_current=True,
            source_conversation_id=conversation.id,
            source_message_id=message.id,
        )

    def _create_semantic_memory(self, message: Message, conversation: Conversation) -> Memory | None:
        """Create a semantic memory from message."""
        return Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.SEMANTIC,
            content=message.content[:1000],
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.6,
            confidence=0.7,
            status=MemoryStatus.ACTIVE,
            valid_from=datetime.now(),
            is_current=True,
            source_conversation_id=conversation.id,
            source_message_id=message.id,
        )

    def _create_compressed_versions(self, memory: Memory) -> None:
        """Create compressed versions of a memory."""
        self.compressor.count_tokens(memory.content)

        # Light compression
        light_content, light_meta = self.compressor.compress_light(memory.content, {})
        light_version = MemoryVersion(
            memory_id=memory.id,
            resolution=ResolutionLevel.LIGHT,
            content=light_content,
            compression_ratio=light_meta.get('compression_ratio'),
            created_at=datetime.now(),
            source='auto',
        )
        self.store.add_memory_version(light_version)

        # Episode compression
        episode_content, episode_meta = self.compressor.compress_episode(memory.content, {})
        episode_version = MemoryVersion(
            memory_id=memory.id,
            resolution=ResolutionLevel.EPISODE,
            content=episode_content,
            compression_ratio=episode_meta.get('compression_ratio'),
            created_at=datetime.now(),
            source='auto',
        )
        self.store.add_memory_version(episode_version)

        # Semantic compression
        semantic_content, semantic_meta = self.compressor.compress_semantic(memory.content, {})
        semantic_version = MemoryVersion(
            memory_id=memory.id,
            resolution=ResolutionLevel.SEMANTIC,
            content=semantic_content,
            compression_ratio=semantic_meta.get('compression_ratio'),
            created_at=datetime.now(),
            source='auto',
        )
        self.store.add_memory_version(semantic_version)

        # Long-term compression
        longterm_content, longterm_meta = self.compressor.compress_long_term(memory.content, {})
        longterm_version = MemoryVersion(
            memory_id=memory.id,
            resolution=ResolutionLevel.LONG_TERM,
            content=longterm_content,
            compression_ratio=longterm_meta.get('compression_ratio'),
            created_at=datetime.now(),
            source='auto',
        )
        self.store.add_memory_version(longterm_version)

        # Log compression events
        for method, meta in [
            (CompressionMethod.LIGHT, light_meta),
            (CompressionMethod.EPISODE, episode_meta),
            (CompressionMethod.SEMANTIC, semantic_meta),
            (CompressionMethod.LONG_TERM, longterm_meta),
        ]:
            event = CompressionEvent(
                source_memory_id=memory.id,
                target_memory_id=None,
                from_resolution=ResolutionLevel.RAW,
                to_resolution=ResolutionLevel[method.name.upper()],
                original_tokens=meta['original_tokens'],
                compressed_tokens=meta['compressed_tokens'],
                compression_ratio=meta['compression_ratio'],
                method=method,
                created_at=datetime.now(),
            )
            self.store.log_compression(event)

    def compile_conversation(self, conversation: Conversation) -> list[Memory]:
        """Compile entire conversation into memories (batch mode - legacy)."""
        messages = self.store.get_messages(conversation.id)
        new_memories = []

        # Create timeline entries
        timeline_memory = self._create_timeline_memory(conversation, messages)
        if timeline_memory:
            self.store.create_memory(timeline_memory)
            new_memories.append(timeline_memory)

        # Create light compression of full conversation
        full_content = '\n'.join([f"{m.role.value}: {m.content}" for m in messages])
        light_content, light_meta = self.compressor.compress_light(full_content, {})

        light_memory = Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.EPISODE,
            content=light_content,
            resolution=ResolutionLevel.LIGHT,
            importance=0.7,
            confidence=0.75,
            status=MemoryStatus.ACTIVE,
            valid_from=conversation.started_at,
            is_current=True,
            source_conversation_id=conversation.id,
        )
        self.store.create_memory(light_memory)
        new_memories.append(light_memory)

        # Create episode summary
        episode_content, episode_meta = self.compressor.compress_episode(full_content, {})
        episode_memory = Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.EPISODE,
            content=episode_content,
            resolution=ResolutionLevel.EPISODE,
            importance=0.75,
            confidence=0.8,
            status=MemoryStatus.ACTIVE,
            valid_from=conversation.started_at,
            is_current=True,
            source_conversation_id=conversation.id,
        )
        self.store.create_memory(episode_memory)
        new_memories.append(episode_memory)

        # Create semantic summary
        semantic_content, semantic_meta = self.compressor.compress_semantic(full_content, {})
        semantic_memory = Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.SEMANTIC,
            content=semantic_content,
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.8,
            confidence=0.85,
            status=MemoryStatus.ACTIVE,
            valid_from=conversation.started_at,
            is_current=True,
            source_conversation_id=conversation.id,
        )
        self.store.create_memory(semantic_memory)
        new_memories.append(semantic_memory)

        # Create compressed versions for each
        for mem in new_memories:
            self._create_compressed_versions(mem)

        return new_memories

    def _create_timeline_memory(self, conversation: Conversation, messages: list[Message]) -> Memory | None:
        """Create timeline entry from conversation."""
        if not messages:
            return None

        user_messages = [m for m in messages if m.role.value == "user"]
        assistant_messages = [m for m in messages if m.role.value == "assistant"]

        first_user = user_messages[0].content[:200] if user_messages else ""

        events = []
        for msg in assistant_messages:
            if self._contains_decision(msg.content):
                events.append(f"決定: {msg.content[:100]}")
            elif self._contains_important_info(msg.content):
                events.append(f"重要: {msg.content[:100]}")

        timeline_content = f"**{conversation.started_at.strftime('%Y-%m-%d')}**\n"
        timeline_content += f"トピック: {first_user}\n"
        if events:
            timeline_content += "\n".join(f"- {e}" for e in events[:5])

        return Memory(
            topic_id=conversation.topic_id,
            memory_type=MemoryType.TIMELINE,
            content=timeline_content,
            resolution=ResolutionLevel.EPISODE,
            importance=0.7,
            confidence=0.75,
            status=MemoryStatus.ACTIVE,
            valid_from=conversation.started_at,
            is_current=True,
            source_conversation_id=conversation.id,
        )

    def consolidate_topic(self, topic_id: int) -> list[Memory]:
        """Run consolidation for a topic - compress old memories."""
        cutoff = datetime.now() - __import__('datetime').timedelta(days=30)
        memories = self.store.get_memories(
            topic_id=topic_id,
            status=MemoryStatus.ACTIVE,
            limit=100
        )

        consolidated = []
        for memory in memories:
            if memory.created_at < cutoff and memory.resolution == ResolutionLevel.RAW:
                light_content, _ = self.compressor.compress_light(memory.content, {})
                memory.content = light_content
                memory.resolution = ResolutionLevel.LIGHT
                memory.status = MemoryStatus.COMPRESSED
                self.store.update_memory(memory)

                event = CompressionEvent(
                    source_memory_id=memory.id,
                    target_memory_id=None,
                    from_resolution=ResolutionLevel.RAW,
                    to_resolution=ResolutionLevel.LIGHT,
                    original_tokens=self.compressor.count_tokens(memory.content),
                    compressed_tokens=self.compressor.count_tokens(light_content),
                    compression_ratio=self.compressor.get_compression_ratio(memory.content, light_content),
                    method=CompressionMethod.LIGHT,
                    created_at=datetime.now(),
                )
                self.store.log_compression(event)
                consolidated.append(memory)

        return consolidated

    def run_consolidation_cycle(self) -> dict[str, int]:
        """Run consolidation across all topics."""
        topics = self.store.list_topics()
        results = {"topics_processed": 0, "memories_compressed": 0}

        for topic in topics:
            consolidated = self.consolidate_topic(topic.id)
            results["topics_processed"] += 1
            results["memories_compressed"] += len(consolidated)

        return results


class CompilerPipelineWrapper:
    """Wrapper around CompilerPipeline that stores results to database."""

    def __init__(
        self,
        store: MemoryStore,
        compressor: RuleBasedCompressor,
        topic_classifier: RuleBasedTopicClassifier,
    ):
        self.store = store
        self.compressor = compressor
        self.topic_classifier = topic_classifier
        self.pipeline = create_compiler_pipeline()
        self.ir_adapter = create_memory_ir_adapter(store)

    def compile_conversation(self, conversation: Conversation) -> list[Memory]:
        """Compile conversation using new pipeline, store as legacy Memory."""
        messages = self.store.get_messages(conversation.id)

        # Run new pipeline to get MemoryIR
        memory_irs = self.pipeline.compile(conversation, messages)

        # Convert and store as legacy Memory
        stored_memories = []
        for mem_ir in memory_irs:
            legacy_mem, versions, associations, compressions = self.ir_adapter.to_legacy(mem_ir)
            # Ensure IDs are set by store
            legacy_mem.id = None
            stored = self.store.create_memory(legacy_mem)

            # Store versions
            for ver in versions:
                ver.memory_id = stored.id
                self.store.add_memory_version(ver)

            # Store associations
            for assoc in associations:
                assoc.source_memory_id = stored.id
                self.store.create_association(assoc)

            # Store compression events
            for comp in compressions:
                comp.source_memory_id = stored.id
                self.store.log_compression(comp)

            stored_memories.append(stored)

        return stored_memories

    def compile_conversation_ir(self, conversation: Conversation) -> list[MemoryIR]:
        """Compile conversation and return MemoryIR (for new code paths)."""
        messages = self.store.get_messages(conversation.id)
        return self.pipeline.compile(conversation, messages)

    def compile_with_diagnostics(self, conversation: Conversation) -> tuple[list[MemoryIR], list]:
        """Compile with full diagnostics."""
        messages = self.store.get_messages(conversation.id)
        return self.pipeline.compile_with_diagnostics(conversation, messages)

    def process_message(self, message: Message, conversation: Conversation,
                       current_memories: list[Memory]) -> list[Memory]:
        """Incremental processing - delegates to legacy compiler for now."""
        # For incremental, use legacy approach
        legacy = IncrementalMemoryCompiler(self.store, self.compressor, self.topic_classifier)
        return legacy.process_message(message, conversation, current_memories)

    def consolidate_topic(self, topic_id: int) -> list[Memory]:
        """Delegates to legacy compiler."""
        legacy = IncrementalMemoryCompiler(self.store, self.compressor, self.topic_classifier)
        return legacy.consolidate_topic(topic_id)

    def run_consolidation_cycle(self) -> dict[str, int]:
        """Delegates to legacy compiler."""
        legacy = IncrementalMemoryCompiler(self.store, self.compressor, self.topic_classifier)
        return legacy.run_consolidation_cycle()


def create_memory_compiler(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
    topic_classifier: RuleBasedTopicClassifier,
) -> MemoryCompiler:
    """Factory function - returns legacy compiler for backward compatibility."""
    return IncrementalMemoryCompiler(store, compressor, topic_classifier)


def create_compiler_pipeline_wrapper(
    store: MemoryStore,
    compressor: RuleBasedCompressor,
    topic_classifier: RuleBasedTopicClassifier,
) -> CompilerPipelineWrapper:
    """Factory function for new pipeline-based compiler."""
    return CompilerPipelineWrapper(store, compressor, topic_classifier)
