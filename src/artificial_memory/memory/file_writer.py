from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, MemoryStatus, MemoryType, Topic


class MemoryFileWriter:
    """Writes memory files to disk (current.md, timeline.md, decisions.md, etc.)."""

    def __init__(self, store: MemoryStore, base_path: Path):
        self.store = store
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_topic_dir(self, topic: Topic) -> Path:
        """Get directory for a topic."""
        # Convert path like "Projects/Artificial-Memory" to directory
        safe_path = topic.path.replace("/", "_").replace(" ", "-")
        topic_dir = self.base_path / safe_path
        topic_dir.mkdir(parents=True, exist_ok=True)
        return topic_dir

    def write_current(self, topic: Topic, memories: list[Memory]) -> Path:
        """Write current.md - current state only."""
        topic_dir = self._get_topic_dir(topic)
        current_file = topic_dir / "current.md"

        # Get latest active memory of each type
        current_memories = []
        for mem_type in [MemoryType.CURRENT, MemoryType.DECISION, MemoryType.SEMANTIC, MemoryType.TIMELINE]:
            # Filter active current memories of this type
            type_memories = [m for m in memories
                           if m.memory_type == mem_type
                           and m.is_current
                           and m.status == MemoryStatus.ACTIVE]
            if type_memories:
                # Get the most recently updated
                latest = max(type_memories, key=lambda m: m.updated_at)
                current_memories.append(latest)

        lines = [
            f"# Current State: {topic.name}",
            "",
            f"**Topic:** {topic.path}",
            f"**Updated:** {datetime.now().isoformat()}",
            f"**Memory Count:** {len(current_memories)}",
            "",
            "---",
            "",
        ]

        for memory in current_memories:
            lines.append(f"## {memory.memory_type.value.title()}")
            lines.append("")
            lines.append(memory.content)
            lines.append("")
            lines.append(f"*Resolution: {memory.resolution.name} | Importance: {memory.importance:.2f} | Confidence: {memory.confidence:.2f}*")
            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        current_file.write_text(content, encoding="utf-8")
        return current_file

    def write_timeline(self, topic: Topic, memories: list[Memory]) -> Path:
        """Write timeline.md - how we got to current state."""
        topic_dir = self._get_topic_dir(topic)
        timeline_file = topic_dir / "timeline.md"

        timeline_memories = [m for m in memories if m.memory_type == MemoryType.TIMELINE]
        timeline_memories.sort(key=lambda m: m.valid_from or m.created_at)

        lines = [
            f"# Timeline: {topic.name}",
            "",
            f"**Topic:** {topic.path}",
            f"**Updated:** {datetime.now().isoformat()}",
            f"**Events:** {len(timeline_memories)}",
            "",
            "---",
            "",
        ]

        current_date = None
        for memory in timeline_memories:
            date = memory.valid_from or memory.created_at
            date_str = date.strftime("%Y-%m-%d")

            if date_str != current_date:
                if current_date is not None:
                    lines.append("")
                lines.append(f"## {date_str}")
                lines.append("")
                current_date = date_str

            lines.append(f"- {memory.content}")
            if memory.importance > 0.7:
                lines.append(f"  *⭐ Importance: {memory.importance:.2f}*")
            lines.append("")

        content = "\n".join(lines)
        timeline_file.write_text(content, encoding="utf-8")
        return timeline_file

    def write_decisions(self, topic: Topic, memories: list[Memory]) -> Path:
        """Write decisions.md - important decisions."""
        topic_dir = self._get_topic_dir(topic)
        decisions_file = topic_dir / "decisions.md"

        decision_memories = [m for m in memories if m.memory_type == MemoryType.DECISION and m.is_current]

        lines = [
            f"# Decisions: {topic.name}",
            "",
            f"**Topic:** {topic.path}",
            f"**Updated:** {datetime.now().isoformat()}",
            f"**Decisions:** {len(decision_memories)}",
            "",
            "---",
            "",
        ]

        for memory in decision_memories:
            lines.append(f"## Decision: {memory.content.split(chr(10))[0][:80]}")
            lines.append("")
            lines.append(memory.content)
            lines.append("")
            lines.append(f"*Confidence: {memory.confidence:.2f} | Valid from: {memory.valid_from.strftime('%Y-%m-%d') if memory.valid_from else 'N/A'}*")
            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        decisions_file.write_text(content, encoding="utf-8")
        return decisions_file

    def write_ideas(self, topic: Topic, memories: list[Memory]) -> Path:
        """Write ideas.md - captured ideas and thoughts."""
        topic_dir = self._get_topic_dir(topic)
        ideas_file = topic_dir / "ideas.md"

        # For MVP, we'll use semantic memories as ideas
        idea_memories = [m for m in memories if m.memory_type == MemoryType.SEMANTIC]

        lines = [
            f"# Ideas & Insights: {topic.name}",
            "",
            f"**Topic:** {topic.path}",
            f"**Updated:** {datetime.now().isoformat()}",
            f"**Ideas:** {len(idea_memories)}",
            "",
            "---",
            "",
        ]

        for memory in idea_memories:
            lines.append(f"## {memory.content.split(chr(10))[0][:80]}")
            lines.append("")
            lines.append(memory.content)
            lines.append("")
            lines.append(f"*Resolution: {memory.resolution.name} | Importance: {memory.importance:.2f}*")
            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        ideas_file.write_text(content, encoding="utf-8")
        return ideas_file

    def write_conversation_style(self, topic: Topic, memories: list[Memory]) -> Path:
        """Write conversation_style.md - tone, fillers, patterns."""
        topic_dir = self._get_topic_dir(topic)
        style_file = topic_dir / "conversation_style.md"

        style_memories = [m for m in memories if m.memory_type == MemoryType.CONVERSATION_STYLE]

        lines = [
            f"# Conversation Style: {topic.name}",
            "",
            f"**Topic:** {topic.path}",
            f"**Updated:** {datetime.now().isoformat()}",
            "",
            "---",
            "",
        ]

        for memory in style_memories:
            lines.append(memory.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        style_file.write_text(content, encoding="utf-8")
        return style_file

    def write_all(self, topic: Topic) -> list[Path]:
        """Write all memory files for a topic."""
        memories = self.store.get_memories(topic_id=topic.id)
        results = []

        results.append(self.write_current(topic, memories))
        results.append(self.write_timeline(topic, memories))
        results.append(self.write_decisions(topic, memories))
        results.append(self.write_ideas(topic, memories))
        results.append(self.write_conversation_style(topic, memories))

        return results


class ConversationFileWriter:
    """Writes individual conversation files to disk."""

    def __init__(self, store: MemoryStore, base_path: Path):
        self.store = store
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_conversation(self, conversation_id: int) -> Path:
        """Write a conversation to a dated file."""
        conversation = self.store.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        topic = self.store.get_topic(conversation.topic_id)
        topic_name = topic.name if topic else "unknown"

        date_str = conversation.started_at.strftime("%Y-%m-%d")
        time_str = conversation.started_at.strftime("%H-%M")
        safe_topic = topic_name.replace(" ", "-").replace("/", "-")

        filename = f"{date_str}_{time_str}_{safe_topic}.md"
        file_path = self.base_path / filename

        messages = self.store.get_messages(conversation_id)

        lines = [
            f"# Conversation: {conversation.title or 'Untitled'}",
            "",
            f"**Date:** {conversation.started_at.isoformat()}",
            f"**Topic:** {topic.path if topic else 'Unknown'}",
            f"**Messages:** {len(messages)}",
            f"**Tokens:** {conversation.token_count}",
            f"**Status:** {conversation.status.value}",
            "",
            "---",
            "",
        ]

        for msg in messages:
            role_label = "👤 User" if msg.role.value == "user" else "🤖 Assistant" if msg.role.value == "assistant" else "⚙️ System"
            lines.append(f"## {role_label}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            if msg.metadata:
                meta_lines = [f"  *{k}: {v}*" for k, v in msg.metadata.items()]
                lines.extend(meta_lines)
                lines.append("")
            lines.append("---")
            lines.append("")

        # Add metadata as YAML frontmatter at the end
        lines.append("## Metadata")
        lines.append("")
        lines.append("```yaml")
        meta = {
            "date": conversation.started_at.isoformat(),
            "topic": topic.path if topic else "unknown",
            "tone": conversation.metadata.get("tone", "neutral"),
            "style": conversation.metadata.get("style", "conversational"),
            "importance": conversation.metadata.get("importance", 0.5),
            "resolution": conversation.metadata.get("resolution", 0),
        }
        lines.append(yaml.dump(meta, allow_unicode=True, sort_keys=False))
        lines.append("```")

        content = "\n".join(lines)
        file_path.write_text(content, encoding="utf-8")
        return file_path
