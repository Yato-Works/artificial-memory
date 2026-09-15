from __future__ import annotations

from datetime import datetime

import tiktoken

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    Project,
    Topic,
)


class ConversationLogger:
    """Handles logging conversations and messages to the memory store."""

    def __init__(self, store: MemoryStore, encoding_name: str = "cl100k_base"):
        self.store = store
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._current_conversation: Conversation | None = None
        self._message_sequence = 0

    def load_conversation(self, conversation: Conversation) -> None:
        """Load an existing conversation."""
        self._current_conversation = conversation
        self._message_sequence = conversation.message_count

    def start_conversation(
        self,
        topic: Topic,
        title: str | None = None,
        metadata: dict | None = None
    ) -> Conversation:
        """Start a new conversation session."""
        conversation = Conversation(
            topic_id=topic.id,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            started_at=datetime.now(),
            metadata=metadata or {},
            status=ConversationStatus.ACTIVE,
        )
        self._current_conversation = self.store.create_conversation(conversation)
        self._message_sequence = 0
        return self._current_conversation

    def log_message(
        self,
        role: MessageRole,
        content: str,
        metadata: dict | None = None
    ) -> Message:
        """Log a message to the current conversation."""
        if not self._current_conversation:
            raise RuntimeError("No active conversation. Call start_conversation() first.")

        tokens = len(self.encoding.encode(content))
        self._message_sequence += 1

        message = Message(
            conversation_id=self._current_conversation.id,
            role=role,
            content=content,
            token_count=tokens,
            sequence_num=self._message_sequence,
            metadata=metadata or {},
        )

        saved_message = self.store.add_message(message)

        # Update conversation token count
        self._current_conversation.token_count += tokens
        self._current_conversation.message_count = self._message_sequence
        self.store.update_conversation(self._current_conversation)

        return saved_message

    def log_user_message(self, content: str, metadata: dict | None = None) -> Message:
        """Log a user message."""
        return self.log_message(MessageRole.USER, content, metadata)

    def log_assistant_message(self, content: str, metadata: dict | None = None) -> Message:
        """Log an assistant message."""
        return self.log_message(MessageRole.ASSISTANT, content, metadata)

    def log_system_message(self, content: str, metadata: dict | None = None) -> Message:
        """Log a system message."""
        return self.log_message(MessageRole.SYSTEM, content, metadata)

    def end_conversation(self) -> Conversation:
        """End the current conversation."""
        if not self._current_conversation:
            raise RuntimeError("No active conversation to end.")

        ended = self.store.end_conversation(self._current_conversation.id)
        self._current_conversation = None
        self._message_sequence = 0
        return ended

    def get_current_conversation(self) -> Conversation | None:
        return self._current_conversation

    def get_conversation_history(self, limit: int | None = None) -> list[Message]:
        """Get messages from the current conversation."""
        if not self._current_conversation:
            return []
        return self.store.get_messages(self._current_conversation.id, limit=limit)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    @property
    def current_conversation_id(self) -> int | None:
        return self._current_conversation.id if self._current_conversation else None


class ConversationManager:
    """Higher-level manager for conversations with topic handling."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.logger = ConversationLogger(store)
        self._ensure_default_project()

    def _ensure_default_project(self) -> Project:
        project = self.store.get_project_by_name("default")
        if not project:
            project = Project(name="default", display_name="Default Project")
            project = self.store.create_project(project)
        return project

    def get_or_create_topic(self, path: str, project_name: str = "default") -> Topic:
        """Get existing topic or create new one from path like 'Projects/Artificial-Memory/Architecture'."""
        project = self.store.get_project_by_name(project_name)
        if not project:
            project = Project(name=project_name, display_name=project_name)
            project = self.store.create_project(project)

        topic = self.store.get_topic_by_path(path)
        if topic:
            return topic

        # Create topic hierarchy
        parts = path.strip("/").split("/")
        parent_id = None
        current_path = ""

        for part in parts:
            current_path = f"{current_path}/{part}" if current_path else part
            topic = self.store.get_topic_by_path(current_path)
            if not topic:
                topic = Topic(
                    project_id=project.id,
                    parent_id=parent_id,
                    name=part,
                    path=current_path,
                )
                topic = self.store.create_topic(topic)
            parent_id = topic.id

        return topic

    def start_conversation(
        self,
        topic_path: str,
        project_name: str = "default",
        title: str | None = None,
        metadata: dict | None = None
    ) -> Conversation:
        topic = self.get_or_create_topic(topic_path, project_name)
        return self.logger.start_conversation(topic, title, metadata)

    def log_user(self, content: str, metadata: dict | None = None) -> Message:
        return self.logger.log_user_message(content, metadata)

    def log_assistant(self, content: str, metadata: dict | None = None) -> Message:
        return self.logger.log_assistant_message(content, metadata)

    def end_conversation(self) -> Conversation:
        return self.logger.end_conversation()

    @property
    def current_conversation(self) -> Conversation | None:
        return self.logger.get_current_conversation()
