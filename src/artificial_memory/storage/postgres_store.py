from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from artificial_memory.core.models import (
    Association,
    AssociationType,
    CompressionEvent,
    CompressionMethod,
    ContextIR,
    Conversation,
    ConversationStatus,
    Decision,
    IRType,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Message,
    MessageRole,
    Project,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
    TokenUsage,
    Topic,
)


@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "artificial_memory"
    user: str = "postgres"
    password: str = "postgres"
    min_connections: int = 2
    max_connections: int = 10

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgresMemoryStore:
    """PostgreSQL/pgvector implementation of MemoryStore."""

    def __init__(self, config: PostgresConfig | str, pool: ConnectionPool | None = None):
        if isinstance(config, str):
            self.config = PostgresConfig()
            self.config.dsn = config
        else:
            self.config = config

        if pool is not None:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = ConnectionPool(
                self.config.dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                kwargs={"row_factory": dict_row},
            )
            self._owns_pool = True

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if needed."""
        with self._pool.connection() as conn:
            # Check if tables exist
            cursor = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'projects')"
            )
            if not cursor.fetchone()[0]:
                self._apply_schema()

            # Register pgvector adapter
            register_vector(conn)

    def _apply_schema(self) -> None:
        """Apply the PostgreSQL schema."""
        schema_path = Path(__file__).parent.parent.parent.parent / "scripts" / "schema_postgres.sql"
        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()

            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute(schema_sql)
        else:
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

    @contextmanager
    def _transaction(self):
        """Transaction context manager."""
        with self._pool.connection() as conn:
            with conn.transaction():
                yield conn

    def _execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query and return cursor."""
        with self._pool.connection() as conn:
            return conn.execute(query, params)

    # Row conversion methods
    def _row_to_project(self, row: dict) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            display_name=row["display_name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=row["is_active"],
        )

    def _row_to_topic(self, row: dict) -> Topic:
        return Topic(
            id=row["id"],
            project_id=row["project_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            path=row["path"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_conversation(self, row: dict) -> Conversation:
        metadata = {}
        if row["metadata_json"]:
            metadata = row["metadata_json"] if isinstance(row["metadata_json"], dict) else json.loads(row["metadata_json"])
        return Conversation(
            id=row["id"],
            topic_id=row["topic_id"],
            title=row["title"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            message_count=row["message_count"],
            token_count=row["token_count"],
            metadata=metadata,
            status=ConversationStatus(row["status"]),
        )

    def _row_to_message(self, row: dict) -> Message:
        metadata = {}
        if row["metadata_json"]:
            metadata = row["metadata_json"] if isinstance(row["metadata_json"], dict) else json.loads(row["metadata_json"])
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            token_count=row["token_count"],
            sequence_num=row["sequence_num"],
            created_at=row["created_at"],
            metadata=metadata,
        )

    def _row_to_memory(self, row: dict) -> Memory:
        return Memory(
            id=row["id"],
            topic_id=row["topic_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            resolution=ResolutionLevel(row["resolution"]),
            importance=row["importance"],
            confidence=row["confidence"],
            status=MemoryStatus(row["status"]),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            is_current=row["is_current"],
            source_conversation_id=row["source_conversation_id"],
            source_message_id=row["source_message_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
        )

    def _row_to_memory_version(self, row: dict) -> MemoryVersion:
        return MemoryVersion(
            id=row["id"],
            memory_id=row["memory_id"],
            resolution=ResolutionLevel(row["resolution"]),
            content=row["content"],
            compression_ratio=row["compression_ratio"],
            created_at=row["created_at"],
            source=row["source"],
        )

    def _row_to_decision(self, row: dict) -> Decision:
        rejected = []
        if row["rejected_options"]:
            rejected = row["rejected_options"] if isinstance(row["rejected_options"], list) else json.loads(row["rejected_options"])
        return Decision(
            id=row["id"],
            topic_id=row["topic_id"],
            memory_id=row["memory_id"],
            decision_text=row["decision_text"],
            rejected_options=rejected,
            reason=row["reason"],
            confidence=row["confidence"],
            decided_at=row["decided_at"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            is_current=row["is_current"],
        )

    def _row_to_association(self, row: dict) -> Association:
        return Association(
            id=row["id"],
            source_memory_id=row["source_memory_id"],
            target_memory_id=row["target_memory_id"],
            association_type=AssociationType(row["association_type"]),
            strength=row["strength"],
            created_at=row["created_at"],
        )

    def _row_to_recall(self, row: dict) -> RecallEvent:
        return RecallEvent(
            id=row["id"],
            query=row["query"],
            topic_id=row["topic_id"],
            recall_level=RecallLevel(row["recall_level"]),
            memories_retrieved=row["memories_retrieved"],
            tokens_returned=row["tokens_returned"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
        )

    def _row_to_compression(self, row: dict) -> CompressionEvent:
        return CompressionEvent(
            id=row["id"],
            source_memory_id=row["source_memory_id"],
            target_memory_id=row["target_memory_id"],
            from_resolution=ResolutionLevel(row["from_resolution"]),
            to_resolution=ResolutionLevel(row["to_resolution"]),
            original_tokens=row["original_tokens"],
            compressed_tokens=row["compressed_tokens"],
            compression_ratio=row["compression_ratio"],
            method=CompressionMethod(row["method"]),
            created_at=row["created_at"],
        )

    def _row_to_context_ir(self, row: dict) -> ContextIR:
        return ContextIR(
            id=row["id"],
            conversation_id=row["conversation_id"],
            ir_type=IRType(row["ir_type"]),
            ir_key=row["ir_key"],
            ir_value=row["ir_value"],
            sequence_num=row["sequence_num"],
            created_at=row["created_at"],
        )

    def _row_to_token_usage(self, row: dict) -> TokenUsage:
        return TokenUsage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            operation=row["operation"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            model=row["model"],
            created_at=row["created_at"],
        )

    # Project operations
    def create_project(self, project: Project) -> Project:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO projects (name, display_name, description, created_at, updated_at, is_active)
                   VALUES (%(name)s, %(display_name)s, %(description)s, %(created_at)s, %(updated_at)s, %(is_active)s)
                   RETURNING id""",
                {
                    "name": project.name,
                    "display_name": project.display_name,
                    "description": project.description,
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                    "is_active": project.is_active,
                }
            )
            project.id = cursor.fetchone()["id"]
        return project

    def get_project(self, project_id: int) -> Project | None:
        row = self._execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        return self._row_to_project(row) if row else None

    def get_project_by_name(self, name: str) -> Project | None:
        row = self._execute("SELECT * FROM projects WHERE name = %s", (name,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self, active_only: bool = True) -> list[Project]:
        query = "SELECT * FROM projects"
        if active_only:
            query += " WHERE is_active = TRUE"
        query += " ORDER BY name"
        rows = self._execute(query).fetchall()
        return [self._row_to_project(r) for r in rows]

    def update_project(self, project: Project) -> Project:
        project.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE projects SET display_name = %s, description = %s, updated_at = %s, is_active = %s
                   WHERE id = %s""",
                (project.display_name, project.description, project.updated_at, project.is_active, project.id)
            )
        return project

    # Topic operations
    def create_topic(self, topic: Topic) -> Topic:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO topics (project_id, parent_id, name, path, description, created_at, updated_at)
                   VALUES (%(project_id)s, %(parent_id)s, %(name)s, %(path)s, %(description)s, %(created_at)s, %(updated_at)s)
                   RETURNING id""",
                {
                    "project_id": topic.project_id,
                    "parent_id": topic.parent_id,
                    "name": topic.name,
                    "path": topic.path,
                    "description": topic.description,
                    "created_at": topic.created_at,
                    "updated_at": topic.updated_at,
                }
            )
            topic.id = cursor.fetchone()["id"]
        return topic

    def get_topic(self, topic_id: int) -> Topic | None:
        row = self._execute("SELECT * FROM topics WHERE id = %s", (topic_id,)).fetchone()
        return self._row_to_topic(row) if row else None

    def get_topic_by_path(self, path: str) -> Topic | None:
        row = self._execute("SELECT * FROM topics WHERE path = %s", (path,)).fetchone()
        return self._row_to_topic(row) if row else None

    def list_topics(self, project_id: int | None = None, parent_id: int | None = None) -> list[Topic]:
        query = "SELECT * FROM topics WHERE 1=1"
        params = []
        if project_id is not None:
            query += " AND project_id = %s"
            params.append(project_id)
        if parent_id is not None:
            query += " AND parent_id = %s"
            params.append(parent_id)
        query += " ORDER BY path"
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_topic(r) for r in rows]

    def update_topic(self, topic: Topic) -> Topic:
        topic.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE topics SET parent_id = %s, name = %s, path = %s, description = %s, updated_at = %s
                   WHERE id = %s""",
                (topic.parent_id, topic.name, topic.path, topic.description, topic.updated_at, topic.id)
            )
        return topic

    # Conversation operations
    def create_conversation(self, conversation: Conversation) -> Conversation:
        metadata_json = json.dumps(conversation.metadata) if conversation.metadata else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO conversations (topic_id, title, started_at, ended_at, message_count, token_count, metadata_json, status)
                   VALUES (%(topic_id)s, %(title)s, %(started_at)s, %(ended_at)s, %(message_count)s, %(token_count)s, %(metadata_json)s, %(status)s)
                   RETURNING id""",
                {
                    "topic_id": conversation.topic_id,
                    "title": conversation.title,
                    "started_at": conversation.started_at,
                    "ended_at": conversation.ended_at,
                    "message_count": conversation.message_count,
                    "token_count": conversation.token_count,
                    "metadata_json": metadata_json,
                    "status": conversation.status.value,
                }
            )
            conversation.id = cursor.fetchone()["id"]
        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        row = self._execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,)).fetchone()
        return self._row_to_conversation(row) if row else None

    def list_conversations(self, topic_id: int | None = None, status: ConversationStatus | None = None) -> list[Conversation]:
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = %s"
            params.append(topic_id)
        if status is not None:
            query += " AND status = %s"
            params.append(status.value)
        query += " ORDER BY started_at DESC"
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_conversation(r) for r in rows]

    def update_conversation(self, conversation: Conversation) -> Conversation:
        metadata_json = json.dumps(conversation.metadata) if conversation.metadata else None
        with self._transaction() as conn:
            conn.execute(
                """UPDATE conversations SET title = %s, ended_at = %s, message_count = %s, token_count = %s, metadata_json = %s, status = %s
                   WHERE id = %s""",
                (conversation.title, conversation.ended_at, conversation.message_count, conversation.token_count,
                 metadata_json, conversation.status.value, conversation.id)
            )
        return conversation

    def end_conversation(self, conversation_id: int) -> Conversation:
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.ended_at = datetime.now()
            conversation.status = ConversationStatus.COMPLETED
            return self.update_conversation(conversation)
        raise ValueError(f"Conversation {conversation_id} not found")

    # Message operations
    def add_message(self, message: Message) -> Message:
        metadata_json = json.dumps(message.metadata) if message.metadata else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO messages (conversation_id, role, content, token_count, sequence_num, created_at, metadata_json)
                   VALUES (%(conversation_id)s, %(role)s, %(content)s, %(token_count)s, %(sequence_num)s, %(created_at)s, %(metadata_json)s)
                   RETURNING id""",
                {
                    "conversation_id": message.conversation_id,
                    "role": message.role.value,
                    "content": message.content,
                    "token_count": message.token_count,
                    "sequence_num": message.sequence_num,
                    "created_at": message.created_at,
                    "metadata_json": metadata_json,
                }
            )
            message.id = cursor.fetchone()["id"]

            # Update conversation message count
            conn.execute(
                "UPDATE conversations SET message_count = message_count + 1 WHERE id = %s",
                (message.conversation_id,)
            )
        return message

    def get_messages(self, conversation_id: int, limit: int | None = None, offset: int = 0) -> list[Message]:
        query = "SELECT * FROM messages WHERE conversation_id = %s ORDER BY sequence_num"
        params = [conversation_id]
        if limit:
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_message_count(self, conversation_id: int) -> int:
        row = self._execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    # Memory operations
    def create_memory(self, memory: Memory) -> Memory:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO memories (topic_id, memory_type, content, resolution, importance, confidence, status,
                   valid_from, valid_until, is_current, source_conversation_id, source_message_id, created_at, updated_at, last_accessed, access_count, embedding)
                   VALUES (%(topic_id)s, %(memory_type)s, %(content)s, %(resolution)s, %(importance)s, %(confidence)s, %(status)s,
                   %(valid_from)s, %(valid_until)s, %(is_current)s, %(source_conversation_id)s, %(source_message_id)s, %(created_at)s, %(updated_at)s, %(last_accessed)s, %(access_count)s, %(embedding)s)
                   RETURNING id""",
                {
                    "topic_id": memory.topic_id,
                    "memory_type": memory.memory_type.value,
                    "content": memory.content,
                    "resolution": memory.resolution.value,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "status": memory.status.value,
                    "valid_from": memory.valid_from,
                    "valid_until": memory.valid_until,
                    "is_current": memory.is_current,
                    "source_conversation_id": memory.source_conversation_id,
                    "source_message_id": memory.source_message_id,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "last_accessed": memory.last_accessed,
                    "access_count": memory.access_count,
                    "embedding": None,  # Will be updated by vector search engine
                }
            )
            memory.id = cursor.fetchone()["id"]
        return memory

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self._execute("SELECT * FROM memories WHERE id = %s", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def get_memories(
        self,
        topic_id: int | None = None,
        memory_type: MemoryType | None = None,
        resolution: ResolutionLevel | None = None,
        status: MemoryStatus | None = None,
        is_current: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = %s"
            params.append(topic_id)
        if memory_type is not None:
            query += " AND memory_type = %s"
            params.append(memory_type.value)
        if resolution is not None:
            query += " AND resolution = %s"
            params.append(resolution.value)
        if status is not None:
            query += " AND status = %s"
            params.append(status.value)
        if is_current is not None:
            query += " AND is_current = %s"
            params.append(is_current)
        query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_memory(self, memory: Memory) -> Memory:
        memory.updated_at = datetime.now()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET topic_id = %s, memory_type = %s, content = %s, resolution = %s, importance = %s,
                   confidence = %s, status = %s, valid_from = %s, valid_until = %s, is_current = %s,
                   source_conversation_id = %s, source_message_id = %s, updated_at = %s, last_accessed = %s, access_count = %s
                   WHERE id = %s""",
                (memory.topic_id, memory.memory_type.value, memory.content, memory.resolution.value,
                 memory.importance, memory.confidence, memory.status.value,
                 memory.valid_from, memory.valid_until, memory.is_current,
                 memory.source_conversation_id, memory.source_message_id,
                 memory.updated_at, memory.last_accessed, memory.access_count, memory.id)
            )
        return memory

    def delete_memory(self, memory_id: int) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            return cursor.rowcount > 0

    # Memory version operations
    def add_memory_version(self, version: MemoryVersion) -> MemoryVersion:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO memory_versions (memory_id, resolution, content, compression_ratio, created_at, source)
                   VALUES (%(memory_id)s, %(resolution)s, %(content)s, %(compression_ratio)s, %(created_at)s, %(source)s)
                   RETURNING id""",
                {
                    "memory_id": version.memory_id,
                    "resolution": version.resolution.value,
                    "content": version.content,
                    "compression_ratio": version.compression_ratio,
                    "created_at": version.created_at,
                    "source": version.source,
                }
            )
            version.id = cursor.fetchone()["id"]
        return version

    def get_memory_versions(self, memory_id: int) -> list[MemoryVersion]:
        rows = self._execute(
            "SELECT * FROM memory_versions WHERE memory_id = %s ORDER BY resolution", (memory_id,)
        ).fetchall()
        return [self._row_to_memory_version(r) for r in rows]

    def get_memory_version(self, memory_id: int, resolution: ResolutionLevel) -> MemoryVersion | None:
        row = self._execute(
            "SELECT * FROM memory_versions WHERE memory_id = %s AND resolution = %s",
            (memory_id, resolution.value)
        ).fetchone()
        return self._row_to_memory_version(row) if row else None

    # Decision operations
    def create_decision(self, decision: Decision) -> Decision:
        rejected_json = json.dumps(decision.rejected_options) if decision.rejected_options else None
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO decisions (topic_id, memory_id, decision_text, rejected_options, reason, confidence,
                   decided_at, valid_from, valid_until, is_current)
                   VALUES (%(topic_id)s, %(memory_id)s, %(decision_text)s, %(rejected_options)s, %(reason)s, %(confidence)s,
                   %(decided_at)s, %(valid_from)s, %(valid_until)s, %(is_current)s)
                   RETURNING id""",
                {
                    "topic_id": decision.topic_id,
                    "memory_id": decision.memory_id,
                    "decision_text": decision.decision_text,
                    "rejected_options": rejected_json,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "decided_at": decision.decided_at,
                    "valid_from": decision.valid_from,
                    "valid_until": decision.valid_until,
                    "is_current": decision.is_current,
                }
            )
            decision.id = cursor.fetchone()["id"]
        return decision

    def get_decisions(self, topic_id: int, current_only: bool = True) -> list[Decision]:
        query = "SELECT * FROM decisions WHERE topic_id = %s"
        params = [topic_id]
        if current_only:
            query += " AND is_current = TRUE"
        query += " ORDER BY decided_at DESC"
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def update_decision(self, decision: Decision) -> Decision:
        rejected_json = json.dumps(decision.rejected_options) if decision.rejected_options else None
        with self._transaction() as conn:
            conn.execute(
                """UPDATE decisions SET memory_id = %s, decision_text = %s, rejected_options = %s, reason = %s,
                   confidence = %s, valid_from = %s, valid_until = %s, is_current = %s
                   WHERE id = %s""",
                (decision.memory_id, decision.decision_text, rejected_json, decision.reason,
                 decision.confidence, decision.valid_from, decision.valid_until, decision.is_current, decision.id)
            )
        return decision

    # Association operations
    def create_association(self, association: Association) -> Association:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO associations (source_memory_id, target_memory_id, association_type, strength, created_at)
                   VALUES (%(source_memory_id)s, %(target_memory_id)s, %(association_type)s, %(strength)s, %(created_at)s)
                   RETURNING id""",
                {
                    "source_memory_id": association.source_memory_id,
                    "target_memory_id": association.target_memory_id,
                    "association_type": association.association_type.value,
                    "strength": association.strength,
                    "created_at": association.created_at,
                }
            )
            association.id = cursor.fetchone()["id"]
        return association

    def get_associations(self, memory_id: int, association_type: AssociationType | None = None) -> list[Association]:
        query = "SELECT * FROM associations WHERE source_memory_id = %s OR target_memory_id = %s"
        params = [memory_id, memory_id]
        if association_type:
            query += " AND association_type = %s"
            params.append(association_type.value)
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_association(r) for r in rows]

    def get_related_memories(self, memory_id: int, min_strength: float = 0.3) -> list[tuple[Memory, Association]]:
        query = """
            SELECT m.*, a.* FROM memories m
            JOIN associations a ON (a.source_memory_id = m.id OR a.target_memory_id = m.id)
            WHERE (a.source_memory_id = %s OR a.target_memory_id = %s) AND a.strength >= %s AND m.id != %s
        """
        params = [memory_id, memory_id, min_strength, memory_id]
        rows = self._execute(query, tuple(params)).fetchall()
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            assoc = Association(
                id=row["id"],
                source_memory_id=row["source_memory_id"],
                target_memory_id=row["target_memory_id"],
                association_type=AssociationType(row["association_type"]),
                strength=row["strength"],
                created_at=row["created_at"],
            )
            results.append((memory, assoc))
        return results

    # Recall event operations
    def log_recall(self, recall: RecallEvent) -> RecallEvent:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO recalls (query, topic_id, recall_level, memories_retrieved, tokens_returned, latency_ms, created_at)
                   VALUES (%(query)s, %(topic_id)s, %(recall_level)s, %(memories_retrieved)s, %(tokens_returned)s, %(latency_ms)s, %(created_at)s)
                   RETURNING id""",
                {
                    "query": recall.query,
                    "topic_id": recall.topic_id,
                    "recall_level": recall.recall_level.value,
                    "memories_retrieved": recall.memories_retrieved,
                    "tokens_returned": recall.tokens_returned,
                    "latency_ms": recall.latency_ms,
                    "created_at": recall.created_at,
                }
            )
            recall.id = cursor.fetchone()["id"]
        return recall

    def get_recent_recalls(self, topic_id: int | None = None, limit: int = 50) -> list[RecallEvent]:
        query = "SELECT * FROM recalls WHERE 1=1"
        params = []
        if topic_id is not None:
            query += " AND topic_id = %s"
            params.append(topic_id)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_recall(r) for r in rows]

    # Compression event operations
    def log_compression(self, event: CompressionEvent) -> CompressionEvent:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO compression_events (source_memory_id, target_memory_id, from_resolution, to_resolution,
                   original_tokens, compressed_tokens, compression_ratio, method, created_at)
                   VALUES (%(source_memory_id)s, %(target_memory_id)s, %(from_resolution)s, %(to_resolution)s,
                   %(original_tokens)s, %(compressed_tokens)s, %(compression_ratio)s, %(method)s, %(created_at)s)
                   RETURNING id""",
                {
                    "source_memory_id": event.source_memory_id,
                    "target_memory_id": event.target_memory_id,
                    "from_resolution": event.from_resolution.value,
                    "to_resolution": event.to_resolution.value,
                    "original_tokens": event.original_tokens,
                    "compressed_tokens": event.compressed_tokens,
                    "compression_ratio": event.compression_ratio,
                    "method": event.method.value,
                    "created_at": event.created_at,
                }
            )
            event.id = cursor.fetchone()["id"]
        return event

    def get_compression_history(self, memory_id: int) -> list[CompressionEvent]:
        rows = self._execute(
            "SELECT * FROM compression_events WHERE source_memory_id = %s ORDER BY created_at",
            (memory_id,)
        ).fetchall()
        return [self._row_to_compression(r) for r in rows]

    # Context IR operations
    def add_context_ir(self, ir: ContextIR) -> ContextIR:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO context_ir (conversation_id, ir_type, ir_key, ir_value, sequence_num, created_at)
                   VALUES (%(conversation_id)s, %(ir_type)s, %(ir_key)s, %(ir_value)s, %(sequence_num)s, %(created_at)s)
                   RETURNING id""",
                {
                    "conversation_id": ir.conversation_id,
                    "ir_type": ir.ir_type.value,
                    "ir_key": ir.ir_key,
                    "ir_value": ir.ir_value,
                    "sequence_num": ir.sequence_num,
                    "created_at": ir.created_at,
                }
            )
            ir.id = cursor.fetchone()["id"]
        return ir

    def get_context_ir(self, conversation_id: int) -> list[ContextIR]:
        rows = self._execute(
            "SELECT * FROM context_ir WHERE conversation_id = %s ORDER BY sequence_num",
            (conversation_id,)
        ).fetchall()
        return [self._row_to_context_ir(r) for r in rows]

    # Token usage operations
    def log_token_usage(self, usage: TokenUsage) -> TokenUsage:
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO token_usage (conversation_id, operation, input_tokens, output_tokens, total_tokens, model, created_at)
                   VALUES (%(conversation_id)s, %(operation)s, %(input_tokens)s, %(output_tokens)s, %(total_tokens)s, %(model)s, %(created_at)s)
                   RETURNING id""",
                {
                    "conversation_id": usage.conversation_id,
                    "operation": usage.operation,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "model": usage.model,
                    "created_at": usage.created_at,
                }
            )
            usage.id = cursor.fetchone()["id"]
        return usage

    def get_token_usage(self, conversation_id: int | None = None, operation: str | None = None) -> list[TokenUsage]:
        query = "SELECT * FROM token_usage WHERE 1=1"
        params = []
        if conversation_id is not None:
            query += " AND conversation_id = %s"
            params.append(conversation_id)
        if operation is not None:
            query += " AND operation = %s"
            params.append(operation)
        query += " ORDER BY created_at DESC"
        rows = self._execute(query, tuple(params)).fetchall()
        return [self._row_to_token_usage(r) for r in rows]

    # Vector search operations
    def update_memory_embedding(self, memory_id: int, embedding: list[float]) -> bool:
        """Update the vector embedding for a memory."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE memories SET embedding = %s WHERE id = %s",
                (embedding, memory_id)
            )
            return cursor.rowcount > 0

    def vector_search(
        self,
        query_embedding: list[float],
        topic_id: int | None = None,
        memory_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        """Search memories by vector similarity using pgvector."""
        query = """
            SELECT m.*, 1 - (m.embedding <=> %s) as similarity
            FROM memories m
            WHERE m.embedding IS NOT NULL AND 1 - (m.embedding <=> %s) >= %s
        """
        params = [query_embedding, query_embedding, threshold]

        if topic_id is not None:
            query += " AND m.topic_id = %s"
            params.append(topic_id)

        if memory_type is not None:
            query += " AND m.memory_type = %s"
            params.append(memory_type)

        query += " ORDER BY similarity DESC LIMIT %s"
        params.append(k)

        rows = self._execute(query, tuple(params)).fetchall()
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            similarity = row["similarity"]
            results.append((memory, similarity))
        return results

    # Health/management
    def health_check(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._owns_pool and self._pool:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> PostgresMemoryStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_postgres_store(
    host: str = "localhost",
    port: int = 5432,
    database: str = "artificial_memory",
    user: str = "postgres",
    password: str = "postgres",
    min_connections: int = 2,
    max_connections: int = 10,
) -> PostgresMemoryStore:
    """Factory function to create a PostgreSQL memory store."""
    config = PostgresConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        min_connections=min_connections,
        max_connections=max_connections,
    )
    return PostgresMemoryStore(config)
