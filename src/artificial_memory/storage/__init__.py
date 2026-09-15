# Storage module init
from artificial_memory.storage.conversation_logger import ConversationManager
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore

# Optional PostgreSQL support
try:
    from artificial_memory.storage.postgres_store import (
        PostgresConfig,
        PostgresMemoryStore,
        create_postgres_store,
    )
    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False
    PostgresMemoryStore = None  # type: ignore
    PostgresConfig = None  # type: ignore
    create_postgres_store = None  # type: ignore

__all__ = [
    "SQLiteMemoryStore",
    "ConversationManager",
]

if _HAS_POSTGRES:
    __all__.extend([
        "PostgresMemoryStore",
        "PostgresConfig",
        "create_postgres_store",
    ])
