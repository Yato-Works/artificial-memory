from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from artificial_memory.auth.models import UserRole
from artificial_memory.core.models import (
    MemoryStatus,
    MessageRole,
    RecallLevel,
    ResolutionLevel,
)
from artificial_memory.runtime.facade import (
    ArtificialMemoryRuntime,
    RuntimeConfig,
)


class AppState:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        db_path: str = "memory.db",
        memory_files_path: str = "memory_files",
    ):
        if config is None:
            use_postgres = os.environ.get("USE_POSTGRES", "").lower() in ("true", "1", "yes")
            config = RuntimeConfig(
                database_path=os.environ.get("DATABASE_PATH", db_path),
                use_postgres=use_postgres,
                database_url=os.environ.get("DATABASE_URL"),
                memory_files_path=os.environ.get("MEMORY_FILES_PATH", memory_files_path),
                vector_index_path=os.environ.get("VECTOR_INDEX_PATH", "vector_index"),
                ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                jwt_secret=os.environ.get("JWT_SECRET"),
                enable_auth=os.environ.get("ENABLE_AUTH", "").lower() in ("true", "1", "yes"),
            )
        self.runtime = ArtificialMemoryRuntime(config)

        # Delegate sub-components to the runtime facade
        self.store = self.runtime.store
        self.conversation_manager = self.runtime.conversation_manager
        self.topic_classifier = self.runtime.topic_classifier
        self.compressor = self.runtime.compressor
        self.recall_engine = self.runtime.recall_engine
        self.context_builder = self.runtime.context_builder
        self.memory_compiler = self.runtime.memory_compiler
        self.association_engine = self.runtime.association_engine
        self.temporal_engine = self.runtime.temporal_engine
        self.confidence_engine = self.runtime.confidence_engine
        self.style_engine = self.runtime.style_engine
        self.human_recall_engine = self.runtime.human_recall_engine
        self.vector_search_engine = self.runtime.vector_search_engine
        self.llm_manager = self.runtime.llm_manager
        self.auth_service = self.runtime.auth_service
        self.ir_compiler = self.runtime.ir_compiler
        self.consolidation_engine = self.runtime.consolidation_engine
        self.consolidation_scheduler = self.runtime.consolidation_scheduler
        self.metrics_collector = self.runtime.metrics_collector
        self.active_websockets: set[WebSocket] = set()

    def close(self) -> None:
        self.runtime.close()


app_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    app_state = AppState()
    yield
    app_state.close()


app = FastAPI(
    title="Artificial Memory / Context Runtime",
    description="Cognitive Runtime for LLMs - HTTP API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class ConversationStart(BaseModel):
    topic_path: str
    project: str = "default"
    title: str | None = None


class MessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    metadata: dict | None = None


class RecallRequest(BaseModel):
    query: str
    topic_path: str | None = None
    level: int = Field(default=2, ge=0, le=4)
    max_tokens: int = 4000


class ContextRequest(BaseModel):
    query: str
    topic_path: str | None = None
    max_tokens: int = 8000


class ExpandRequest(BaseModel):
    target_resolution: int = Field(..., ge=0, le=5)


class ConsolidationRequest(BaseModel):
    topic_path: str | None = None


class AssociationRequest(BaseModel):
    topic_path: str


class ConfidenceRequest(BaseModel):
    query: str
    topic_path: str | None = None
    level: int = Field(default=2, ge=0, le=4)


class HumanRecallRequest(BaseModel):
    query: str
    topic_path: str | None = None
    max_tokens: int = 4000


class TemporalStateRequest(BaseModel):
    topic_path: str
    timestamp: str | None = None
    include_archived: bool = False


class TemporalChangesRequest(BaseModel):
    topic_path: str
    start: str
    end: str


class StyleProfileRequest(BaseModel):
    topic_path: str | None = None


class StyleReconstructRequest(BaseModel):
    memory_id: int
    target_style: str = Field(default="combined", pattern="^(user|assistant|combined)$")


# Dependency
def get_app_state() -> AppState:
    if app_state is None:
        raise HTTPException(status_code=503, detail="App not initialized")
    return app_state


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


ws_manager = ConnectionManager()


# Routes
@app.get("/")
async def root():
    return {
        "name": "Artificial Memory / Context Runtime",
        "version": "1.0.0",
        "description": "Cognitive Runtime for LLMs",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check(state: AppState = Depends(get_app_state)):
    return {"status": "healthy", "db": state.store.health_check()}


# Conversation endpoints
@app.post("/conversations/start")
async def start_conversation(req: ConversationStart, state: AppState = Depends(get_app_state)):
    conv = state.conversation_manager.start_conversation(req.topic_path, req.project, req.title)
    await ws_manager.broadcast({"type": "conversation_started", "conversation_id": conv.id})
    return {"conversation_id": conv.id, "topic": req.topic_path, "project": req.project}


@app.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: int, req: MessageRequest, state: AppState = Depends(get_app_state)):
    conv = state.store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM}
    role = role_map[req.role]

    if role == MessageRole.USER:
        msg = state.conversation_manager.log_user(req.content, req.metadata)
    elif role == MessageRole.ASSISTANT:
        msg = state.conversation_manager.log_assistant(req.content, req.metadata)
    else:
        msg = state.conversation_manager.log_system(req.content, req.metadata)

    # Incremental memory compilation
    current_memories = state.store.get_memories(
        topic_id=conv.topic_id,
        is_current=True,
        status=MemoryStatus.ACTIVE
    )
    new_memories = state.memory_compiler.process_message(msg, conv, current_memories)

    await ws_manager.broadcast({
        "type": "message_added",
        "conversation_id": conversation_id,
        "message_id": msg.id,
        "role": req.role,
        "new_memories": len(new_memories)
    })

    return {"message_id": msg.id, "new_memories_created": len(new_memories)}


@app.post("/conversations/{conversation_id}/end")
async def end_conversation(conversation_id: int, state: AppState = Depends(get_app_state)):
    conv = state.store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    ended = state.conversation_manager.end_conversation()

    # Compile memories
    new_memories = state.memory_compiler.compile_conversation(ended)

    # Write files
    topic = state.store.get_topic(ended.topic_id)
    if topic:
        state.memory_writer.write_all(topic)
        state.conversation_writer.write_conversation(ended.id)

    # Run consolidation
    consolidation_stats = state.consolidation_scheduler.force_run()

    await ws_manager.broadcast({
        "type": "conversation_ended",
        "conversation_id": conversation_id,
        "memories_compiled": len(new_memories),
        "consolidation": consolidation_stats
    })

    return {
        "conversation_id": conversation_id,
        "memories_compiled": len(new_memories),
        "consolidation": consolidation_stats
    }


# Memory endpoints
@app.get("/memory")
async def get_memory(topic_path: str, state: AppState = Depends(get_app_state)):
    topic = state.store.get_topic_by_path(topic_path)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    memories = state.store.get_memories(topic_id=topic.id, is_current=True, status=MemoryStatus.ACTIVE)

    return {
        "topic": topic.path,
        "memories": [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "resolution": m.resolution.name,
                "importance": m.importance,
                "confidence": m.confidence,
                "preview": m.content[:200] + ("..." if len(m.content) > 200 else "")
            }
            for m in memories
        ]
    }


@app.get("/memory/{memory_id}")
async def get_memory_detail(memory_id: int, state: AppState = Depends(get_app_state)):
    memory = state.store.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    versions = state.store.get_memory_versions(memory_id)

    return {
        "id": memory.id,
        "topic_id": memory.topic_id,
        "type": memory.memory_type.value,
        "content": memory.content,
        "resolution": memory.resolution.name,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "status": memory.status.value,
        "versions": [
            {"resolution": v.resolution.name, "compression_ratio": v.compression_ratio}
            for v in versions
        ]
    }


@app.post("/memory/{memory_id}/expand")
async def expand_memory(memory_id: int, req: ExpandRequest, state: AppState = Depends(get_app_state)):
    memory = state.store.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    target_res = ResolutionLevel(req.target_resolution)
    expanded = state.recall_engine.expand_resolution(memory, target_res)

    if not expanded:
        raise HTTPException(status_code=400, detail=f"No version available at {target_res.name}")

    return {
        "memory_id": memory_id,
        "target_resolution": target_res.name,
        "content": expanded.content
    }


# Recall endpoints
@app.post("/recall")
async def recall(req: RecallRequest, state: AppState = Depends(get_app_state)):
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    recall_level = RecallLevel(req.level)
    memories, tokens = state.recall_engine.recall(req.query, topic_id, recall_level, req.max_tokens)

    return {
        "query": req.query,
        "level": recall_level.name,
        "memories_retrieved": len(memories),
        "tokens": tokens,
        "memories": [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "resolution": m.resolution.name,
                "importance": m.importance,
                "confidence": m.confidence,
                "content": m.content
            }
            for m in memories
        ]
    }


@app.post("/recall/human")
async def human_recall(req: HumanRecallRequest, state: AppState = Depends(get_app_state)):
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    response = state.human_recall_engine.simulate_human_recall(req.query, topic_id, req.max_tokens)
    return {"query": req.query, "response": response}


@app.get("/recall/explain")
async def recall_explain(query: str, topic_path: str | None = None, state: AppState = Depends(get_app_state)):
    topic_id = None
    if topic_path:
        topic = state.store.get_topic_by_path(topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    explanation = state.human_recall_engine.get_recall_explanation(query, topic_id)
    return explanation


# Context endpoints
@app.post("/context")
async def build_context(req: ContextRequest, state: AppState = Depends(get_app_state)):
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    current_memories = state.store.get_memories(
        topic_id=topic_id,
        is_current=True,
        status=MemoryStatus.ACTIVE
    ) if topic_id else None

    context = state.context_builder.build_context(req.query, topic_id, req.max_tokens, current_memories)
    stats = state.context_builder.get_context_stats()

    return {
        "context": context,
        "stats": stats
    }


# Consolidation endpoints
@app.post("/consolidate")
async def consolidate(req: ConsolidationRequest, state: AppState = Depends(get_app_state)):
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        stats = state.consolidation_engine.consolidate_topic(topic.id)
    else:
        stats = state.consolidation_scheduler.force_run()

    return stats


@app.get("/consolidation/status")
async def consolidation_status(state: AppState = Depends(get_app_state)):
    stats = state.consolidation_engine.run_consolidation_cycle()
    return stats


# Association endpoints
@app.post("/associations/analyze")
async def analyze_associations(req: AssociationRequest, state: AppState = Depends(get_app_state)):
    topic = state.store.get_topic_by_path(req.topic_path)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    associations = state.association_engine.analyze_and_create_associations(topic.id)
    return {"created": len(associations)}


@app.get("/associations/{memory_id}")
async def get_associations(memory_id: int, min_strength: float = 0.3, depth: int = 2, state: AppState = Depends(get_app_state)):
    memory = state.store.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Direct associations
    related = state.store.get_related_memories(memory_id, min_strength)

    result = {
        "memory_id": memory_id,
        "direct": [
            {
                "target_id": m.id,
                "type": assoc.association_type.value,
                "strength": assoc.strength,
                "preview": m.content[:200] + ("..." if len(m.content) > 200 else "")
            }
            for m, assoc in related
        ]
    }

    # Multi-depth exploration
    if depth > 1:
        visited = {memory_id}
        current_level = [(memory_id, 0)]
        all_paths = []

        for d in range(1, depth + 1):
            next_level = []
            for mem_id, _ in current_level:
                related = state.store.get_related_memories(mem_id, min_strength)
                for mem, assoc in related:
                    if mem.id not in visited:
                        visited.add(mem.id)
                        next_level.append((mem.id, assoc.strength))
                        all_paths.append({
                            "depth": d,
                            "memory_id": mem.id,
                            "strength": assoc.strength,
                            "preview": mem.content[:100] + "..."
                        })

            current_level = next_level
            if not current_level:
                break

        result["multi_depth"] = all_paths

    return result


@app.get("/associations/path/{source_id}/{target_id}")
async def find_assoc_path(source_id: int, target_id: int, max_depth: int = 3, state: AppState = Depends(get_app_state)):
    path = state.association_engine.find_association_path(source_id, target_id, max_depth)

    if path:
        return {
            "found": True,
            "length": len(path),
            "path": [
                {
                    "source_id": a.source_memory_id,
                    "target_id": a.target_memory_id,
                    "type": a.association_type.value,
                    "strength": a.strength
                }
                for a in path
            ]
        }
    else:
        return {"found": False, "message": f"No path found within depth {max_depth}"}


# Confidence endpoints
@app.post("/confidence")
async def compute_confidence(req: ConfidenceRequest, state: AppState = Depends(get_app_state)):
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    recall_level = RecallLevel(req.level)
    memories, _ = state.recall_engine.recall(req.query, topic_id, recall_level, max_tokens=4000)

    if not memories:
        return {"query": req.query, "confidence": 0.0, "level": "vague", "message": "No memories found"}

    confidence = state.confidence_engine.compute_overall_confidence(req.query, memories, recall_level)

    return {
        "query": req.query,
        "recall_level": recall_level.name,
        "memories_found": len(memories),
        "overall_confidence": confidence.overall,
        "level": confidence.level.value,
        "natural_language": confidence.to_natural_language(),
        "breakdown": {
            "memory": confidence.memory_confidence,
            "retrieval": confidence.retrieval_confidence,
            "temporal": confidence.temporal_confidence,
            "source": confidence.source_confidence,
        }
    }


@app.get("/confidence/memory/{memory_id}")
async def memory_confidence(memory_id: int, state: AppState = Depends(get_app_state)):
    memory = state.store.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    conf = state.confidence_engine.compute_memory_confidence(memory)

    return {
        "memory_id": memory_id,
        "type": memory.memory_type.value,
        "resolution": memory.resolution.name,
        "status": memory.status.value,
        "base_confidence": memory.confidence,
        "computed_confidence": conf,
        "factors": {
            "base": memory.confidence,
            "status": 1.0 if memory.status == MemoryStatus.ACTIVE else 0.9,
            "resolution": 1.0 - memory.resolution.value * 0.05,
            "age": max(0.5, 1.0 - (datetime.now() - memory.created_at).days / 365 * 0.3),
            "access": min(1.0, 1.0 + memory.access_count * 0.02),
        }
    }


# Temporal endpoints
@app.post("/temporal/state")
async def temporal_state(req: TemporalStateRequest, state: AppState = Depends(get_app_state)):
    from datetime import datetime
    topic = state.store.get_topic_by_path(req.topic_path)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    timestamp = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()
    t_state = state.temporal_engine.get_state_at(timestamp, topic.id, req.include_archived)

    return {
        "timestamp": t_state.timestamp.isoformat(),
        "active_memories": len(t_state.active_memories),
        "active_decisions": len(t_state.active_decisions),
        "memories": [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "resolution": m.resolution.name,
                "status": m.status.value,
                "preview": m.content[:200] + "..."
            }
            for m in t_state.active_memories[:50]
        ]
    }


@app.post("/temporal/changes")
async def temporal_changes(req: TemporalChangesRequest, state: AppState = Depends(get_app_state)):
    from datetime import datetime
    topic = state.store.get_topic_by_path(req.topic_path)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    start_ts = datetime.fromisoformat(req.start)
    end_ts = datetime.fromisoformat(req.end)

    changes = state.temporal_engine.get_changes_between(start_ts, end_ts, topic.id)

    return {
        "period": {"start": req.start, "end": req.end},
        "added": len(changes["added"]),
        "removed": len(changes["removed"]),
        "modified": len(changes["modified"]),
        "details": {
            "added": [{"id": m["id"], "type": m["type"], "preview": m["preview"]} for m in changes["added"]],
            "removed": [{"id": m["id"], "type": m["type"], "preview": m["preview"]} for m in changes["removed"]],
            "modified": changes["modified"][:20]
        }
    }


@app.post("/temporal/timeline")
async def temporal_timeline(req: TemporalChangesRequest, state: AppState = Depends(get_app_state)):
    from datetime import datetime
    topic = state.store.get_topic_by_path(req.topic_path)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    start_ts = datetime.fromisoformat(req.start) if req.start else None
    end_ts = datetime.fromisoformat(req.end) if req.end else None

    timeline = state.temporal_engine.get_timeline_for_topic(topic.id, start_ts, end_ts)

    return {
        "topic": req.topic_path,
        "events": timeline[:100]
    }


# Style endpoints
@app.post("/style/profile")
async def style_profile(req: StyleProfileRequest, state: AppState = Depends(get_app_state)):
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topics = [topic]
    else:
        topics = state.store.list_topics()

    profiles = []
    for topic in topics:
        profile = state.style_engine.build_topic_profile(topic.id)
        profiles.append({
            "topic": topic.path,
            "profile": profile
        })

    return {"profiles": profiles}


@app.post("/style/reconstruct")
async def reconstruct_style(req: StyleReconstructRequest, state: AppState = Depends(get_app_state)):
    memory = state.store.get_memory(req.memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    profile = state.style_engine.build_topic_profile(memory.topic_id)

    if req.target_style == "user":
        target = profile["user_style"]
    elif req.target_style == "assistant":
        target = profile["assistant_style"]
    else:
        target = profile["combined_style"]

    from artificial_memory.memory.style import ConversationStyle
    target_style = ConversationStyle.from_dict(target)

    reconstructed_text = state.style_engine.reconstruct_style(memory.content, target_style)

    return {
        "memory_id": req.memory_id,
        "original": memory.content[:500],
        "reconstructed": reconstructed_text[:500],
        "target_style": req.target_style,
        "style_diff": {
            "tone": target_style.tone,
            "formality": target_style.formality,
            "fillers_added": target_style.fillers[:5]
        }
    }


# Metrics endpoints
@app.get("/metrics")
async def get_metrics(state: AppState = Depends(get_app_state)):
    return {
        "summary": state.metrics_collector.get_summary(),
        "compression": state.metrics_collector.get_compression_stats(),
        "recall": state.metrics_collector.get_recall_stats(),
        "context": state.metrics_collector.get_context_stats(),
    }


@app.get("/metrics/export")
async def export_metrics(state: AppState = Depends(get_app_state)):
    return state.metrics_collector.export_all()


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, state: AppState = Depends(get_app_state)):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Echo back with server timestamp
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Vector Search endpoints
class VectorSearchRequest(BaseModel):
    query: str
    topic_path: str | None = None
    k: int = 10
    threshold: float = 0.0


class HybridSearchRequest(BaseModel):
    query: str
    topic_path: str | None = None
    k: int = 10
    vector_weight: float = 0.7
    keyword_weight: float = 0.3


@app.post("/vector/search")
async def vector_search(req: VectorSearchRequest, state: AppState = Depends(get_app_state)):
    """Search memories using vector similarity."""
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    results = state.vector_search_engine.search(req.query, topic_id=topic_id, k=req.k, threshold=req.threshold)

    return {
        "query": req.query,
        "results": [
            {
                "memory_id": r.memory_id,
                "score": r.score,
                "distance": r.distance,
                "metadata": r.metadata,
            }
            for r in results
        ]
    }


@app.post("/vector/hybrid")
async def hybrid_search(req: HybridSearchRequest, state: AppState = Depends(get_app_state)):
    """Hybrid vector + keyword search."""
    topic_id = None
    if req.topic_path:
        topic = state.store.get_topic_by_path(req.topic_path)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        topic_id = topic.id

    results = state.vector_search_engine.hybrid_search(
        req.query, topic_id=topic_id, k=req.k,
        vector_weight=req.vector_weight, keyword_weight=req.keyword_weight
    )

    return {
        "query": req.query,
        "results": [
            {
                "memory_id": r.memory_id,
                "score": r.score,
                "distance": r.distance,
                "metadata": r.metadata,
            }
            for r in results
        ]
    }


@app.get("/vector/stats")
async def vector_stats(state: AppState = Depends(get_app_state)):
    """Get vector index statistics."""
    stats = state.vector_search_engine.get_stats()
    return stats


@app.post("/vector/rebuild")
async def vector_rebuild(state: AppState = Depends(get_app_state)):
    """Rebuild vector index from all memories."""
    state.vector_search_engine._rebuild_index()
    stats = state.vector_search_engine.get_stats()
    return {"message": "Index rebuilt", "stats": stats}


# Auth endpoints
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool


class APIKeyCreate(BaseModel):
    name: str
    scopes: list[str] = []
    expires_days: int = 365


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    expires_at: str | None


class APIKeyCreateResponse(BaseModel):
    api_key: APIKeyResponse
    raw_key: str


@app.post("/auth/register", response_model=UserResponse)
async def register(req: UserRegister, state: AppState = Depends(get_app_state)):
    """Register a new user."""
    if state.auth_service.get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    if state.auth_service.get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already exists")

    user = state.auth_service.create_user(req.username, req.email, req.password, UserRole(req.role))
    return UserResponse(id=user.id, username=user.username, email=user.email, role=user.role.value, is_active=user.is_active)


@app.post("/auth/login")
async def login(req: UserLogin, state: AppState = Depends(get_app_state)):
    """Login and get session token."""
    user = state.auth_service.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session = state.auth_service.create_session(user.id)
    return {"token": session.token, "expires_at": session.expires_at.isoformat(), "user": UserResponse(id=user.id, username=user.username, email=user.email, role=user.role.value, is_active=user.is_active)}


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user(authorization: str = None, state: AppState = Depends(get_app_state)):
    """Get current user info."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization[7:]
    session = state.auth_service.validate_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = state.auth_service.get_user(session.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(id=user.id, username=user.username, email=user.email, role=user.role.value, is_active=user.is_active)


@app.post("/auth/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(req: APIKeyCreate, authorization: str = None, state: AppState = Depends(get_app_state)):
    """Create an API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization[7:]
    session = state.auth_service.validate_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    api_key, raw_key = state.auth_service.create_api_key(
        session.user_id, req.name, req.scopes, req.expires_days
    )

    return APIKeyCreateResponse(
        api_key=APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            scopes=api_key.scopes,
            is_active=api_key.is_active,
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        ),
        raw_key=raw_key,
    )


@app.get("/auth/api-keys")
async def list_api_keys(authorization: str = None, state: AppState = Depends(get_app_state)):
    """List user's API keys."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization[7:]
    session = state.auth_service.validate_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    api_keys = state.auth_service.list_user_api_keys(session.user_id)
    return [APIKeyResponse(
        id=k.id, name=k.name, key_prefix=k.key_prefix,
        scopes=k.scopes, is_active=k.is_active,
        expires_at=k.expires_at.isoformat() if k.expires_at else None
    ) for k in api_keys]


@app.delete("/auth/api-keys/{key_id}")
async def revoke_api_key(key_id: int, authorization: str = None, state: AppState = Depends(get_app_state)):
    """Revoke an API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization[7:]
    session = state.auth_service.validate_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    success = state.auth_service.revoke_api_key(key_id)
    return {"success": success}


# Chat endpoints
class ChatRequest(BaseModel):
    message: str
    topic_path: str | None = None
    conversation_id: int | None = None
    provider: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4000
    stream: bool = False
    use_context: bool = True


class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    context_used: bool
    topic_id: int | None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """Chat with LLM using Context Runtime."""
    result = await state.llm_manager.chat(
        user_message=req.message,
        topic_path=req.topic_path,
        conversation_id=req.conversation_id,
        provider_name=req.provider,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
        use_context=req.use_context,
    )

    if req.stream:
        # For streaming, we return a different format
        raise HTTPException(status_code=501, detail="Streaming not yet implemented in this endpoint")

    return ChatResponse(**result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, state: AppState = Depends(get_app_state)):
    """Stream chat response."""
    # For streaming, we use Server-Sent Events
    from fastapi.responses import StreamingResponse

    async def stream_response():
        async for chunk in state.llm_manager.chat_stream(
            user_message=req.message,
            topic_path=req.topic_path,
            conversation_id=req.conversation_id,
            provider_name=req.provider,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            use_context=req.use_context,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# Static files and UI


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return FileResponse("static/index.html")


# CLI integration - allow running CLI commands via API
class CLICommandRequest(BaseModel):
    command: str
    args: list[str] = []


@app.post("/cli/execute")
async def execute_cli(req: CLICommandRequest, state: AppState = Depends(get_app_state)):
    # This would integrate with the CLI - for now return available commands
    return {
        "available_commands": [
            "start", "user", "assistant", "end", "memory", "context",
            "recall", "expand", "timeline", "trace", "dump_ir",
            "stats", "compress", "consolidation_status", "consolidate_topic",
            "associate", "associations", "assoc_path",
            "confidence", "mem_confidence",
            "temporal_state", "temporal_changes", "temporal_timeline",
            "style_profile", "reconstruct_style",
            "human_recall", "recall_explain",
            "vector_search", "hybrid_search", "vector_stats", "vector_rebuild"
        ]
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
