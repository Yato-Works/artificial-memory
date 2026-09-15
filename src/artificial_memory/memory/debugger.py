from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.compiler import CompilerPipeline
from artificial_memory.context.builder import EnhancedContextBuilder
from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import (
    Conversation,
    Memory,
    RecallLevel,
)
from artificial_memory.memory.belief import BeliefEngine
from artificial_memory.memory.contradiction import ContradictionDetector
from artificial_memory.memory.counterfactual import CounterfactualEngine
from artificial_memory.memory.dependency import DependencyGraph
from artificial_memory.memory.evolution import MemoryEvolutionEngine
from artificial_memory.memory.healing import MemoryHealer
from artificial_memory.memory.integrity import IntegrityMetrics
from artificial_memory.memory.stale import StaleMemoryDetector
from artificial_memory.recall.adaptive import AdaptiveRecallEngine


class DebugEventType(StrEnum):
    RECALL_QUERY = "recall_query"
    RECALL_CANDIDATES = "recall_candidates"
    RECALL_SCORED = "recall_scored"
    RECALL_SELECTED = "recall_selected"
    RECALL_REJECTED = "recall_rejected"
    CONTEXT_BUILD = "context_build"
    CONTEXT_PART_ADDED = "context_part_added"
    CONTEXT_PART_REJECTED = "context_part_rejected"
    CONTEXT_TRUNCATED = "context_truncated"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_EXPANDED = "memory_expanded"
    COMPILATION_START = "compilation_start"
    COMPILATION_STAGE = "compilation_stage"
    COMPILATION_COMPLETE = "compilation_complete"
    HEALING_ACTION = "healing_action"
    BELIEF_UPDATE = "belief_update"
    CONTRADICTION_DETECTED = "contradiction_detected"
    DEPENDENCY_ADDED = "dependency_added"
    IMPACT_ANALYZED = "impact_analyzed"
    COUNTERFACTUAL_EVALUATED = "counterfactual_evaluated"
    ADAPTIVE_RECALL = "adaptive_recall"


@dataclass
class DebugEvent:
    """A single debug event in the trace."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: DebugEventType = DebugEventType.RECALL_QUERY
    component: str = ""  # e.g., "recall_engine", "context_builder"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""


@dataclass
class RecallTrace:
    """Complete trace of a recall operation."""
    trace_id: str
    query: str
    topic_id: int | None
    recall_level: RecallLevel
    max_tokens: int
    start_time: datetime
    end_time: datetime | None = None
    events: list[DebugEvent] = field(default_factory=list)
    final_memories: list[int] = field(default_factory=list)  # memory IDs
    total_tokens: int = 0
    latency_ms: int = 0

    @property
    def duration_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0


@dataclass
class ContextTrace:
    """Complete trace of context building."""
    trace_id: str
    query: str
    topic_id: int | None
    max_tokens: int
    start_time: datetime
    end_time: datetime | None = None
    events: list[DebugEvent] = field(default_factory=list)
    final_context_tokens: int = 0
    selected_parts: int = 0
    rejected_parts: int = 0
    truncated_parts: int = 0
    budget_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationTrace:
    """Complete trace of memory compilation."""
    trace_id: str
    conversation_id: int
    start_time: datetime
    end_time: datetime | None = None
    events: list[DebugEvent] = field(default_factory=list)
    input_messages: int = 0
    ir_units_generated: int = 0
    memories_created: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


class MemoryDebugger:
    """Debugger for the Artificial Memory runtime.

    Provides detailed tracing and explanation of:
    - Why a memory was selected/rejected in recall
    - Why resolution was expanded/not expanded
    - How context was built and budget allocated
    - What compilation stages did
    - Why healing actions were taken
    """

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: RecallEngine,
        context_builder: EnhancedContextBuilder,
        compiler_pipeline: CompilerPipeline | None = None,
        evolution_engine: MemoryEvolutionEngine | None = None,
        belief_engine: BeliefEngine | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        dependency_graph: DependencyGraph | None = None,
        counterfactual_engine: CounterfactualEngine | None = None,
        adaptive_recall: AdaptiveRecallEngine | None = None,
        integrity_metrics: IntegrityMetrics | None = None,
        stale_detector: StaleMemoryDetector | None = None,
        healer: MemoryHealer | None = None,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.compiler_pipeline = compiler_pipeline
        self.evolution_engine = evolution_engine
        self.belief_engine = None  # Would be passed if needed
        self.contradiction_detector = None
        self.dependency_graph = None
        self.counterfactual_engine = None
        self.adaptive_recall = adaptive_recall
        self.integrity_metrics = None
        self.stale_detector = None
        self.healer = None

        # Active traces
        self._active_recall_trace: RecallTrace | None = None
        self._active_context_trace: ContextTrace | None = None
        self._active_compilation_trace: CompilationTrace | None = None

        # History
        self._recall_history: list[RecallTrace] = []
        self._context_history: list[ContextTrace] = []
        self._compilation_history: list[CompilationTrace] = []

        # Trace counter
        self._trace_counter = 0

    def _new_trace_id(self, prefix: str) -> str:
        self._trace_counter += 1
        return f"{prefix}_{self._trace_counter}_{datetime.now().strftime('%H%M%S%f')}"

    def _add_event(self, trace: Any, event_type: DebugEventType,
                   component: str, message: str, data: dict = None):
        event = DebugEvent(
            event_type=event_type,
            component=component,
            message=message,
            data=data or {},
            trace_id=getattr(trace, 'trace_id', ''),
        )
        trace.events.append(event)

    # ==================== Recall Tracing ====================

    def trace_recall(
        self,
        query: str,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> RecallTrace:
        """Execute recall with full tracing."""
        trace_id = self._new_trace_id("recall")
        trace = RecallTrace(
            trace_id=trace_id,
            query=query,
            topic_id=topic_id,
            recall_level=level,
            max_tokens=max_tokens,
            start_time=datetime.now(),
        )

        self._active_recall_trace = trace
        self._add_event(trace, DebugEventType.RECALL_QUERY, "recall_engine",
                       f"Starting recall for: {query}",
                       {"query": query, "topic_id": topic_id, "level": level.name, "max_tokens": max_tokens})

        try:
            # Get candidates from each level (simplified)
            self._add_event(trace, DebugEventType.RECALL_CANDIDATES, "recall_engine",
                           "Fetching candidates from recall engine")

            memories, tokens = self.recall_engine.recall(query, topic_id, level, max_tokens)

            self._add_event(trace, DebugEventType.RECALL_CANDIDATES, "recall_engine",
                           f"Retrieved {len(memories)} candidates, {tokens} tokens",
                           {"candidate_count": len(memories), "tokens": tokens})

            # Score them (if adaptive)
            if self.adaptive_recall:
                self._add_event(trace, DebugEventType.RECALL_SCORED, "adaptive_recall",
                               "Scoring candidates with utility function")
                # Would call adaptive recall scoring here

            # Selection
            self._add_event(trace, DebugEventType.RECALL_SELECTED, "recall_engine",
                           f"Selected {len(memories)} memories",
                           {"selected_ids": [m.id for m in memories]})

            trace.final_memories = [m.id for m in memories]
            trace.total_tokens = tokens

        except Exception as e:
            self._add_event(trace, DebugEventType.RECALL_REJECTED, "recall_engine",
                           f"Error during recall: {e}")
            raise
        finally:
            trace.end_time = datetime.now()
            trace.latency_ms = trace.duration_ms
            self._active_recall_trace = None
            self._recall_history.append(trace)

        return trace

    def explain_recall(self, trace: RecallTrace) -> dict[str, Any]:
        """Generate human-readable explanation from recall trace."""
        explanation = {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "recall_level": trace.recall_level.name,
            "duration_ms": trace.duration_ms,
            "total_candidates": len(trace.final_memories),
            "total_tokens": trace.total_tokens,
            "events": [],
            "why_selected": [],
            "why_rejected": [],
        }

        for event in trace.events:
            explanation["events"].append({
                "time": event.timestamp.isoformat(),
                "type": event.event_type.value,
                "component": event.component,
                "message": event.message,
            })

            if event.event_type == DebugEventType.RECALL_SELECTED:
                for mem_id in event.data.get("selected_ids", []):
                    mem = self.store.get_memory(mem_id)
                    if mem:
                        explanation["why_selected"].append({
                            "memory_id": mem_id,
                            "type": mem.memory_type.value,
                            "resolution": mem.resolution.name,
                            "importance": mem.importance,
                            "confidence": mem.confidence,
                            "reason": f"Matched query at {trace.recall_level.name} level",
                        })

            # Would add rejected memories if tracked

        return explanation

    # ==================== Context Building Tracing ====================

    def trace_context_build(
        self,
        query: str,
        topic_id: int | None = None,
        max_tokens: int = 8000,
        current_memories: list[Memory] | None = None,
    ) -> ContextTrace:
        """Execute context building with full tracing."""
        trace_id = self._new_trace_id("context")
        trace = ContextTrace(
            trace_id=trace_id,
            query=query,
            topic_id=topic_id,
            max_tokens=max_tokens,
            start_time=datetime.now(),
        )

        self._active_context_trace = trace
        self._add_event(trace, DebugEventType.CONTEXT_BUILD, "context_builder",
                       f"Building context for: {query}")

        try:
            # Use the context builder's IR method if available
            context_ir = self.context_builder.build_context_ir(
                query, topic_id, max_tokens, current_memories
            )

            trace.final_context_tokens = context_ir.stats.effective_tokens
            trace.selected_parts = context_ir.stats.selected_parts
            trace.truncated_parts = context_ir.stats.truncated_parts
            trace.budget_stats = {
                "raw_tokens": context_ir.stats.raw_tokens,
                "effective_tokens": context_ir.stats.effective_tokens,
                "compression_ratio": context_ir.stats.compression_ratio,
                "tier_distribution": context_ir.stats.tier_distribution,
                "priority_distribution": context_ir.stats.priority_distribution,
            }

            # Trace parts
            for part in context_ir.parts:
                selected_content = getattr(context_ir.stats, "selected_content", [])
                if part.content in selected_content:
                    self._add_event(trace, DebugEventType.CONTEXT_PART_ADDED, "context_builder",
                                   f"Added part: {part.source} ({part.tokens} tokens)",
                                   {"source": part.source, "tokens": part.tokens,
                                    "priority": part.priority, "tier": part.tier.value})
                    trace.selected_parts += 1
                else:
                    self._add_event(trace, DebugEventType.CONTEXT_PART_REJECTED, "context_builder",
                                   f"Rejected part: {part.source} ({part.tokens} tokens)",
                                   {"source": part.source, "tokens": part.tokens,
                                    "reason": "Budget exceeded or low priority"})
                    trace.rejected_parts += 1

            if trace.truncated_parts > 0:
                self._add_event(trace, DebugEventType.CONTEXT_TRUNCATED, "context_builder",
                               f"Truncated {trace.truncated_parts} parts to fit budget")

        except Exception as e:
            self._add_event(trace, DebugEventType.CONTEXT_BUILD, "context_builder",
                           f"Error: {e}")
            raise
        finally:
            trace.end_time = datetime.now()
            self._active_context_trace = None
            self._context_history.append(trace)

        return trace

    def explain_context(self, trace: ContextTrace) -> dict[str, Any]:
        """Explain why context was built this way."""
        return {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "max_tokens": trace.max_tokens,
            "final_tokens": trace.final_context_tokens,
            "selected_parts": trace.selected_parts,
            "rejected_parts": trace.rejected_parts,
            "truncated_parts": trace.truncated_parts,
            "budget_stats": trace.budget_stats,
            "explanation": (
                f"Context built from {trace.selected_parts} parts ({trace.final_context_tokens} tokens). "
                f"{trace.rejected_parts} parts rejected due to budget/priority. "
                f"{trace.truncated_parts} parts truncated."
            ),
        }

    # ==================== Compilation Tracing ====================

    def trace_compilation(
        self,
        conversation: Conversation,
        messages: list,
    ) -> CompilationTrace:
        """Execute compilation with full tracing."""
        if not self.compiler_pipeline:
            raise RuntimeError("Compiler pipeline not available")

        trace_id = self._new_trace_id("compile")
        trace = CompilationTrace(
            trace_id=trace_id,
            conversation_id=conversation.id,
            start_time=datetime.now(),
            input_messages=len(messages),
        )

        self._active_compilation_trace = trace
        self._add_event(trace, DebugEventType.COMPILATION_START, "compiler_pipeline",
                       f"Starting compilation of conversation {conversation.id}",
                       {"message_count": len(messages)})

        try:
            # Use pipeline with diagnostics
            memory_irs, diagnostics = self.compiler_pipeline.compile_with_diagnostics(
                conversation, messages
            )

            trace.ir_units_generated = sum(len(m) for m in memory_irs) if isinstance(memory_irs, list) else 0
            trace.memories_created = len(memory_irs) if isinstance(memory_irs, list) else 0
            trace.diagnostics = diagnostics

            for stage in self.compiler_pipeline.stages:
                self._add_event(trace, DebugEventType.COMPILATION_STAGE, "compiler_pipeline",
                               f"Stage: {stage.name}",
                               {"stage": stage.name, "memories_so_far": trace.memories_created})

            self._add_event(trace, DebugEventType.COMPILATION_COMPLETE, "compiler_pipeline",
                           f"Compilation complete: {trace.memories_created} memories",
                           {"diagnostics_count": len(diagnostics)})

        except Exception as e:
            self._add_event(trace, DebugEventType.COMPILATION_COMPLETE, "compiler_pipeline",
                           f"Compilation failed: {e}")
            raise
        finally:
            trace.end_time = datetime.now()
            self._active_compilation_trace = None
            self._compilation_history.append(trace)

        return trace

    def explain_compilation(self, trace: CompilationTrace) -> dict[str, Any]:
        """Explain compilation decisions."""
        return {
            "trace_id": trace.trace_id,
            "conversation_id": trace.conversation_id,
            "duration_ms": int((trace.end_time - trace.start_time).total_seconds() * 1000) if trace.end_time else 0,
            "messages_processed": trace.input_messages,
            "ir_units": trace.ir_units_generated,
            "memories_created": trace.memories_created,
            "stages": [
                {
                    "type": e.event_type.value,
                    "component": e.component,
                    "message": e.message,
                }
                for e in trace.events
            ],
            "diagnostics": trace.diagnostics,
        }

    # ==================== General Inspection ====================

    def inspect_memory(self, memory_id: int) -> dict[str, Any]:
        """Complete inspection of a memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            return {"error": "Memory not found"}

        # Get related data
        versions = self.store.get_memory_versions(memory_id)
        associations = self.store.get_associations(memory_id)
        compression_history = self.store.get_compression_history(memory_id)
        self.store.get_memory_versions(memory_id)  # Would need proper provenance

        # Integrity check
        integrity_report = None
        if self.integrity_metrics:
            integrity_report = self.integrity_metrics.compute_overall_integrity(memory)

        # Staleness
        stale_assessment = None
        if self.stale_detector:
            stale_assessment = self.stale_detector.assess_memory(memory)

        # Compression validation
        if hasattr(self, 'compression_validator'):
            self.compression_validator.validate_all_versions(
                self.store.get_memory(memory_id)
            )

        return {
            "memory": {
                "id": memory.id,
                "type": memory.memory_type.value,
                "resolution": memory.resolution.name,
                "status": memory.status.value,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "content": memory.content,
                "content_length": len(memory.content),
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                "access_count": memory.access_count,
            },
            "provenance": {
                "source_conversation_id": memory.source_conversation_id,
                "source_message_id": memory.source_message_id,
                "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
                "is_current": memory.is_current,
            },
            "versions": [
                {
                    "resolution": v.resolution.name,
                    "compression_ratio": v.compression_ratio,
                    "created_at": v.created_at.isoformat(),
                    "source": v.source,
                    "content_length": len(v.content),
                }
                for v in versions
            ],
            "associations": [
                {
                    "target_id": a.target_memory_id if a.source_memory_id == memory_id else a.source_memory_id,
                    "type": a.association_type.value,
                    "strength": a.strength,
                }
                for a in associations
            ],
            "compression_history": [
                {
                    "from": c.from_resolution.name,
                    "to": c.to_resolution.name,
                    "ratio": c.compression_ratio,
                    "method": c.method.value,
                }
                for c in compression_history
            ],
            "integrity": {
                "overall_score": integrity_report.overall_score if integrity_report else None,
                "issues": [
                    {
                        "type": i.issue_type.value,
                        "severity": i.severity.value,
                        "description": i.description,
                    }
                    for i in integrity_report.issues
                ] if integrity_report else [],
            } if integrity_report else None,
            "staleness": {
                "is_stale": stale_assessment.is_stale if stale_assessment else None,
                "score": stale_assessment.staleness_score if stale_assessment else None,
                "signals": [
                    {"reason": s.reason.value, "weight": s.weight, "desc": s.description}
                    for s in stale_assessment.signals
                ] if stale_assessment else [],
            } if stale_assessment else None,
        }

    def get_recent_traces(
        self,
        trace_type: str = "all",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent debug traces."""
        if trace_type == "recall" or trace_type == "all":
            recall_traces = [
                {
                    "trace_id": t.trace_id,
                    "query": t.query,
                    "level": t.recall_level.name,
                    "duration_ms": t.duration_ms,
                    "memories": len(t.final_memories),
                    "tokens": t.total_tokens,
                    "time": t.start_time.isoformat(),
                }
                for t in self._recall_history[-limit:]
            ]
        else:
            recall_traces = []

        if trace_type == "context" or trace_type == "all":
            context_traces = [
                {
                    "trace_id": t.trace_id,
                    "query": t.query,
                    "duration_ms": int((t.end_time - t.start_time).total_seconds() * 1000) if t.end_time else 0,
                    "tokens": t.final_context_tokens,
                    "selected": t.selected_parts,
                    "rejected": t.rejected_parts,
                    "time": t.start_time.isoformat(),
                }
                for t in self._context_history[-limit:]
            ]
        else:
            context_traces = []

        if trace_type == "compile" or trace_type == "all":
            compile_traces = [
                {
                    "trace_id": t.trace_id,
                    "conversation_id": t.conversation_id,
                    "duration_ms": int((t.end_time - t.start_time).total_seconds() * 1000) if t.end_time else 0,
                    "messages": t.input_messages,
                    "memories": t.memories_created,
                    "time": t.start_time.isoformat(),
                }
                for t in self._compilation_history[-limit:]
            ]
        else:
            compile_traces = []

        return {
            "recall": recall_traces,
            "context": context_traces,
            "compilation": compile_traces,
        }


def create_memory_debugger(
    store: MemoryStore,
    recall_engine: RecallEngine,
    context_builder: EnhancedContextBuilder,
    compiler_pipeline=None,
    evolution_engine=None,
    belief_engine=None,
    contradiction_detector=None,
    dependency_graph=None,
    counterfactual_engine=None,
    adaptive_recall=None,
    integrity_metrics=None,
    stale_detector=None,
    healer=None,
) -> MemoryDebugger:
    return MemoryDebugger(
        store, recall_engine, context_builder, compiler_pipeline,
        evolution_engine, belief_engine, contradiction_detector,
        dependency_graph, counterfactual_engine, adaptive_recall,
        integrity_metrics, stale_detector, healer
    )
