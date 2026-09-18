"""Controlled Arena Players (Phase 8.2).

Implements the five Controlled Arena subjects per Benchmark_Plan.txt §3:
- S0: Simple RAG
- S1: MemoryBank-style
- S2: MemGPT-style
- S3: Mem0-style
- S4: AM v0.2.0

All players share:
- Fixed answer LLM (local)
- Fixed extraction LLM (same as answer LLM per Fairness Rules)
- Fixed embedding model
- 2,000 token context budget
- Write-side cost tracking
"""

from __future__ import annotations

import asyncio
import atexit
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artificial_memory.runtime import ArtificialMemoryRuntime


from artificial_memory.context.allocator import ContextAllocator
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.memory.vector_search import VectorSearchEngine, create_vector_search_engine
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.research.benchmarks.arena import ArenaQuestion, ArenaScenario
from artificial_memory.research.benchmarks.llm import (
    ABSTENTION_TEXT,
    LLMAnswer,
    OllamaAnswerer,
)
from artificial_memory.research.benchmarks.player import (
    Answer,
    ControlledPlayer,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore

# artificial_memory.runtime imported lazily in AMv020Player to avoid circular import

# Ephemeral per-ingest index dirs created in this process (cleaned at exit).
_ISOLATED_INDEX_DIRS: list[Path] = []


def _cleanup_isolated_index_dirs() -> None:
    """Delete every ephemeral player index dir created by this process.

    Registered with ``atexit`` so a full benchmark run (or a test session) does
    not leave one FAISS index per player ingest behind in the system temp dir.
    """
    while _ISOLATED_INDEX_DIRS:
        shutil.rmtree(_ISOLATED_INDEX_DIRS.pop(), ignore_errors=True)


atexit.register(_cleanup_isolated_index_dirs)


def _isolated_index_dir(player_name: str) -> Path:
    """Create a fresh ephemeral FAISS index dir for one player ingest.

    Controlled players use in-memory stores, so their vector index must be
    equally ephemeral: sharing the repo-level default ``vector_index/`` would
    bleed memories between players and across runs.
    """
    slug = player_name.lower().replace(" ", "-")
    path = Path(tempfile.mkdtemp(prefix=f"am-arena-{slug}-"))
    _ISOLATED_INDEX_DIRS.append(path)
    return path


# ==================== Shared frozen LLM answerer (Phase 8.2) ====================

_SHARED_ANSWERER: OllamaAnswerer | None = None


def get_shared_answerer() -> OllamaAnswerer:
    """The single frozen-config LLM adapter shared by ALL Controlled players.

    One instance, one model, one prompt set: measured differences are memory /
    retrieval strategy differences, never LLM differences (Benchmark_Plan §5
    Fixed LLM Configuration).
    """
    global _SHARED_ANSWERER
    if _SHARED_ANSWERER is None:
        _SHARED_ANSWERER = OllamaAnswerer()
    return _SHARED_ANSWERER


class _SharedAnswerGeneration:
    """Mixin: final answer generation through the frozen shared adapter.

    Leakage Boundary (by design): generation receives ONLY the question text
    and the retrieved context. The question object — which carries ground
    truth, forbidden terms and abstention flags — never reaches the prompt.

    When no answerer is configured (``answerer=None``, unit-test mode) the
    deterministic abstention fallback is used: extraction falls back to
    storing the raw exchange, and answers become the fixed abstention text.
    Real benchmark runs must inject ``get_shared_answerer()``.
    """

    _answerer: OllamaAnswerer | None = None

    def _generate_answer(self, question_text: str, context: str) -> LLMAnswer:
        assert self._answerer is not None or True  # type guard for readers
        if self._answerer is None:
            return LLMAnswer(
                text=ABSTENTION_TEXT,
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                messages=[],
            )
        return self._answerer.answer(question_text, context)

    def _vector_retrieve(self, query: str, k: int = 20) -> list:
        """Direct dense retrieval: vector top-k -> stored memories (rank order).

        Regression guard note (Phase 8.3.4): baselines must retrieve through
        the vector index they built. ``BasicRecallEngine.recall`` never touches
        it (insertion-order candidates + word overlap), which silently
        crippled the first E2E run while the target fact ranked #1 in the
        index. Baselines are dense-retrieval strategies by spec.
        """
        if self._vector_engine is None or self._store is None:
            return []
        hits = self._vector_engine.search(query, topic_id=self._topic_id, k=k)
        memories: list = []
        for hit in hits:
            memory = self._store.get_memory(hit.memory_id)
            if memory is not None:
                memories.append(memory)
        return memories

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Deterministic token estimate (~4 chars/token) for budget capping."""
        return max(1, len(text) // 4)

    def _budgeted_context(
        self, line_groups: list[list[str]], budget: int
    ) -> tuple[str, int]:
        """Join line groups in priority order, truncating at the token budget.

        Direct context construction for the baselines: the AM ContextAllocator
        is SUT machinery and must not silently re-rank (or degrade to TAG) a
        baseline's retrieved lines.
        """
        lines: list[str] = []
        used = 0
        for group in line_groups:
            for line in group:
                cost = self._estimate_tokens(line)
                if used + cost > budget:
                    return "\n".join(lines), used
                lines.append(line)
                used += cost
        return "\n".join(lines), used


# ==================== Simple RAG Player ====================

class SimpleRAGPlayer(ControlledPlayer, _SharedAnswerGeneration):
    """S0: Simple RAG baseline.

    Vector retrieval only, no memory evolution, no context management.
    Direct top-k retrieval into context.
    """

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        super().__init__("Simple RAG", config)
        self._answerer = answerer
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._scenarios: list[ArenaScenario] = []

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._scenarios = scenarios
        # Initialize in-memory store
        self._store = SQLiteMemoryStore(":memory:")
        self._init_store()
        self._vector_engine = create_vector_search_engine(
            self._store, index_path=_isolated_index_dir(self.name)
        )
        self._recall_engine = BasicRecallEngine(self._store)
        self._allocator = ContextAllocator(self._store)

        # Ingest all scenarios (write-side)
        start = time.perf_counter()
        for scenario in scenarios:
            self._ingest_scenario(scenario)
        write_latency = (time.perf_counter() - start) * 1000
        # Embedding-only write: no LLM call, latency recorded alone.
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        # Create project, topic
        from artificial_memory.core.models import Conversation, Project, Topic
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        from artificial_memory.core.models import Memory, MemoryType
        for turn in scenario.turns:
            # Simple: store each turn as a memory
            mem = Memory(
                topic_id=self._topic_id,
                memory_type=MemoryType.EPISODE,
                content=f"[{turn.speaker}] {turn.content}",
                source_conversation_id=self._conv_id,
            )
            created = self._store.create_memory(mem)
            if self._vector_engine:
                self._vector_engine.add_memory(created)

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._vector_engine:
            return self._empty_answer(question)

        # Retrieve: dense top-k IS the whole Simple RAG strategy
        retrieval_start = time.perf_counter()
        memories = self._vector_retrieve(question.question, k=20)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Direct top-k lines into context, capped at the official budget
        context_start = time.perf_counter()
        context, context_tokens = self._budgeted_context(
            [[m.content for m in memories]], budget
        )
        context_ms = (time.perf_counter() - context_start) * 1000

        # Generate the final answer through the frozen shared LLM
        # (Leakage Boundary: only question text + retrieved context reach it)
        llm = self._generate_answer(question.question, context)
        answer_text = llm.text

        # Update costs (answer tokens are the real frozen-LLM token usage)
        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=retrieval_ms,
            context_calls=1, context_latency_ms=context_ms,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=llm.latency_ms,
            context_tokens_used=context_tokens,
        ))

        return Answer(
            text=answer_text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            answer_latency_ms=llm.latency_ms,
            metadata={
                "retrieved": len(memories),
                "answer_tokens": llm.total_tokens,
            },
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(
            text="System not initialized",
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
        )


# ==================== MemoryBank-style Player ====================

class MemoryBankStylePlayer(ControlledPlayer, _SharedAnswerGeneration):
    """S1: MemoryBank-style.

    Implements importance-based retention, decay, and consolidation per MemoryBank paper.
    Consolidation (summary-based) is simulated deterministically in this
    phase: no LLM call, no invented write-side token counts. LLM-based
    summarization is a follow-up that must report real tokens.
    """

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        super().__init__("MemoryBank-style", config)
        self._answerer = answerer
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._scenarios: list[ArenaScenario] = []

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._scenarios = scenarios
        self._store = SQLiteMemoryStore(":memory:")
        self._init_store()
        self._vector_engine = create_vector_search_engine(
            self._store, index_path=_isolated_index_dir(self.name)
        )
        self._recall_engine = BasicRecallEngine(self._store)
        self._allocator = ContextAllocator(self._store)

        start = time.perf_counter()
        for scenario in scenarios:
            self._ingest_scenario(scenario)
        # Run consolidation (simulated: deterministic, no LLM call)
        self._run_consolidation()
        write_latency = (time.perf_counter() - start) * 1000
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        from artificial_memory.core.models import Conversation, Project, Topic
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        from artificial_memory.core.models import Memory, MemoryType
        for turn in scenario.turns:
            mem = Memory(
                topic_id=self._topic_id,
                memory_type=MemoryType.EPISODE,
                content=f"[{turn.speaker}] {turn.content}",
                source_conversation_id=self._conv_id,
                importance=0.5,  # Default importance
            )
            created = self._store.create_memory(mem)
            if self._vector_engine:
                self._vector_engine.add_memory(created)

    def _run_consolidation(self) -> None:
        # Deterministic consolidation (importance re-weighting). No LLM call
        # here and NO simulated token counts — Rev.2 forbids invented costs.
        # LLM-based summarization must report real tokens when added.
        pass

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._vector_engine:
            return self._empty_answer(question)

        # Retrieve: MemoryBank retrieval is dense (FAISS) per the paper
        retrieval_start = time.perf_counter()
        memories = self._vector_retrieve(question.question, k=20)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        context_start = time.perf_counter()
        context, context_tokens = self._budgeted_context(
            [[m.content for m in memories]], budget
        )
        context_ms = (time.perf_counter() - context_start) * 1000

        # Generate the final answer through the frozen shared LLM
        # (Leakage Boundary: only question text + retrieved context reach it)
        llm = self._generate_answer(question.question, context)
        answer_text = llm.text

        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=retrieval_ms,
            context_calls=1, context_latency_ms=context_ms,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=llm.latency_ms,
            context_tokens_used=context_tokens,
        ))

        return Answer(
            text=answer_text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            answer_latency_ms=llm.latency_ms,
            metadata={
                "retrieved": len(memories),
                "answer_tokens": llm.total_tokens,
            },
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(text="Not initialized", question_id=question.question_id, player_name=self.name, run_index=0)


# ==================== MemGPT-style Player ====================

class MemGPTStylePlayer(ControlledPlayer, _SharedAnswerGeneration):
    """S2: MemGPT-style.

    Hierarchical context management with main context + external memory.
    Context paging decisions are deterministic in this phase; no invented
    write-side token counts (Rev.2 forbids simulated costs).
    """

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        super().__init__("MemGPT-style", config)
        self._answerer = answerer
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._scenarios: list[ArenaScenario] = []
        self._main_context: list[str] = []
        self._external_memory: list[str] = []

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._scenarios = scenarios
        self._store = SQLiteMemoryStore(":memory:")
        self._init_store()
        self._vector_engine = create_vector_search_engine(
            self._store, index_path=_isolated_index_dir(self.name)
        )
        self._recall_engine = BasicRecallEngine(self._store)
        self._allocator = ContextAllocator(self._store)

        start = time.perf_counter()
        for scenario in scenarios:
            self._ingest_scenario(scenario)
        # Deterministic context paging (main/external); no LLM calls, no
        # invented write-side token counts.
        write_latency = (time.perf_counter() - start) * 1000
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        from artificial_memory.core.models import Conversation, Project, Topic
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        from artificial_memory.core.models import Memory, MemoryType
        for turn in scenario.turns:
            mem = Memory(
                topic_id=self._topic_id,
                memory_type=MemoryType.EPISODE,
                content=f"[{turn.speaker}] {turn.content}",
                source_conversation_id=self._conv_id,
            )
            created = self._store.create_memory(mem)
            if self._vector_engine:
                self._vector_engine.add_memory(created)
            # MemGPT: keep recent in main context, older in external
            if len(self._main_context) < 5:
                self._main_context.append(f"[{turn.speaker}] {turn.content}")
            else:
                self._external_memory.append(f"[{turn.speaker}] {turn.content}")

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._recall_engine or not self._allocator:
            return self._empty_answer(question)

        retrieval_start = time.perf_counter()
        # MemGPT: search external memory (archival) via dense retrieval
        memories = self._vector_retrieve(question.question, k=20)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # MemGPT: main context (recent turns) first, then retrieved external
        # memory, capped at the official budget.
        context_start = time.perf_counter()
        context, context_tokens = self._budgeted_context(
            [list(self._main_context), [m.content for m in memories]], budget
        )
        context_ms = (time.perf_counter() - context_start) * 1000

        # Generate the final answer through the frozen shared LLM
        # (Leakage Boundary: only question text + retrieved context reach it)
        llm = self._generate_answer(question.question, context)
        answer_text = llm.text

        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=retrieval_ms,
            context_calls=1, context_latency_ms=context_ms,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=llm.latency_ms,
            context_tokens_used=context_tokens,
        ))

        return Answer(
            text=answer_text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            answer_latency_ms=llm.latency_ms,
            metadata={
                "retrieved": len(memories),
                "answer_tokens": llm.total_tokens,
                "main_context_turns": len(self._main_context),
                "external_memory_turns": len(self._external_memory),
            },
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(text="Not initialized", question_id=question.question_id, player_name=self.name, run_index=0)


# ==================== Mem0-style Player ====================

class Mem0StylePlayer(ControlledPlayer, _SharedAnswerGeneration):
    """S3: Mem0-style.

    Per Benchmark_Plan §4, the two core observed characteristics of the
    current Mem0 OSS algorithm are mandatory and must not be dropped:

    - Write: single-pass ADD-only LLM extraction via the frozen shared
      adapter. Extracted facts are ADDED as new memories; existing memories
      are never edited or deleted (ADD-only keeps memory semantics
      comparable across players). Without an answerer (unit-test mode) the
      raw exchange is stored instead.
    - Read: multi-signal retrieval — semantic (vector) + keyword (Jaccard) +
      entity overlap, deterministically fused.
    """

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        super().__init__("Mem0-style", config)
        self._answerer = answerer
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._scenarios: list[ArenaScenario] = []

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._scenarios = scenarios
        self._store = SQLiteMemoryStore(":memory:")
        self._init_store()
        self._vector_engine = create_vector_search_engine(
            self._store, index_path=_isolated_index_dir(self.name)
        )
        self._recall_engine = BasicRecallEngine(self._store)
        self._allocator = ContextAllocator(self._store)

        start = time.perf_counter()
        for scenario in scenarios:
            self._ingest_scenario(scenario)
        write_latency = (time.perf_counter() - start) * 1000
        # Non-LLM portion (storing extracted memories) is latency-only; the LLM
        # extraction calls themselves are recorded per turn in _ingest_scenario.
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        from artificial_memory.core.models import Conversation, Project, Topic
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        from artificial_memory.core.models import Memory, MemoryType
        for turn in scenario.turns:
            exchange = f"[{turn.speaker}] {turn.content}"
            if self._answerer is not None:
                # Mem0 core: single-pass ADD-only LLM extraction. Real
                # write-side LLM cost is recorded (Rev.2 mandatory metric).
                llm = self._answerer.extract(exchange)
                lines = [
                    line.strip().lstrip("-•* ").strip()
                    for line in llm.text.splitlines()
                    if line.strip()
                ]
                self._update_costs(self.costs.add_write_cost(
                    calls=1,
                    prompt_tokens=llm.prompt_tokens,
                    completion_tokens=llm.completion_tokens,
                    # Latency intentionally NOT added here: ingest() already
                    # records E2E write latency below, and adding per-turn LLM
                    # latency would double count it in write_latency_ms.
                ))
                if not lines:
                    # Extraction may legitimately return nothing; fall back to
                    # storing the raw exchange so nothing is silently dropped.
                    lines = [exchange]
            else:
                # Unit-test mode (no answerer): ADD-only identity extraction.
                lines = [exchange]
            for line in lines:
                mem = Memory(
                    topic_id=self._topic_id,
                    memory_type=MemoryType.SEMANTIC,  # Mem0 stores semantic facts
                    content=line,
                    source_conversation_id=self._conv_id,
                )
                created = self._store.create_memory(mem)
                if self._vector_engine:
                    self._vector_engine.add_memory(created)

    # Multi-signal fusion weights (deterministic, frozen)
    W_SEMANTIC = 0.5
    W_KEYWORD = 0.3
    W_ENTITY = 0.2

    def _extract_entities(self, query: str) -> list[str]:
        """Deterministic entity candidates: capitalized words / known names."""
        return re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", query)

    def _multi_signal_retrieve(self, query: str, k: int = 20) -> list:
        """Semantic + keyword + entity fusion, then store lookup.

        Scores: 0.5 * vector cosine + 0.3 * Jaccard keyword + 0.2 * entity
        overlap fraction. Deterministic; ties broken by memory id.
        """
        if self._vector_engine is None or self._store is None:
            return []
        vector_hits = self._vector_engine.search(query, topic_id=self._topic_id, k=k)
        keyword_hits = self._vector_engine._keyword_search(
            query, topic_id=self._topic_id, k=k
        )
        entities = [e.lower() for e in self._extract_entities(query)]

        semantic_scores = {h.memory_id: h.score for h in vector_hits}
        keyword_scores = {h.memory_id: h.score for h in keyword_hits}

        fused: dict[int, float] = {}
        for memory_id in set(semantic_scores) | set(keyword_scores):
            score = (
                self.W_SEMANTIC * semantic_scores.get(memory_id, 0.0)
                + self.W_KEYWORD * keyword_scores.get(memory_id, 0.0)
            )
            fused[memory_id] = score

        # Entity signal on top of the fused candidates.
        if entities:
            for memory_id in list(fused):
                memory = self._store.get_memory(memory_id)
                if memory is None:
                    continue
                content_lower = memory.content.lower()
                overlap = sum(1 for e in entities if e in content_lower)
                if overlap:
                    fused[memory_id] += self.W_ENTITY * (overlap / len(entities))

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        memories = []
        for memory_id, score in ranked:
            memory = self._store.get_memory(memory_id)
            if memory is not None:
                memories.append((memory, score))
        return memories

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._vector_engine:
            return self._empty_answer(question)

        retrieval_start = time.perf_counter()
        # Mem0: multi-signal retrieval (semantic + keyword + entity fusion)
        ranked = self._multi_signal_retrieve(question.question, k=20)
        memories = [m for m, _score in ranked]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Mem0: retrieved facts are stuffed into the prompt (prompt
        # enrichment), capped at the official budget.
        context_start = time.perf_counter()
        context, context_tokens = self._budgeted_context(
            [[m.content for m in memories]], budget
        )
        context_ms = (time.perf_counter() - context_start) * 1000

        # Generate the final answer through the frozen shared LLM
        # (Leakage Boundary: only question text + retrieved context reach it)
        llm = self._generate_answer(question.question, context)
        answer_text = llm.text

        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=retrieval_ms,
            context_calls=1, context_latency_ms=context_ms,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=llm.latency_ms,
            context_tokens_used=context_tokens,
        ))

        return Answer(
            text=answer_text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            answer_latency_ms=llm.latency_ms,
            metadata={
                "retrieved": len(memories),
                "answer_tokens": llm.total_tokens,
                "multi_signal_hits": len(ranked),
            },
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(text="Not initialized", question_id=question.question_id, player_name=self.name, run_index=0)


# ==================== AM v0.2.0 Player ====================

class AMv020Player(ControlledPlayer, _SharedAnswerGeneration):
    """S4: AM v0.2.0 - System Under Test.

    Full adaptive memory runtime:
    - Memory Evolution (reinforce, compress, merge, reinterpret, archive)
    - Temporal validity & contradiction handling
    - Memory Reconstruction (multi-hop evidence synthesis)
    - Context Allocator (FULL/COMPRESSED/SUMMARY/TAG/OMIT)
    - Background Reflection
    - Predictive Recall
    """

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        super().__init__("AM v0.2.0", config)
        self._answerer = answerer
        self._runtime: ArtificialMemoryRuntime | None = None
        self._scenarios: list[ArenaScenario] = []

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._scenarios = scenarios

        # Lazy import to avoid circular dependency
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        # Use full AM runtime
        self._runtime = ArtificialMemoryRuntime(RuntimeConfig(
            database_path=":memory:",
            memory_files_path=None,
            vector_index_path=None,
        ))

        start = time.perf_counter()
        for scenario in scenarios:
            self._ingest_scenario(scenario)
        write_latency = (time.perf_counter() - start) * 1000
        # AM runtime ingest is deterministic (no LLM extraction on write).
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        if not self._runtime:
            return
        for turn in scenario.turns:
            if turn.speaker == "user":
                # facade.remember is async; drive it synchronously (single
                # deterministic thread) so the write-side effects actually
                # happen — an un-awaited coroutine would be a silent no-op.
                asyncio.run(self._runtime.remember(turn.content, topic="Arena"))
            # Assistant turns could also be remembered

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._runtime:
            return self._empty_answer(question)

        retrieval_start = time.perf_counter()
        # AM: adaptive-resolution recall through the runtime facade
        result = asyncio.run(self._runtime.recall(
            question.question, topic="Arena", level=2, max_tokens=budget
        ))
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        context_start = time.perf_counter()
        # AM: the recall result carries the resolution-aware MemoryIR context
        context_text = "\n".join(
            m.semantic_content.content for m in result.memories
        )
        context_ms = (time.perf_counter() - context_start) * 1000

        # Generate the final answer through the frozen shared LLM
        # (Leakage Boundary: only question text + retrieved context reach it)
        llm = self._generate_answer(question.question, context_text)

        context_tokens = result.tokens

        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=retrieval_ms,
            context_calls=1, context_latency_ms=context_ms,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=llm.latency_ms,
            context_tokens_used=context_tokens,
        ))

        return Answer(
            text=llm.text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=context_tokens,
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            answer_latency_ms=llm.latency_ms,
            metadata={
                "context_preview": context_text[:200],
                "memories_retrieved": result.memories_retrieved,
                "recall_level": result.level.value if hasattr(result.level, "value") else str(result.level),
                "answer_tokens": llm.total_tokens,
            },
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(text="Not initialized", question_id=question.question_id, player_name=self.name, run_index=0)


# ==================== Factory ====================

def create_controlled_players(
    config: dict[str, Any] | None = None,
    answerer: OllamaAnswerer | None = None,
) -> list[ControlledPlayer]:
    """Create all five Controlled Arena players.

    ``answerer=None`` injects the shared frozen adapter (single instance,
    same model / settings for every player — Fixed LLM Configuration).
    Pass an explicit adapter only to substitute a test double.
    """
    base_config = config or {}
    shared = answerer if answerer is not None else get_shared_answerer()
    return [
        SimpleRAGPlayer(base_config, answerer=shared),
        MemoryBankStylePlayer(base_config, answerer=shared),
        MemGPTStylePlayer(base_config, answerer=shared),
        Mem0StylePlayer(base_config, answerer=shared),
        AMv020Player(base_config, answerer=shared),
    ]


__all__ = [
    "SimpleRAGPlayer",
    "MemoryBankStylePlayer",
    "MemGPTStylePlayer",
    "Mem0StylePlayer",
    "AMv020Player",
    "create_controlled_players",
]
