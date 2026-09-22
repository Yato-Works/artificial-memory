"""MemoryBank Controlled Player Adapter (Phase 1).

Loads parameters directly from benchmark_config/memorybank.yaml and implements
the ControlledPlayer protocol with Ebbinghaus forgetting curve.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import yaml

from artificial_memory.context.allocator import ContextAllocator
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Conversation, Memory, MemoryType, Project, Topic
from artificial_memory.memory.vector_search import VectorSearchEngine, create_vector_search_engine
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.research.benchmarks.arena import ArenaQuestion, ArenaScenario
from artificial_memory.research.benchmarks.llm import LLMAnswer, OllamaAnswerer
from artificial_memory.research.benchmarks.player import Answer, ControlledPlayer
from artificial_memory.research.benchmarks.players import (
    _SharedAnswerGeneration,
    _isolated_index_dir,
    get_shared_answerer,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


def load_memorybank_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent.parent / "benchmark_config" / "memorybank.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


class MemoryBankAdapter(ControlledPlayer, _SharedAnswerGeneration):
    """MemoryBank Controlled Arena Player governed by benchmark_config/memorybank.yaml."""

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        yaml_cfg = load_memorybank_config()
        merged_cfg = dict(yaml_cfg)
        if config:
            merged_cfg.update(config)

        super().__init__(
            merged_cfg.get("player", {}).get("name", "MemoryBank-style"),
            merged_cfg,
        )
        self.yaml_config = merged_cfg
        self._answerer = answerer or get_shared_answerer()
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._timestamps: dict[int, float] = {}

        params = self.yaml_config.get("parameters", {})
        self.alpha = params.get("alpha_similarity_weight", 0.6)
        self.decay_factor = params.get("decay_factor", 0.95)
        self.top_k = self.yaml_config.get("retrieval", {}).get("top_k", 10)

    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        self._store = SQLiteMemoryStore(":memory:")
        self._init_store()
        self._vector_engine = create_vector_search_engine(
            self._store, index_path=_isolated_index_dir(self.name)
        )
        self._recall_engine = BasicRecallEngine(self._store)
        self._allocator = ContextAllocator(self._store)

        start = time.perf_counter()
        t = 0.0
        for scenario in scenarios:
            for turn in scenario.turns:
                line = f"[{turn.speaker}] {turn.content}"
                mem = Memory(
                    topic_id=self._topic_id,
                    memory_type=MemoryType.EPISODE,
                    content=line,
                    source_conversation_id=self._conv_id,
                )
                created = self._store.create_memory(mem)
                self._timestamps[created.id] = t
                t += 1.0
                if self._vector_engine:
                    self._vector_engine.add_memory(created)

        self._current_time = t
        write_latency = (time.perf_counter() - start) * 1000
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ebbinghaus_retention(self, memory_id: int) -> float:
        created_at = self._timestamps.get(memory_id, 0.0)
        delta_t = max(0.0, self._current_time - created_at)
        return math.exp(-delta_t * (1.0 - self.decay_factor))

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._store or not self._vector_engine:
            return self._empty_answer(question)

        start = time.perf_counter()
        hits = self._vector_engine.search(question.question, topic_id=self._topic_id, k=self.top_k * 2)

        rescored = []
        for h in hits:
            retention = self._ebbinghaus_retention(h.memory_id)
            composite = self.alpha * h.score + (1.0 - self.alpha) * retention
            rescored.append((composite, h.memory_id))

        rescored.sort(reverse=True)
        top_ids = [mid for _, mid in rescored[:self.top_k]]
        memories = [self._store.get_memory(mid) for mid in top_ids]
        memories = [m for m in memories if m is not None]
        retrieval_ms = (time.perf_counter() - start) * 1000

        cstart = time.perf_counter()
        context_str = "\n".join(m.content for m in memories)
        context_tokens = len(context_str) // 4
        context_ms = (time.perf_counter() - cstart) * 1000

        llm = self._generate_answer(question.question, context_str)
        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1,
            retrieval_latency_ms=retrieval_ms,
            context_calls=1,
            context_latency_ms=context_ms,
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
            metadata={"retrieved": len(memories), "answer_tokens": llm.total_tokens},
        )

    def _empty_answer(self, question: ArenaQuestion) -> Answer:
        return Answer(text="Not initialized", question_id=question.question_id, player_name=self.name, run_index=0)
