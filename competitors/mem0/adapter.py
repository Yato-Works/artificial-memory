"""Mem0 Controlled Player Adapter (Phase 1).

Loads parameters directly from benchmark_config/mem0.yaml and implements
the ControlledPlayer protocol.
"""

from __future__ import annotations

import re
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


def load_mem0_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent.parent / "benchmark_config" / "mem0.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


class Mem0Adapter(ControlledPlayer, _SharedAnswerGeneration):
    """Mem0 Controlled Arena Player governed by benchmark_config/mem0.yaml."""

    def __init__(self, config: dict[str, Any] | None = None, answerer: OllamaAnswerer | None = None):
        yaml_cfg = load_mem0_config()
        merged_cfg = dict(yaml_cfg)
        if config:
            merged_cfg.update(config)

        super().__init__(
            merged_cfg.get("player", {}).get("name", "Mem0-style"),
            merged_cfg,
        )
        self.yaml_config = merged_cfg
        self._answerer = answerer or get_shared_answerer()
        self._store: MemoryStore | None = None
        self._vector_engine: VectorSearchEngine | None = None
        self._recall_engine: BasicRecallEngine | None = None
        self._allocator: ContextAllocator | None = None
        self._scenarios: list[ArenaScenario] = []

        # Weights from YAML config
        retrieval_cfg = self.yaml_config.get("retrieval", {})
        weights = retrieval_cfg.get("weights", {})
        self.w_semantic = weights.get("semantic_similarity", 0.5)
        self.w_keyword = weights.get("keyword_jaccard", 0.3)
        self.w_entity = weights.get("entity_overlap", 0.2)
        self.top_k = retrieval_cfg.get("top_k", 10)

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
        self._update_costs(self.costs.add_write_cost(calls=0, latency_ms=write_latency))

    def _init_store(self) -> None:
        project = self._store.create_project(Project(name="arena"))
        topic = self._store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
        self._topic_id = topic.id
        self._conv_id = self._store.create_conversation(Conversation(topic_id=topic.id, title="Arena")).id

    def _ingest_scenario(self, scenario: ArenaScenario) -> None:
        for turn in scenario.turns:
            exchange = f"[{turn.speaker}] {turn.content}"
            if self._answerer is not None:
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
                ))
                if not lines:
                    lines = [exchange]
            else:
                lines = [exchange]

            for line in lines:
                mem = Memory(
                    topic_id=self._topic_id,
                    memory_type=MemoryType.SEMANTIC,
                    content=line,
                    source_conversation_id=self._conv_id,
                )
                created = self._store.create_memory(mem)
                if self._vector_engine:
                    self._vector_engine.add_memory(created)

    def _extract_entities(self, query: str) -> list[str]:
        return re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", query)

    def _multi_signal_retrieve(self, query: str, k: int = 10) -> list:
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
                self.w_semantic * semantic_scores.get(memory_id, 0.0)
                + self.w_keyword * keyword_scores.get(memory_id, 0.0)
            )
            fused[memory_id] = score

        if entities:
            for memory_id in list(fused):
                memory = self._store.get_memory(memory_id)
                if memory is None:
                    continue
                content_lower = memory.content.lower()
                overlap = sum(1 for e in entities if e in content_lower)
                if overlap:
                    fused[memory_id] += self.w_entity * (overlap / len(entities))

        ranked_ids = sorted(fused, key=lambda mid: (fused[mid], -mid), reverse=True)[:k]
        result = []
        for mid in ranked_ids:
            mem = self._store.get_memory(mid)
            if mem:
                result.append(mem)
        return result

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        if not self._store or not self._vector_engine:
            return self._empty_answer(question)

        start = time.perf_counter()
        memories = self._multi_signal_retrieve(question.question, k=self.top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        cstart = time.perf_counter()
        parts = [m.content for m in memories]
        context_str = "\n".join(parts)
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
