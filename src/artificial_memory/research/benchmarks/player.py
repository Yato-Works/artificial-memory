"""Player Protocol for Memory Arena (Phase 8.1).

Defines the interface that all benchmark subjects must implement.
Controlled Arena players implement the full protocol with budget enforcement.
Real-System Arena players wrap existing implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.research.benchmarks.arena import ArenaQuestion, ArenaScenario


def majority_index(texts: list[str] | tuple[str, ...]) -> int:
    """Index of the majority answer text, with a deterministic tie-break.

    Repeat Protocol (§9): answer runs are reduced by *exact text* majority.
    Ties are broken by the smallest index (i.e. the earliest repeat), which
    keeps the reduction deterministic and independent of dict ordering, so the
    runner and the Scorer always agree on which answer is the official one.
    """
    if not texts:
        return -1
    counts = Counter(texts)
    best = max(counts.values())
    for index, text in enumerate(texts):
        if counts[text] == best:
            return index
    return 0  # pragma: no cover - Counter guarantees a winner above


@dataclass
class Answer:
    """A single answer from a player.

    The text is the full last assistant message (per Answer Extraction Protocol).
    """
    text: str
    question_id: str
    player_name: str
    run_index: int
    timestamp: datetime = field(default_factory=datetime.now)
    context_tokens_used: int = 0
    retrieval_latency_ms: float = 0.0
    context_build_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "question_id": self.question_id,
            "player_name": self.player_name,
            "run_index": self.run_index,
            "timestamp": self.timestamp.isoformat(),
            "context_tokens_used": self.context_tokens_used,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "context_build_latency_ms": self.context_build_latency_ms,
            "answer_latency_ms": self.answer_latency_ms,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CostReport:
    """Complete cost accounting for a player run (Write-side + Read-side).

    Per Benchmark_Plan.txt §6: Write-side LLM cost is mandatory.
    Total Cost = Write LLM + Retrieval + Context Construction + Answer LLM
    """
    player_name: str

    # Write-side costs (memory formation)
    write_llm_calls: int = 0
    write_llm_prompt_tokens: int = 0
    write_llm_completion_tokens: int = 0
    write_latency_ms: float = 0.0

    # Read-side costs (retrieval + answer)
    retrieval_calls: int = 0
    retrieval_latency_ms: float = 0.0
    context_construction_calls: int = 0
    context_construction_latency_ms: float = 0.0
    answer_llm_calls: int = 0
    answer_llm_prompt_tokens: int = 0
    answer_llm_completion_tokens: int = 0
    answer_latency_ms: float = 0.0

    # Aggregate
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0

    # Context budget (Controlled Arena)
    context_budget_tokens: int = 2000
    context_tokens_used: int = 0
    budget_violations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'total_llm_calls',
            self.write_llm_calls + self.retrieval_calls + self.context_construction_calls + self.answer_llm_calls)
        object.__setattr__(self, 'total_prompt_tokens',
            self.write_llm_prompt_tokens + self.answer_llm_prompt_tokens)
        object.__setattr__(self, 'total_completion_tokens',
            self.write_llm_completion_tokens + self.answer_llm_completion_tokens)
        object.__setattr__(self, 'total_tokens',
            self.total_prompt_tokens + self.total_completion_tokens)
        object.__setattr__(self, 'total_latency_ms',
            self.write_latency_ms + self.retrieval_latency_ms +
            self.context_construction_latency_ms + self.answer_latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "write_llm_calls": self.write_llm_calls,
            "write_llm_prompt_tokens": self.write_llm_prompt_tokens,
            "write_llm_completion_tokens": self.write_llm_completion_tokens,
            "write_latency_ms": self.write_latency_ms,
            "retrieval_calls": self.retrieval_calls,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "context_construction_calls": self.context_construction_calls,
            "context_construction_latency_ms": self.context_construction_latency_ms,
            "answer_llm_calls": self.answer_llm_calls,
            "answer_llm_prompt_tokens": self.answer_llm_prompt_tokens,
            "answer_llm_completion_tokens": self.answer_llm_completion_tokens,
            "answer_latency_ms": self.answer_latency_ms,
            "total_llm_calls": self.total_llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "context_budget_tokens": self.context_budget_tokens,
            "context_tokens_used": self.context_tokens_used,
            "budget_violations": self.budget_violations,
        }

    @classmethod
    def empty(cls, player_name: str, context_budget: int = 2000) -> CostReport:
        return cls(player_name=player_name, context_budget_tokens=context_budget)

    def with_budget_violation(self, context_tokens_used: int) -> CostReport:
        """Record a Controlled-Arena budget violation (no cost added).

        Used by ``ControlledPlayer.answer`` so enforcement is independent of
        how (or whether) a player records its own read costs.
        """
        return CostReport(
            player_name=self.player_name,
            write_llm_calls=self.write_llm_calls,
            write_llm_prompt_tokens=self.write_llm_prompt_tokens,
            write_llm_completion_tokens=self.write_llm_completion_tokens,
            write_latency_ms=self.write_latency_ms,
            retrieval_calls=self.retrieval_calls,
            retrieval_latency_ms=self.retrieval_latency_ms,
            context_construction_calls=self.context_construction_calls,
            context_construction_latency_ms=self.context_construction_latency_ms,
            answer_llm_calls=self.answer_llm_calls,
            answer_llm_prompt_tokens=self.answer_llm_prompt_tokens,
            answer_llm_completion_tokens=self.answer_llm_completion_tokens,
            answer_latency_ms=self.answer_latency_ms,
            context_budget_tokens=self.context_budget_tokens,
            context_tokens_used=self.context_tokens_used,
            budget_violations=self.budget_violations + 1,
        )

    def add_write_cost(self, calls: int = 0, prompt_tokens: int = 0,
                       completion_tokens: int = 0, latency_ms: float = 0.0) -> CostReport:
        """Accumulate write-side cost.

        ``calls`` counts *LLM* write calls only. Non-LLM writes (embedding-only
        ingest, deterministic consolidation / context paging) MUST pass
        ``calls=0`` and record latency alone, so that ``write_llm_calls`` stays
        a truthful measure of memory-formation LLM usage
        (Benchmark_Plan.txt §6 Write-side Cost).
        """
        return CostReport(
            player_name=self.player_name,
            write_llm_calls=self.write_llm_calls + calls,
            write_llm_prompt_tokens=self.write_llm_prompt_tokens + prompt_tokens,
            write_llm_completion_tokens=self.write_llm_completion_tokens + completion_tokens,
            write_latency_ms=self.write_latency_ms + latency_ms,
            retrieval_calls=self.retrieval_calls,
            retrieval_latency_ms=self.retrieval_latency_ms,
            context_construction_calls=self.context_construction_calls,
            context_construction_latency_ms=self.context_construction_latency_ms,
            answer_llm_calls=self.answer_llm_calls,
            answer_llm_prompt_tokens=self.answer_llm_prompt_tokens,
            answer_llm_completion_tokens=self.answer_llm_completion_tokens,
            answer_latency_ms=self.answer_latency_ms,
            context_budget_tokens=self.context_budget_tokens,
            context_tokens_used=self.context_tokens_used,
            budget_violations=self.budget_violations,
        )

    def add_read_cost(self, retrieval_calls: int = 0, retrieval_latency_ms: float = 0.0,
                      context_calls: int = 0, context_latency_ms: float = 0.0,
                      answer_calls: int = 0, answer_prompt_tokens: int = 0,
                      answer_completion_tokens: int = 0, answer_latency_ms: float = 0.0,
                      context_tokens_used: int = 0) -> CostReport:
        violations = self.budget_violations
        if context_tokens_used > self.context_budget_tokens:
            violations += 1
        return CostReport(
            player_name=self.player_name,
            write_llm_calls=self.write_llm_calls,
            write_llm_prompt_tokens=self.write_llm_prompt_tokens,
            write_llm_completion_tokens=self.write_llm_completion_tokens,
            write_latency_ms=self.write_latency_ms,
            retrieval_calls=self.retrieval_calls + retrieval_calls,
            retrieval_latency_ms=self.retrieval_latency_ms + retrieval_latency_ms,
            context_construction_calls=self.context_construction_calls + context_calls,
            context_construction_latency_ms=self.context_construction_latency_ms + context_latency_ms,
            answer_llm_calls=self.answer_llm_calls + answer_calls,
            answer_llm_prompt_tokens=self.answer_llm_prompt_tokens + answer_prompt_tokens,
            answer_llm_completion_tokens=self.answer_llm_completion_tokens + answer_completion_tokens,
            answer_latency_ms=self.answer_latency_ms + answer_latency_ms,
            context_budget_tokens=self.context_budget_tokens,
            context_tokens_used=self.context_tokens_used + context_tokens_used,
            budget_violations=violations,
        )


class Player(ABC):
    """Abstract base class for all benchmark players.

    Controlled Arena players:
    - Implement full protocol with 2,000 token budget enforcement
    - Use fixed extraction LLM (same as answer LLM per Fairness Rules)
    - Report write-side costs

    Real-System Arena players:
    - Wrap existing implementations (Mem0 OSS, Letta, etc.)
    - Native context management (no budget enforcement)
    - Still report actual token usage
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self._costs = CostReport.empty(name, self.config.get('context_budget', 2000))
        self._initialized = False

    @property
    def costs(self) -> CostReport:
        return self._costs

    def _update_costs(self, new_costs: CostReport) -> None:
        self._costs = new_costs

    @abstractmethod
    def ingest(self, scenarios: list[ArenaScenario]) -> None:
        """Ingest the shared conversation history.

        Called once per run before any questions are asked.
        This is where memory formation / write-side LLM calls happen.
        """
        pass

    @abstractmethod
    def answer(self, question: ArenaQuestion, context_budget: int) -> Answer:
        """Answer a single question within the context budget.

        For Controlled Arena: context_budget = 2,000 (fixed)
        For Real-System Arena: context_budget is advisory; native behavior used

        Returns Answer with full text (last assistant message) and token/latency metrics.
        """
        pass

    def finalize(self) -> CostReport:
        """Called after all questions; return final cost report."""
        return self._costs

    def reset_costs(self) -> None:
        """Reset cost tracking for a new repeat run."""
        self._costs = CostReport.empty(self.name, self.config.get('context_budget', 2000))


class ControlledPlayer(Player):
    """Base class for Controlled Arena players.

    Enforces:
    - Fixed 2,000 token context budget (budget violations recorded)
    - Fixed extraction LLM (configured via config)
    - Write-side cost tracking
    """
    ARENA_CLASS = "controlled"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self.context_budget = self.config.get('context_budget', 2000)
        self.extraction_llm = self.config.get('extraction_llm', self.config.get('llm', 'local'))

    def answer(self, question: ArenaQuestion, context_budget: int) -> Answer:
        # Controlled Arena always uses the fixed budget
        answer = self._answer_with_budget(question, self.context_budget)
        # Budget enforcement lives here, independent of how a player records
        # its own costs: an over-budget context is a protocol violation of the
        # Controlled Arena (§5) and is recorded as such.
        if answer.context_tokens_used > self.context_budget:
            self._update_costs(
                self.costs.with_budget_violation(answer.context_tokens_used)
            )
        return answer

    @abstractmethod
    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        """Implement answer logic with explicit budget."""
        pass


class RealSystemPlayer(Player):
    """Base class for Real-System Arena players.

    No budget enforcement - native behavior.
    Still measures actual token usage.
    """
    ARENA_CLASS = "real"

    def answer(self, question: ArenaQuestion, context_budget: int) -> Answer:
        # Real Arena: budget is advisory only; native behavior
        return self._answer_native(question)

    @abstractmethod
    def _answer_native(self, question: ArenaQuestion) -> Answer:
        """Implement answer logic with native context management."""
        pass


__all__ = [
    "Answer",
    "CostReport",
    "Player",
    "ControlledPlayer",
    "RealSystemPlayer",
]
