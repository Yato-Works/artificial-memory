from __future__ import annotations

import tiktoken

from artificial_memory.core.interfaces import ContextBuilder, MemoryStore
from artificial_memory.core.ir import (
    BudgetAllocation,
    ContextIR,
    ContextPart,
    ContextProvenance,
    ContextStats,
    PriorityTier,
)
from artificial_memory.core.models import Memory, RecallLevel
from artificial_memory.recall.engine import BasicRecallEngine


class TokenCounter:
    """Token counter using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens for chat messages (OpenAI format)."""
        total = 0
        for msg in messages:
            total += 4  # message overhead
            for key, value in msg.items():
                total += self.count(str(value))
        total += 2  # reply priming
        return total


class BudgetManager:
    """Manages context token budget with priority tiers."""

    DEFAULT_RATIOS = {
        PriorityTier.HIGH: 0.50,
        PriorityTier.MEDIUM: 0.35,
        PriorityTier.LOW: 0.15,
    }

    def __init__(self, max_tokens: int, ratios: dict[PriorityTier, float] | None = None):
        self.max_tokens = max_tokens
        self.ratios = ratios or self.DEFAULT_RATIOS
        self.allocation = self._calculate_allocation()
        self.used = {tier: 0 for tier in PriorityTier}

    def _calculate_allocation(self) -> BudgetAllocation:
        allocation = BudgetAllocation()
        for tier, ratio in self.ratios.items():
            setattr(allocation, tier.value, int(self.max_tokens * ratio))
        return allocation

    def can_fit(self, tier: PriorityTier, tokens: int) -> bool:
        available = self.allocation.get_for_tier(tier) - self.used[tier]
        return tokens <= available

    def allocate(self, tier: PriorityTier, tokens: int) -> bool:
        if self.can_fit(tier, tokens):
            self.used[tier] += tokens
            return True
        return False

    def get_remaining(self, tier: PriorityTier) -> int:
        return self.allocation.get_for_tier(tier) - self.used[tier]

    def get_total_remaining(self) -> int:
        return sum(self.get_remaining(tier) for tier in PriorityTier)

    def get_usage_stats(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "allocation": {tier.value: self.allocation.get_for_tier(tier) for tier in PriorityTier},
            "used": {tier.value: self.used[tier] for tier in PriorityTier},
            "remaining": {tier.value: self.get_remaining(tier) for tier in PriorityTier},
            "total_used": sum(self.used.values()),
            "total_remaining": self.get_total_remaining(),
        }


class EnhancedContextBuilder:
    """Enhanced context builder with priority-based budget management."""

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: BasicRecallEngine,
        encoding_name: str = "cl100k_base",
        budget_ratios: dict[PriorityTier, float] | None = None,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.token_counter = TokenCounter(encoding_name)
        self.budget_ratios = budget_ratios
        self._stats = ContextStats()

    def _determine_tier(self, priority: float) -> PriorityTier:
        if priority >= 0.7:
            return PriorityTier.HIGH
        elif priority >= 0.3:
            return PriorityTier.MEDIUM
        return PriorityTier.LOW

    def _get_memory_priority(self, memory: Memory, base_priority: float) -> float:
        return base_priority * memory.importance * memory.confidence

    def _build_context_parts(
        self,
        query: str,
        topic_id: int | None,
        max_tokens: int,
        current_memories: list[Memory] | None,
    ) -> tuple[list[ContextPart], BudgetManager, ContextStats]:
        """Internal method to build context parts and return raw data."""
        budget = BudgetManager(max_tokens, self.budget_ratios)
        parts = []

        self._stats = ContextStats()
        self._stats.budget_allocation = budget.allocation

        # 1. Current state memories (HIGH priority)
        if current_memories:
            for mem in current_memories:
                if mem.is_current and mem.status.value == "active":
                    tokens = self.token_counter.count(mem.content)
                    priority = 1.0
                    tier = PriorityTier.HIGH

                    if budget.allocate(tier, tokens):
                        parts.append(ContextPart(
                            content=mem.content,
                            tokens=tokens,
                            priority=priority,
                            source=f"current:{mem.memory_type.value}",
                            tier=tier,
                            memory_id=mem.id,
                        ))
                        self._stats.parts_count += 1
                        self._stats.tier_distribution[tier.value] = \
                            self._stats.tier_distribution.get(tier.value, 0) + 1
                    else:
                        self._try_lower_tiers(budget, mem, tokens, 0.9)

        # 2. Recall relevant memories at progressive levels
        recall_configs = [
            (RecallLevel.CURRENT_ONLY, 0.9, PriorityTier.HIGH),
            (RecallLevel.LONG_TERM_SUMMARY, 0.7, PriorityTier.HIGH),
            (RecallLevel.EPISODE, 0.5, PriorityTier.MEDIUM),
            (RecallLevel.LIGHT_COMPRESSION, 0.3, PriorityTier.MEDIUM),
        ]

        for level, base_priority, default_tier in recall_configs:
            memories, _ = self.recall_engine.recall(query, topic_id, level, max_tokens=max_tokens // 2)
            for mem in memories:
                if current_memories and any(m.id == mem.id for m in current_memories):
                    continue

                tokens = self.token_counter.count(mem.content)
                priority = self._get_memory_priority(mem, base_priority)
                tier = self._determine_tier(priority)

                if budget.allocate(tier, tokens):
                    parts.append(ContextPart(
                        content=mem.content,
                        tokens=tokens,
                        priority=priority,
                        source=f"recall:{level.name}:{mem.memory_type.value}",
                        tier=tier,
                        memory_id=mem.id,
                    ))
                    self._stats.parts_count += 1
                    self._stats.tier_distribution[tier.value] = \
                        self._stats.tier_distribution.get(tier.value, 0) + 1
                else:
                    self._try_lower_tiers(budget, mem, tokens, priority)

        # 3. Sort by priority
        parts.sort(key=lambda p: (-p.priority, p.tier.value))

        return parts, budget, self._stats

    def _try_lower_tiers(self, budget: BudgetManager, mem: Memory, tokens: int, priority: float):
        for tier in [PriorityTier.MEDIUM, PriorityTier.LOW]:
            priority * (0.7 if tier == PriorityTier.MEDIUM else 0.3)
            if budget.allocate(tier, tokens):
                return True
        return False

    def _select_within_budget(
        self,
        parts: list[ContextPart],
        max_tokens: int
    ) -> tuple[list[str], ContextStats, int]:
        """Select parts within token budget."""
        selected = []
        total_tokens = 0

        for part in parts:
            if total_tokens + part.tokens <= max_tokens:
                selected.append(part.content)
                total_tokens += part.tokens
                self._stats.selected_parts += 1
                self._stats.priority_distribution[part.tier.value] = \
                    self._stats.priority_distribution.get(part.tier.value, 0) + 1
            else:
                remaining = max_tokens - total_tokens
                if remaining > 100 and part.tokens > remaining:
                    char_budget = remaining * 3
                    if char_budget > 100:
                        truncated_content = part.content[:char_budget] + "... [truncated]"
                        selected.append(truncated_content)
                        total_tokens += self.token_counter.count(truncated_content)
                        self._stats.truncated_parts += 1
                        self._stats.selected_parts += 1
                break

        # Update final stats
        raw_tokens = sum(p.tokens for p in parts)
        effective_tokens = self.token_counter.count("\n\n---\n\n".join(selected))

        self._stats.raw_tokens = raw_tokens
        self._stats.effective_tokens = effective_tokens
        self._stats.compression_ratio = raw_tokens / effective_tokens if effective_tokens > 0 else 0.0
        self._stats.parts_count = len(parts)

        return selected, self._stats, total_tokens

    def build_context(
        self,
        query: str,
        topic_id: int | None = None,
        max_tokens: int = 8000,
        current_memories: list[Memory] | None = None,
    ) -> str:
        """Build optimized context for LLM (backward compatible - returns string)."""
        parts, budget, stats = self._build_context_parts(query, topic_id, max_tokens, current_memories)
        selected, _, _ = self._select_within_budget(parts, max_tokens)
        return "\n\n---\n\n".join(selected)

    def build_context_ir(
        self,
        query: str,
        topic_id: int | None = None,
        max_tokens: int = 8000,
        current_memories: list[Memory] | None = None,
    ) -> ContextIR:
        """Build optimized context and return ContextIR with full provenance."""
        parts, budget, stats = self._build_context_parts(query, topic_id, max_tokens, current_memories)
        selected, final_stats, total_tokens = self._select_within_budget(parts, max_tokens)

        # Build provenance
        provenance = ContextProvenance(
            parts_provenance=[
                {
                    "memory_id": p.memory_id,
                    "source": p.source,
                    "tier": p.tier.value,
                    "priority": p.priority,
                    "tokens": p.tokens,
                    "selected": p.content in selected,
                    "selection_reason": f"priority={p.priority:.2f}, tier={p.tier.value}"
                }
                for p in parts
            ],
            total_memories_considered=len(parts),
            total_memories_selected=len(selected),
        )

        # Build system prompt
        system_prompt = self.build_system_prompt("General")

        # Extract BudgetAllocation from BudgetManager
        budget_allocation = budget.allocation

        context_ir = ContextIR(
            query=query,
            budget=budget_allocation,
            parts=parts,
            stats=final_stats,
            provenance=provenance,
            system_prompt=system_prompt,
        )

        return context_ir

    def optimize_context(
        self,
        context_parts: list[tuple[str, int, float]],
        max_tokens: int,
    ) -> str:
        """Legacy optimize_context for compatibility."""
        sorted_parts = sorted(context_parts, key=lambda x: -x[2])

        selected = []
        total_tokens = 0

        for content, tokens, priority in sorted_parts:
            if total_tokens + tokens <= max_tokens:
                selected.append(content)
                total_tokens += tokens
            else:
                if tokens > max_tokens * 0.3 and total_tokens < max_tokens * 0.5:
                    remaining = max_tokens - total_tokens
                    char_budget = remaining * 3
                    if char_budget > 100:
                        truncated = content[:char_budget] + "... [truncated]"
                        selected.append(truncated)
                        total_tokens += self.token_counter.count(truncated)
                break

        return "\n\n---\n\n".join(selected)

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def get_context_stats(self) -> ContextStats:
        return self._stats

    def get_budget_usage(self) -> dict:
        if hasattr(self, '_last_budget'):
            return self._last_budget.get_usage_stats()
        return {}

    def build_system_prompt(self, topic_name: str = "General") -> str:
        return f"""You are an AI assistant with access to a cognitive memory system for the topic: {topic_name}.

Your memory system provides you with:
- **Current State**: What is true right now
- **Timeline**: How we got here (key events)
- **Decisions**: What was decided and why
- **Episodes**: What happened in past conversations
- **Semantic Memory**: Key insights and patterns

The context you receive is already optimized and prioritized. Higher priority information appears first.

When answering:
1. Use the memory context as your knowledge base
2. If memory shows conflicting information, acknowledge it
3. If you need more detail, you can request recall at higher resolution
4. Maintain conversation continuity with the user

Current date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}"""


def create_context_builder(
    store: MemoryStore,
    recall_engine: BasicRecallEngine,
    encoding_name: str = "cl100k_base",
    budget_ratios: dict[PriorityTier, float] | None = None,
) -> ContextBuilder:
    return EnhancedContextBuilder(store, recall_engine, encoding_name, budget_ratios)


# Backward compatibility
BasicContextBuilder = EnhancedContextBuilder
TokenCounter = TokenCounter
