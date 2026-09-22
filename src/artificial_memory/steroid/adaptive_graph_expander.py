"""Adaptive Graph Expander for AM Apex Steroid Phase (Steroid #2).

Implements quality-gated adaptive graph expansion:
    E_0 = WideSlice(Q)
    E_{k+1} = Expand(E_k)
    E* = Union_{k=0}^K E_k

Features:
- Dynamically checks Evidence Completeness at each hop:
  * If the target entity, predicate, and referents are grounded -> Early exit (STOP).
  * If referents remain unresolved (e.g. "home country", "partner's dog", "which event") -> Expand along entity/relation edges.
- Bounded at max K <= 4 hops with cycle prevention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph


@dataclass
class HopResult:
    """Result of a single expansion hop."""
    hop_index: int
    added_records: list[StructuredIR]
    total_evidence_count: int
    unresolved_referents: list[str] = field(default_factory=list)
    is_complete: bool = False


@dataclass
class AdaptiveExpansionResult:
    """Overall result of adaptive graph expansion."""
    evidence_pool: list[StructuredIR]
    hops_performed: int
    hop_history: list[HopResult] = field(default_factory=list)
    final_referents_resolved: bool = True


class AdaptiveGraphExpander:
    """Adaptive evidence expansion along entity, proposition, and temporal edges."""

    UNRESOLVED_PATTERNS = [
        re.compile(r"\b(?:her|his|their|my)\s+(home country|partner|friend|colleague|dog|cat|pet|car|bike|hometown|university|degree|job)\b", re.IGNORECASE),
        re.compile(r"\b(?:the)\s+(country|city|hotel|event|conference|company|university)\b", re.IGNORECASE),
    ]

    def __init__(self, max_hops: int = 3, per_hop_budget: int = 15) -> None:
        self.max_hops = max_hops
        self.per_hop_budget = per_hop_budget

    def _find_unresolved_referents(self, text: str) -> list[str]:
        """Detect unresolved referents in the current evidence text."""
        referents = []
        for pat in self.UNRESOLVED_PATTERNS:
            for m in pat.finditer(text):
                referents.append(m.group(1).lower())
        return list(set(referents))

    def expand(
        self,
        query: str,
        initial_records: Sequence[StructuredIR],
        all_corpus_records: Sequence[StructuredIR],
        graph: UnifiedPropositionGraph | None = None,
    ) -> AdaptiveExpansionResult:
        """Perform quality-gated adaptive expansion."""
        q_lower = query.lower()
        q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2)

        evidence_pool = list(initial_records)
        seen_keys = {r.raw_content for r in evidence_pool if r.raw_content}
        hop_history: list[HopResult] = []

        # Check initial completeness
        current_text = " ".join(r.raw_content for r in evidence_pool).lower()
        unresolved = self._find_unresolved_referents(current_text)

        hop_0 = HopResult(
            hop_index=0,
            added_records=list(initial_records),
            total_evidence_count=len(evidence_pool),
            unresolved_referents=unresolved,
            is_complete=(len(unresolved) == 0),
        )
        hop_history.append(hop_0)

        # If already complete and has substantial evidence, stop early
        if hop_0.is_complete and len(evidence_pool) >= 5:
            return AdaptiveExpansionResult(
                evidence_pool=evidence_pool,
                hops_performed=0,
                hop_history=hop_history,
                final_referents_resolved=True,
            )

        # Iterative expansion hops
        for k in range(1, self.max_hops + 1):
            if not unresolved and k > 1:
                break

            # Find candidate records connecting to unresolved referents or key entities
            hop_added: list[StructuredIR] = []
            search_targets = set(unresolved)
            for w in q_words:
                if w.istitle() or len(w) > 4:
                    search_targets.add(w.lower())

            for r in all_corpus_records:
                key = r.raw_content
                if not key or key in seen_keys:
                    continue

                r_text = key.lower()
                # Connect if mentions any unresolved referent or edge entity
                if any(t in r_text for t in search_targets):
                    hop_added.append(r)
                    seen_keys.add(key)
                    if len(hop_added) >= self.per_hop_budget:
                        break

            evidence_pool.extend(hop_added)
            current_text = " ".join(r.raw_content for r in evidence_pool).lower()
            unresolved = self._find_unresolved_referents(current_text)

            is_complete = (len(unresolved) == 0) or (len(hop_added) == 0)
            h_res = HopResult(
                hop_index=k,
                added_records=hop_added,
                total_evidence_count=len(evidence_pool),
                unresolved_referents=unresolved,
                is_complete=is_complete,
            )
            hop_history.append(h_res)

            if is_complete and len(evidence_pool) >= 6:
                break

        return AdaptiveExpansionResult(
            evidence_pool=evidence_pool,
            hops_performed=len(hop_history) - 1,
            hop_history=hop_history,
            final_referents_resolved=(len(unresolved) == 0),
        )
