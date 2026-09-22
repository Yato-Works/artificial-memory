"""State & Supersession Engine for AM Apex Overdrive Core (Potion 3).

Resolves dynamic knowledge updates, state transitions, and supersession histories:
- "What do they currently use / live in / do?" -> max(valid_time)
- "What did they use / live in / do before?" -> history traversal
- Generates deterministic State Transition Certificates (< 20 tokens)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from artificial_memory.core.ir.proposition import StateHistory, StateSnapshot
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
from artificial_memory.recall.query_planner import QueryPlan


@dataclass
class StateResolution:
    """Result of state supersession resolution."""
    target_entity: str
    target_attribute: str
    is_current_query: bool
    resolved_snapshot: Optional[StateSnapshot]
    grounding_certificate: str


class StateSupersessionEngine:
    """Resolves knowledge updates and state histories deterministically."""

    def resolve(
        self,
        plan: QueryPlan,
        graph: UnifiedPropositionGraph,
    ) -> Optional[StateResolution]:
        """Analyze query plan against state histories in proposition graph."""
        q_lower = plan.raw_query.lower()
        is_previous_query = any(w in q_lower for w in ["previous", "previously", "used to", "before", "old", "earlier"])
        is_current_query = any(w in q_lower for w in ["currently", "now", "recent", "new", "present"]) or not is_previous_query

        # Match entity and attribute in state histories
        for (ent, attr), history in graph.state_histories.items():
            ent_match = any(e in ent or ent in e for e in plan.target_entities) if plan.target_entities else True
            attr_clean = attr.rstrip("s")
            attr_match = attr in q_lower or attr_clean in q_lower or any(p in q_lower for p in plan.target_predicates)

            if ent_match and attr_match and len(history.snapshots) >= 1:
                if is_previous_query and len(history.snapshots) >= 2:
                    snap = history.get_previous()
                    cert = (
                        f"[State Resolution: Previously, {history.entity} {history.attribute} was "
                        f"'{snap.value}' (as of {snap.timestamp}).]"
                    )
                    return StateResolution(
                        target_entity=history.entity,
                        target_attribute=history.attribute,
                        is_current_query=False,
                        resolved_snapshot=snap,
                        grounding_certificate=cert,
                    )
                else:
                    snap = history.get_latest()
                    cert = (
                        f"[State Resolution: Currently, {history.entity} {history.attribute} is "
                        f"'{snap.value}' (as of {snap.timestamp}).]"
                    )
                    return StateResolution(
                        target_entity=history.entity,
                        target_attribute=history.attribute,
                        is_current_query=True,
                        resolved_snapshot=snap,
                        grounding_certificate=cert,
                    )

        return None
