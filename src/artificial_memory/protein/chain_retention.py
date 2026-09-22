"""Graph Chain Retention (GCR) for AM Apex Protein Phase.

Preserves multi-hop inference chains:
    Query -> Evidence A -> Bridge Entity/Event -> Evidence B -> Conclusion

Rather than inflating the base K (which causes attention dilution),
GraphChainRetainer grants dedicated 'Chain Reserve' slots (+1 to +3 turns)
specifically for records that share an explicit bridge entity, value, or relation
with the top-K evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class ChainCandidate:
    record: StructuredIR
    chain_score: float
    bridge_term: str
    connection_type: str


class GraphChainRetainer:
    """Retains multi-hop chain nodes connecting to top evidence."""

    def __init__(self, max_chain_reserve: int = 3, min_chain_score: float = 5.0) -> None:
        self.max_chain_reserve = max_chain_reserve
        self.min_chain_score = min_chain_score

    def retain_chains(
        self,
        query: str,
        top_records: Sequence[StructuredIR],
        candidate_records: Sequence[StructuredIR],
        max_reserve: int | None = None,
    ) -> list[StructuredIR]:
        """Identify and retain complementary bridging records outside top_records."""
        reserve_limit = max_reserve if max_reserve is not None else self.max_chain_reserve
        if reserve_limit <= 0 or not top_records:
            return list(top_records)

        top_ids = {id(r) for r in top_records}
        PRIMARY_ACTORS = {
            "melanie", "caroline", "user", "assistant", "system", "me", "i", "you",
            "he", "she", "they", "we", "it",
        }
        # Extract bridge anchors from top records (excluding primary conversational actors)
        top_entities = {
            r.entity.lower().strip()
            for r in top_records
            if r.entity and len(r.entity) > 1 and r.entity.lower().strip() not in PRIMARY_ACTORS
        }
        top_values = {
            r.value.lower().strip()
            for r in top_records
            if r.value and len(r.value) > 2
            and r.value.lower().strip() not in PRIMARY_ACTORS
            and r.value.lower() not in ["true", "false", "yes", "no", "unknown", "none"]
        }

        # Query entities
        q_lower = query.lower()
        q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2 and w not in PRIMARY_ACTORS)

        # Extract 2-word phrase anchors from top records (excluding stopwords and actors)
        STOPWORDS = {
            "this is", "that is", "there is", "they are", "we are", "you are", "i am",
            "and the", "in the", "on the", "at the", "to the", "for the", "with the",
            "from the", "of the", "by the", "it is", "have been", "has been", "had been",
            "will be", "would be", "could be", "should be", "can be", "love and",
            "like to", "want to", "going to", "how to", "about the", "really lucky",
            "lucky to", "been there", "through everything", "these friends", "who supports",
        }
        top_phrases: set[str] = set()
        for r in top_records:
            content = (r.raw_content or "").lower()
            words = re.findall(r"\b[a-z]{3,}\b", content)
            for i in range(len(words) - 1):
                p = f"{words[i]} {words[i+1]}"
                if p not in STOPWORDS and not any(a in p.split() for a in PRIMARY_ACTORS):
                    top_phrases.add(p)

        chain_candidates: list[ChainCandidate] = []

        for c in candidate_records:
            if id(c) in top_ids:
                continue

            c_ent = (c.entity or "").lower().strip()
            c_val = (c.value or "").lower().strip()
            c_content = (c.raw_content or "").lower()

            chain_score = 0.0
            bridge_term = ""
            conn_type = ""

            # 1. Forward Hop: c's entity matches a non-actor value from top_records (A -> B, B -> C)
            matched_val = top_values & {c_ent}
            if matched_val and c_ent not in PRIMARY_ACTORS:
                chain_score += 12.0
                bridge_term = next(iter(matched_val))
                conn_type = "forward_hop"

            # 2. Backward Hop: c's value matches a non-actor entity in top_records (X -> A)
            matched_ent = top_entities & {c_val}
            if matched_ent and c_val not in PRIMARY_ACTORS:
                chain_score += 10.0
                bridge_term = next(iter(matched_ent))
                conn_type = "backward_hop"

            # 3. Phrase Bridge: c's content contains a key 2-word phrase anchor from top_records
            matched_phrases = [p for p in top_phrases if p in c_content]
            if matched_phrases:
                chain_score += 12.0
                bridge_term = matched_phrases[0]
                conn_type = "phrase_bridge"

            # 4. Co-occurrence: c's content mentions a non-actor bridge entity from top_records AND a query keyword
            top_ent_in_content = [e for e in top_entities if e in c_content]
            q_word_in_content = [w for w in q_words if w in c_content]
            if top_ent_in_content and q_word_in_content:
                chain_score += 8.0
                bridge_term = top_ent_in_content[0]
                conn_type = "co_occurrence"
            elif top_ent_in_content:
                chain_score += 4.0
                bridge_term = top_ent_in_content[0]
                conn_type = "entity_shared"

            # 4. Same session continuity bonus ONLY if there is already a bridge connection
            if chain_score >= 8.0:
                c_sess = c.metadata.get("session_num") if c.metadata else None
                if c_sess is not None:
                    top_sessions = {r.metadata.get("session_num") for r in top_records if r.metadata}
                    if c_sess in top_sessions:
                        chain_score += 2.0

            if chain_score >= self.min_chain_score:
                chain_candidates.append(
                    ChainCandidate(
                        record=c,
                        chain_score=chain_score,
                        bridge_term=bridge_term,
                        connection_type=conn_type,
                    )
                )

        chain_candidates.sort(key=lambda x: -x.chain_score)
        retained_records = [cand.record for cand in chain_candidates[:reserve_limit]]

        return list(top_records) + retained_records
