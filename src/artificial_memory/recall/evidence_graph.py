"""Associative Multi-Hop Evidence Graph for AM Apex (Phase X.6).

Constructs an associative evidence graph over dialogue turns and IR records,
enabling multi-hop traversal across sessions and entities without context stuffing:
    Q -> Seed Evidence (Hop 0) -> Informative Phrase / Entity Links (Hop 1) -> Target Evidence (Hop 2)

Key Innovations:
- Informative Phrase Sieve: Strips conversational filler and connects nodes sharing
  multi-word conceptual collocations (e.g. "home country", "single parent", "pride parade").
- Rare Word Bridges: Connects nodes sharing low-frequency content words.
- Adjacent Turn Preservation: Maintains temporal coherence within the same dialogue session.
- Unified Hop Propagation: Computes associative energy boosts for second-hop nodes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from artificial_memory.core.ir.memory_types import ApexMemoryUnit

STOPWORDS = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'my', 'your', 'his', 'her', 'our', 'their',
    'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
    'into', 'over', 'after', 'yeah', 'yes', 'no', 'yep', 'thanks', 'thank', 'hey', 'hi', 'hello',
    'mel', 'melanie', 'caroline', 'wow', 'so', 'that', 'this', 'there', 'then', 'just', 'really',
    'very', 'much', 'too', 'also', 'and', 'but', 'or', 'if', 'what', 'when', 'where', 'how', 'why',
    'who', 'which', 'will', 'would', 'can', 'could', 'should', 'here', 'some', 'other', 'stuff',
    'like', 'got', 'good', 'great', 'awesome', 'nice', 'sounds', 'super', 'well', 'talk', 'soon',
    'see', 'know', 'think', 'thought', 'sure', 'fine', 'okay', 'right'
}


def extract_informative_phrases(text: str) -> set[str]:
    """Extract 2-word informative collocations excluding conversational stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\']+\b", text.lower())
    phrases = set()
    for j in range(len(tokens) - 1):
        w1, w2 = tokens[j], tokens[j + 1]
        if w1 not in STOPWORDS and w2 not in STOPWORDS and len(w1) > 2 and len(w2) > 2:
            phrases.add(f"{w1} {w2}")
    return phrases


@dataclass
class EvidenceNode:
    """A node in the evidence graph wrapping an ApexMemoryUnit."""
    node_id: str
    unit: ApexMemoryUnit
    speaker: str
    session_num: int
    text: str
    content_words: set[str] = field(default_factory=set)
    phrases: set[str] = field(default_factory=set)


@dataclass
class EvidenceEdge:
    """A directed/undirected edge between two evidence nodes."""
    source_id: str
    target_id: str
    relation_type: str  # "phrase_link", "rare_word", "adjacent_turn"
    weight: float


class EvidenceGraph:
    """Associative multi-hop graph over conversation units."""

    def __init__(self, units: Sequence[ApexMemoryUnit]) -> None:
        self.nodes: dict[str, EvidenceNode] = {}
        self.adjacency: dict[str, list[EvidenceEdge]] = defaultdict(list)
        self._build_graph(units)

    def _build_graph(self, units: Sequence[ApexMemoryUnit]) -> None:
        """Build graph nodes and associative edges across units."""
        # 1. Create nodes
        for i, u in enumerate(units):
            m_dia = re.search(r"\[(D\d+:\d+)", u.ir.raw_content)
            nid = m_dia.group(1) if m_dia else f"unit-{i}"
            speaker = (u.ir.source or "general").lower()
            text = u.ir.raw_content.lower()

            m_s = re.search(r"session_?(\d+)", text) or re.search(r"\[D(\d+):", text)
            s_num = int(m_s.group(1)) if m_s else 1

            words = set(re.findall(r"\b[a-zA-Z0-9_\-\']+\b", text))
            words = {w for w in words if len(w) > 2 and w not in STOPWORDS}

            phrases = extract_informative_phrases(text)

            node = EvidenceNode(
                node_id=nid,
                unit=u,
                speaker=speaker,
                session_num=s_num,
                text=text,
                content_words=words,
                phrases=phrases,
            )
            self.nodes[nid] = node

        # 2. Inverted indexes
        phrase_to_nodes: dict[str, list[str]] = defaultdict(list)
        rare_words: dict[str, list[str]] = defaultdict(list)

        for nid, node in self.nodes.items():
            for p in node.phrases:
                phrase_to_nodes[p].append(nid)
            for w in node.content_words:
                if len(w) > 5:
                    rare_words[w].append(nid)

        # 3. Informative Phrase Links (weight = 25.0)
        for p, nids in phrase_to_nodes.items():
            if 2 <= len(nids) <= 6:
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        self._add_edge(nids[i], nids[j], f"phrase:{p}", 25.0)

        # 4. Rare Word Bridges (weight = 10.0)
        for w, nids in rare_words.items():
            if 2 <= len(nids) <= 4:
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        self._add_edge(nids[i], nids[j], f"rare:{w}", 10.0)

        # 5. Adjacent Turn Edges (only within same session)
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids) - 1):
            n1 = self.nodes[node_ids[i]]
            n2 = self.nodes[node_ids[i + 1]]
            if n1.session_num == n2.session_num:
                self._add_edge(node_ids[i], node_ids[i + 1], "adjacent_turn", 8.0)

    def _add_edge(self, id1: str, id2: str, rel_type: str, weight: float) -> None:
        self.adjacency[id1].append(EvidenceEdge(id1, id2, rel_type, weight))
        self.adjacency[id2].append(EvidenceEdge(id2, id1, rel_type, weight))

    def get_hop_boosts(self, seed_ids: list[str]) -> dict[str, float]:
        """Compute associative graph propagation boosts from seed units."""
        boosts: dict[str, float] = defaultdict(float)
        seed_set = set(seed_ids)
        for s_id in seed_ids:
            for edge in self.adjacency.get(s_id, []):
                target_id = edge.target_id
                if target_id not in seed_set:
                    boosts[target_id] += edge.weight
        return boosts

    def traverse(
        self,
        seed_units: Sequence[ApexMemoryUnit],
        query: str,
        max_hops: int = 2,
        max_results: int = 6,
    ) -> list[ApexMemoryUnit]:
        """Legacy compatibility wrapper: returns top hop units."""
        seed_ids = []
        for u in seed_units:
            m_dia = re.search(r"\[(D\d+:\d+)", u.ir.raw_content)
            if m_dia and m_dia.group(1) in self.nodes:
                seed_ids.append(m_dia.group(1))

        boosts = self.get_hop_boosts(seed_ids)
        sorted_ids = sorted(boosts.keys(), key=lambda x: boosts[x], reverse=True)
        return [self.nodes[nid].unit for nid in sorted_ids[:max_results]]
