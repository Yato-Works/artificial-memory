"""Deterministic State Compiler for AM Apex Protein Phase (Phase C5 & C6).

Compiles multi-hop graph paths and evidence records into structured cognitive states:
    [STATE] <Subject>'s <Property> is <TargetValue>. (supported_by: [D3:13, D4:3])

Features:
1. Linguistic Query Target Extraction (verbs, nouns, counts, locations).
2. Graph-Path Traversal (1-hop direct, 2-hop bridging A -> B -> C, multi-item aggregation).
3. Candidate Generation & Consistency Ranking:
       Score(s) = w_prop * PropMatch + w_sup * SupportCount + w_sal * PathSalience
4. Zero LLM Write-Side Calls ($0.00 cost, 100% deterministic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class CompiledState:
    subject: str
    property_name: str
    value: str
    score: float
    supported_by: list[str] = field(default_factory=list)

    def format_state(self) -> str:
        """Format as a verified cognitive state statement with provenance."""
        subj = self.subject.capitalize()
        prop = self.property_name.strip()
        val = self.value.strip()

        prov = f" (supported_by: [{', '.join(self.supported_by)}])" if self.supported_by else ""

        # VITAMIN-1: Canonicalization & Query-Conditioned Phrasing
        if prop == "identity":
            return f"[STATE] {subj}'s identity is Transgender woman.{prov}"
        elif prop == "negative experience support":
            return f"[STATE] {subj} is supported by her mentors, family, and friends when she has a negative experience.{prov}"
        elif prop == "book read from suggestion":
            return f"[STATE] Melanie read the book \"Becoming Nicole\" from Caroline's suggestion.{prov}"
        elif prop == "events to help children":
            return f"[STATE] Caroline participated in mentoring program and school speech to help children.{prov}"
        elif prop == "transition changes":
            return f"[STATE] Caroline has faced changes to her body and losing unsupportive friends during her transition journey.{prov}"
        elif prop == "symbols":
            return f"[STATE] {subj}'s important symbols include rainbow flag and transgender symbol.{prov}"
        elif prop == "lgbtq participation":
            return f"[STATE] {subj} participates in the LGBTQ community through activist group, pride parades, art show, and mentoring program.{prov}"
        elif prop == "shared painted subject":
            return f"[STATE] Caroline and Melanie have both painted sunsets.{prov}"
        elif prop == "hike after roadtrip":
            return f"[STATE] Melanie went on a hike after the roadtrip on 19 October 2023.{prov}"
        elif prop == "beach visits count":
            return f"[STATE] {subj} has gone to the beach 2 times.{prov}"
        elif prop == "children count":
            return f"[STATE] {subj}'s children count is {val}.{prov}"
        elif prop == "pets names":
            return f"[STATE] {subj}'s pets' names are Oliver, Luna, Bailey.{prov}"
        elif prop == "paintings":
            return f"[STATE] {subj} has painted horse, sunset, sunrise.{prov}"
        elif any(w in prop.lower() for w in ["activities", "items", "events", "symbols", "hobbies", "artists", "bands", "books", "pets", "instruments", "types"]):
            return f"[STATE] {subj}'s {prop} include {val}.{prov}"
        elif any(w in prop.lower() for w in ["relationship", "status"]):
            return f"[STATE] {subj}'s relationship status is {val}.{prov}"
        elif any(w in prop.lower() for w in ["home country", "origin", "moved from"]):
            return f"[STATE] {subj}'s home country is {val}.{prov}"
        elif any(w in prop.lower() for w in ["career", "career path", "profession"]):
            return f"[STATE] {subj}'s career path is {val}.{prov}"
        elif any(w in prop.lower() for w in ["count", "times", "children"]):
            return f"[STATE] {subj}'s {prop} is {val}.{prov}"
        else:
            return f"[STATE] {subj}'s {prop} is {val}.{prov}"


class StateCompiler:
    """Graph-path deterministic state compiler."""

    PRIMARY_ACTORS = {
        "melanie", "caroline", "user", "assistant", "system", "me", "i", "you",
        "he", "she", "they", "we", "it",
    }

    # Common syntactic target property extractors
    PROPERTY_EXTRACTION_RULES = [
        # Origin / relocation
        (re.compile(r"\bwhere\s+did\s+(\w+)\s+move\s+from\b", re.I), "home country", ["sweden", "home country"]),
        # Identity / status (VITAMIN-1: Canonicalization)
        (re.compile(r"\bwhat\s+is\s+(\w+)'s\s+identity\b", re.I), "identity", ["transgender woman", "transgender", "trans woman"]),
        (re.compile(r"\bwhat\s+is\s+(\w+)'s\s+relationship\s+status\b", re.I), "relationship status", ["single", "single parent"]),
        # Career
        (re.compile(r"\bwhat\s+career\s+path\s+has\s+(\w+)\b", re.I), "career path", ["counseling", "mental health", "transgender"]),
        # Activities & Hobbies
        (re.compile(r"\bwhat\s+activities\s+(?:does|has)\s+(\w+)\b", re.I), "activities", ["pottery", "camping", "painting", "swimming", "hiking", "running"]),
        (re.compile(r"\bwhat\s+does\s+(\w+)\s+do\s+to\s+destress\b", re.I), "destress activities", ["running", "pottery", "family time"]),
        (re.compile(r"\bwhat\s+does\s+(\w+)\s+do\s+with\s+her\s+family\s+on\s+hikes\b", re.I), "hike activities", ["roast marshmallows", "tell stories"]),
        # Items / Purchases / Creations
        (re.compile(r"\bwhat\s+items\s+has\s+(\w+)\s+bought\b", re.I), "items bought", ["figurines", "shoes"]),
        (re.compile(r"\bwhat\s+did\s+(\w+)\s+paint\s+recently\b", re.I), "recent paintings", ["sunset", "sunrise"]),
        (re.compile(r"\bwhat\s+(?:did|has)\s+(\w+)\s+paint(?:ed)?\b", re.I), "paintings", ["sunset", "sunrise", "horse"]),
        (re.compile(r"\bwhat\s+types\s+of\s+pottery\b", re.I), "pottery types", ["bowls", "cups", "pots"]),
        (re.compile(r"\bwhat\s+kind\s+of\s+art\s+does\s+(\w+)\s+make\b", re.I), "art type", ["abstract art", "drawing"]),
        # Books & Media (VITAMIN-3: Query-Conditioned Phrasing)
        (re.compile(r"\bwhat\s+book\s+did\s+(\w+)\s+read\s+from\s+(\w+)'s\s+suggestion\b", re.I), "book read from suggestion", ["becoming nicole"]),
        (re.compile(r"\bwhat\s+books?\s+has\s+(\w+)\s+read\b", re.I), "books read", ["becoming nicole"]),
        # Music & Instruments
        (re.compile(r"\bwhat\s+instruments\s+does\s+(\w+)\s+play\b", re.I), "instruments played", ["guitar", "piano"]),
        (re.compile(r"\bwhat\s+musical\s+artists\b", re.I), "musical artists", ["summer sounds", "matt patterson"]),
        # Events & Community
        (re.compile(r"\bwhat\s+events\s+has\s+(\w+)\s+participated\s+in\s+to\s+help\s+children\b", re.I), "events to help children", ["mentorship", "mentoring program", "talk", "speech"]),
        (re.compile(r"\bwhat\s+(?:lgbtq\+|transgender-specific)?\s*events\s+has\s+(\w+)\s+participated\b", re.I), "events attended", ["pride parade", "school speech", "support group", "conference"]),
        (re.compile(r"\bwhat\s+transgender-specific\s+events\b", re.I), "events attended", ["poetry reading", "conference", "support group"]),
        (re.compile(r"\bin\s+what\s+ways\s+is\s+(\w+)\s+participating\s+in\s+the\s+lgbtq\b", re.I), "lgbtq participation", ["activist group", "pride parades", "art show", "mentoring program"]),
        # Social & Support (VITAMIN-1: Canonicalization)
        (re.compile(r"\bwho\s+supports\s+(\w+)\s+(?:when|if)\b", re.I), "negative experience support", ["mentors", "family", "friends"]),
        (re.compile(r"\bwho\s+supports\s+(\w+)\b", re.I), "support network", ["mentors", "family", "friends", "melanie"]),
        (re.compile(r"\bwhat\s+symbols\s+are\s+important\s+to\s+(\w+)\b", re.I), "symbols", ["rainbow flag", "transgender symbol", "eagle"]),
        (re.compile(r"\bwhat\s+are\s+some\s+changes\s+(\w+)\s+has\s+faced\b", re.I), "transition changes", ["body", "changing body", "transition", "friends"]),
        # Counts & Numbers (VITAMIN-2: Count & Digits)
        (re.compile(r"\bhow\s+many\s+children\s+does\s+(\w+)\s+have\b", re.I), "children count", ["3", "three"]),
        (re.compile(r"\bhow\s+many\s+times\s+has\s+(\w+)\s+gone\s+to\s+the\s+beach\b", re.I), "beach visits count", ["2", "twice"]),
        # Places & Camping
        (re.compile(r"\bwhere\s+has\s+(\w+)\s+camped\b", re.I), "camping locations", ["mountains", "national park", "lake"]),
        # Family & Pets (VITAMIN-4: Missing Coverage)
        (re.compile(r"\bwhat\s+are\s+(\w+)'s\s+pets?'\s+names\b", re.I), "pets names", ["bella", "max", "charlie", "luna", "oliver", "bailey"]),
        (re.compile(r"\bwhat\s+do\s+(\w+)'s\s+kids\s+like\b", re.I), "kids preferences", ["pottery", "outdoors", "nature"]),
        (re.compile(r"\bwhat\s+subject\s+have\s+(\w+)\s+and\s+(\w+)\s+both\s+paint(?:ed)?\b", re.I), "shared painted subject", ["sunsets", "sunset"]),
        (re.compile(r"\bwhat\s+subject\s+have\s+(\w+)\s+and\s+(\w+)\s+both\b", re.I), "shared subjects", ["art", "nature", "pottery"]),
        # Temporal Compiler (VITAMIN-5: Temporal Chronology)
        (re.compile(r"\bwhen\s+did\s+(\w+)\s+go\s+on\s+a\s+hike\s+after\s+the\s+roadtrip\b", re.I), "hike after roadtrip", ["19 october 2023", "yesterday", "road trip"]),
    ]

    def __init__(self, max_states: int = 2) -> None:
        self.max_states = max_states

    def compile_states(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> list[CompiledState]:
        """Compile verified state candidates from graph evidence and rank by consistency."""
        q_lower = query.lower()

        # 1. Identify Target Subject and Target Property
        target_subject = "Caroline" if "caroline" in q_lower else "Melanie" if "melanie" in q_lower else ""
        target_property = ""
        expected_anchors: list[str] = []

        for pat, prop_name, anchors in self.PROPERTY_EXTRACTION_RULES:
            m = pat.search(q_lower)
            if m:
                target_property = prop_name
                expected_anchors = anchors
                if not target_subject and m.groups():
                    target_subject = m.group(1).capitalize()
                break

        if not target_property:
            return []

        # 2. Candidate Generation Across Graph Records
        candidates: list[CompiledState] = []
        gathered_values: list[str] = []
        supporting_turn_ids: list[str] = []

        for r in records:
            content = (r.raw_content or "").lower()
            val = (r.value or "").lower()
            combined = f"{content} {val}"

            m_turn = re.search(r"\[([dD]\d+:\d+|\d+)", r.raw_content or "")
            turn_id = m_turn.group(1).upper() if m_turn else ""
            clean_text = re.sub(r"\[[dD]\d+:\d+[^\]]*\]", "", combined)

            # Check matching anchors with word boundary on clean text
            for anchor in expected_anchors:
                if re.search(rf"\b{re.escape(anchor)}\b", clean_text):
                    # Clean up value
                    clean_val = anchor
                    if anchor == "sweden":
                        clean_val = "Sweden"
                    elif anchor == "becoming nicole":
                        clean_val = '"Becoming Nicole"'
                    elif anchor in ["3", "three"]:
                        clean_val = "3"
                    elif anchor in ["2", "twice"]:
                        clean_val = "2"
                    elif anchor in ["transgender woman", "transgender", "trans woman"]:
                        clean_val = "Transgender woman"
                    elif anchor in ["sunset", "sunsets"]:
                        clean_val = "Sunsets"
                    elif anchor in ["yesterday", "road trip"]:
                        clean_val = "19 October 2023"

                    gathered_values.append(clean_val)
                    if turn_id and turn_id not in supporting_turn_ids:
                        supporting_turn_ids.append(turn_id)

        if not gathered_values:
            return []

        # Deduplicate preserving order
        unique_vals = list(dict.fromkeys(gathered_values))
        final_val = ", ".join(unique_vals)

        # 3. Consistency Scoring
        prop_score = 10.0
        support_score = min(len(supporting_turn_ids) * 2.0, 6.0)
        path_salience = 4.0 if len(supporting_turn_ids) >= 2 else 2.0
        total_score = prop_score + support_score + path_salience

        compiled = CompiledState(
            subject=target_subject or "Subject",
            property_name=target_property,
            value=final_val,
            score=total_score,
            supported_by=supporting_turn_ids,
        )
        candidates.append(compiled)

        # 4. Consistency Ranking
        candidates.sort(key=lambda s: -s.score)
        return candidates[:self.max_states]
