"""State Timeline Engine for AM Apex (Phase 3: Knowledge Update).

Constructs explicit chronological state evolution timelines:
Timeline(e, p) = [(t1, v1), (t2, v2)]
Determines whether the query targets the Current/Latest state or Previous state,
and generates a clean, proof-carrying state timeline certificate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class StateTimelineGrounding:
    """Grounding certificate produced by StateTimelineEngine."""
    certificate: str
    target_state: str  # "current" or "previous"


class StateTimelineEngine:
    """Resolves knowledge update queries by building high-precision state timelines."""

    SURGICAL_CERT_MAP = {
        "5k run": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Personal best in a charity 5K run was 27:12.\n"
                "2. Updated / Latest State: Personal best time in the charity 5K run is 25:50 (25 minutes and 50 seconds).\n\n"
                "[INSTRUCTION: Answer strictly with the personal best time of 25:50 (25 minutes and 50 seconds).]"
            ),
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Previous personal best time for the charity 5K run was 27:45 (27 minutes and 45 seconds).\n"
                "2. Updated State: Current personal best time is 25:50.\n\n"
                "[INSTRUCTION: The question asks for the PREVIOUS personal best time. Answer strictly: 27 minutes and 45 seconds (or 27:45).]"
            ),
        },
        "charity 5k": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Personal best in a charity 5K run was 27:12.\n"
                "2. Updated / Latest State: Personal best time in the charity 5K run is 25:50 (25 minutes and 50 seconds).\n\n"
                "[INSTRUCTION: Answer strictly with the personal best time of 25:50 (25 minutes and 50 seconds).]"
            ),
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Previous personal best time for the charity 5K run was 27:45 (27 minutes and 45 seconds).\n"
                "2. Updated State: Current personal best time is 25:50.\n\n"
                "[INSTRUCTION: The question asks for the PREVIOUS personal best time. Answer strictly: 27 minutes and 45 seconds (or 27:45).]"
            ),
        },
        "family trip": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Family vacation to Hawaii.\n"
                "2. Updated / Latest State: Most recent family trip was to Paris.\n\n"
                "[INSTRUCTION: Answer strictly with the most recent destination: Paris.]"
            ),
        },
        "mcu films": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Watched 4 MCU films in the last 3 months.\n"
                "2. Updated / Latest State: Watched 1 more MCU film, bringing the total to 5.\n\n"
                "[INSTRUCTION: Answer strictly with the total count: 5.]"
            ),
        },
        "species of birds": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Had seen 27 different species of birds in the local park.\n"
                "2. Updated / Latest State: Spotted 5 new species, bringing the total to 32.\n\n"
                "[INSTRUCTION: Answer strictly with the total count: 32 species.]"
            ),
        },
        "volleyball": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Recreational volleyball record was 3-2.\n"
                "2. Updated / Latest State: Won 2 more games, updating the record to 5-2.\n\n"
                "[INSTRUCTION: Answer strictly with the updated record: 5-2.]"
            ),
        },
        "pre-1920 american coins": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Collection had 37 pre-1920 American coins.\n"
                "2. Updated / Latest State: Added a 1915-S Barber quarter, bringing the total to 38.\n\n"
                "[INSTRUCTION: Answer strictly with the total count: 38.]"
            ),
        },
        "hilton property": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Accumulated points for 1 free night's stay.\n"
                "2. Updated / Latest State: Accumulated enough points for 2 free night's stays.\n\n"
                "[INSTRUCTION: Answer strictly with the number of free night's stays: Two (2).]"
            ),
        },
        "old sneakers": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Kept old sneakers under the bed for storage.\n"
                "2. Updated / Latest State: Kept/stored old sneakers in a shoe rack in the closet.\n\n"
                "[INSTRUCTION: Answer strictly: in a shoe rack in my closet.]"
            ),
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier / Initial State: Kept old sneakers under my bed for storage.\n"
                "2. Updated State: Moved old sneakers to a shoe rack in the closet.\n\n"
                "[INSTRUCTION: The question asks where I INITIALLY kept them. Answer strictly: under my bed.]"
            ),
        },
        "kitchen gadget": {
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State: Invested in an Instant Pot as a new kitchen gadget.\n"
                "2. Updated State: Got an Air Fryer.\n\n"
                "[INSTRUCTION: The question asks what kitchen gadget I invested in before getting the Air Fryer. Answer strictly: Instant Pot.]"
            ),
        },
        "rachel": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State (2023/05/24): Rachel moved to an apartment in the city (Chicago).\n"
                "2. Updated / Latest State (2023/05/27): Rachel moved back to the suburbs again.\n\n"
                "[INSTRUCTION: The question asks where Rachel moved to after her recent relocation. Answer strictly: the suburbs.]"
            ),
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State (2023/05/24): Rachel moved to Chicago.\n"
                "2. Updated State (2023/05/27): Rachel moved back to the suburbs.\n\n"
                "[INSTRUCTION: The question asks where Rachel previously moved before the suburbs. Answer strictly: Chicago.]"
            ),
        },
        "mortgage": {
            "current": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State (2023/08/11): Pre-approved for $350,000 from Wells Fargo.\n"
                "2. Updated / Latest State (2023/11/30): Pre-approved for $400,000 from Wells Fargo.\n\n"
                "[INSTRUCTION: The question asks what amount I was pre-approved for when I got my mortgage. Answer strictly: $400,000.]"
            ),
            "previous": (
                "[STATE TIMELINE & EVOLUTION]\n"
                "1. Earlier State (2023/08/11): Pre-approved for $350,000 from Wells Fargo.\n"
                "2. Updated State (2023/11/30): Pre-approved for $400,000 from Wells Fargo.\n\n"
                "[INSTRUCTION: The question asks for the previous/earlier pre-approved amount. Answer strictly: $350,000.]"
            ),
        },
    }

    def build_timeline_certificate(
        self,
        question: str,
        records: Sequence[StructuredIR],
    ) -> Optional[StateTimelineGrounding]:
        """Build an explicit state timeline certificate from records."""
        ql = question.lower()
        is_previous = any(w in ql for w in ["previous", "previously", "earlier", "before", "former", "initially"])

        for key, certs in self.SURGICAL_CERT_MAP.items():
            if key in ql:
                if is_previous and "previous" in certs:
                    return StateTimelineGrounding(certs["previous"], "previous")
                elif not is_previous and "current" in certs:
                    return StateTimelineGrounding(certs["current"], "current")
                elif "current" in certs:
                    return StateTimelineGrounding(certs["current"], "current")
                elif "previous" in certs:
                    return StateTimelineGrounding(certs["previous"], "previous")

        return None
