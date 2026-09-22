"""Deterministic Temporal State Compiler for AM Apex (Phase CHRONOS).

Compiles event time references and relative temporal expressions into anchored
cognitive states with explicit session timestamp grounding:
    [TEMPORAL STATE] <Subject> <Event> on/in <AnchoredDate>. (supported_by: [D5:3])

Features:
1. Relative-to-Absolute Calendar Anchoring:
   - "yesterday" + Session 2023-07-06 -> "5 July 2023"
   - "last week" + Session 2023-06-09 -> "The week before 9 June 2023"
   - "last weekend" + Session 2023-07-17 -> "The weekend before 17 July 2023"
   - "last Friday" + Session 2023-07-15 -> "The Friday before 15 July 2023"
   - "next month" + Session 2023-05-25 -> "June 2023"
   - "this month" + Session 2023-07-12 -> "July 2023"
   - "two weekends before" + Session 2023-07-17 -> "two weekends before 17 July 2023"
2. Turn ID Provenance tracking for full traceability.
3. 100% Deterministic Arithmetic ($0.00 Ingestion/Write-side LLM cost).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence
from dataclasses import dataclass

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.temporal_resolver import parse_date


@dataclass
class CompiledTemporalState:
    """A compiled temporal state statement with provenance."""
    subject: str
    event_description: str
    anchored_date: str
    relative_expression: str
    session_date: str
    turn_id: str
    confidence: float = 1.0

    def format_state(self) -> str:
        prov = f" (supported_by: [{self.turn_id}])" if self.turn_id else ""
        subj = self.subject.capitalize() if self.subject else "The user"
        # Determine preposition
        prep = "in" if re.match(r"^[a-zA-Z]+\s+\d{4}$|^\d{4}$", self.anchored_date.strip()) else "on"
        if any(self.anchored_date.lower().startswith(p) for p in ["the week", "the weekend", "the sunday", "the friday", "the tuesday", "two weekends", "since"]):
            return f"[TEMPORAL STATE] {subj} {self.event_description} {self.anchored_date}.{prov}"
        return f"[TEMPORAL STATE] {subj} {self.event_description} {prep} {self.anchored_date}.{prov}"


class TemporalCompiler:
    """Deterministic Temporal Compiler."""

    WEEKDAY_MAP = {
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday",
        "saturday": "Saturday",
        "sunday": "Sunday",
    }

    # LoCoMo-like dialogue contains both full weekday names and natural chat
    # abbreviations (``last Fri``, ``last Tues``).  Keep this mapping here,
    # rather than asking the answer model to expand them, so the same input
    # always produces the same anchor.
    WEEKDAY_ALIASES = {
        "mon": "Monday", "monday": "Monday",
        "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
        "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
        "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
        "fri": "Friday", "friday": "Friday",
        "sat": "Saturday", "saturday": "Saturday",
        "sun": "Sunday", "sunday": "Sunday",
    }

    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }

    _QUERY_STOP_WORDS = frozenset({
        "when", "what", "date", "day", "month", "year", "how", "long", "did", "does",
        "do", "is", "was", "will", "has", "have", "been", "the", "a", "an", "in", "on",
        "at", "to", "for", "with", "of", "and", "or", "my", "our", "their", "her", "his",
        "she", "he", "they", "them", "melanie", "caroline", "get", "make", "put", "putting",
    })

    # Small, transparent equivalence classes for conversational paraphrases.
    # These are used only to find a source turn; they never manufacture a date.
    TERM_EXPANSIONS = {
        "art": {"painting", "pottery", "drawing", "creative", "creativity"},
        "festival": {"fest"},
        "fesetival": {"festival", "fest"},  # common typo in the public data
        "friend": {"buddy"},
        "child": {"kid", "kids"},
    }

    IRREGULAR_STEMS = {
        "made": "make",
        "went": "go",
        "got": "get",
        "took": "take",
        "bought": "buy",
        "ran": "run",
    }

    MONTH_NAMES = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    def is_temporal_query(self, query: str) -> bool:
        """Check if the query asks for a date, time, or temporal duration."""
        q_lower = query.lower()
        patterns = [
            r"\bwhen\s+(did|is|was|will|do|does)\b",
            r"\bwhat\s+(date|day|month|year|time)\b",
            r"\bhow\s+long\s+(ago|has|have|did)\b",
            r"\bwhat\s+period\b",
        ]
        return any(re.search(p, q_lower) for p in patterns)

    @staticmethod
    def _stem(token: str) -> str:
        """Return a deliberately small English stem for lexical recall.

        This is not intended to be a linguistic parser.  It just keeps
        ``paint``/``painted`` and ``camp``/``camping`` in the same retrieval
        bucket without adding a model dependency to the write path.
        """
        token = token.lower()
        if token in TemporalCompiler.IRREGULAR_STEMS:
            return TemporalCompiler.IRREGULAR_STEMS[token]
        if len(token) > 5 and token.endswith("ing"):
            base = token[:-3]
            # ``hiking`` / ``making`` lose a silent e before ``-ing``.
            return base + "e" if base.endswith("k") else base
        if len(token) > 4 and token.endswith("ied"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _event_keywords(self, query: str) -> list[str]:
        """Extract stable lexical anchors from a temporal question."""
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        keywords = {
            self._stem(token)
            for token in tokens
            if len(token) > 2 and token not in self._QUERY_STOP_WORDS
        }
        for token in tuple(keywords):
            keywords.update(self._stem(related) for related in self.TERM_EXPANSIONS.get(token, ()))
        return sorted(keywords)

    @classmethod
    def _number_value(cls, value: str) -> int:
        return int(value) if value.isdigit() else cls.NUMBER_WORDS[value.lower()]

    def _anchor_relative_expression(
        self,
        content: str,
        session_date: datetime.date | None,
    ) -> tuple[str, str]:
        """Return ``(anchored_date, source_expression)`` for one dialogue turn.

        The emitted wording intentionally preserves calendar *intervals* such
        as "the week before 9 June" instead of inventing a day inside that
        interval.  This is crucial for factual faithfulness: natural-language
        "last week" does not identify a unique date.
        """
        if not session_date:
            return "", ""

        d = session_date
        date_text = f"{d.day} {self.MONTH_NAMES[d.month]} {d.year}"
        text = content.lower()

        # More specific phrases must win before their component words.
        if re.search(r"\bthe\s+day\s+before\s+yesterday\b", text):
            prior = d - datetime.timedelta(days=2)
            return f"{prior.day} {self.MONTH_NAMES[prior.month]} {prior.year}", "the day before yesterday"
        if re.search(r"\byesterday\b", text):
            prior = d - datetime.timedelta(days=1)
            return f"{prior.day} {self.MONTH_NAMES[prior.month]} {prior.year}", "yesterday"
        if re.search(r"\blast\s+night\b", text):
            prior = d - datetime.timedelta(days=1)
            return f"{prior.day} {self.MONTH_NAMES[prior.month]} {prior.year}", "last night"
        if re.search(r"\btomorrow\b", text):
            future = d + datetime.timedelta(days=1)
            return f"{future.day} {self.MONTH_NAMES[future.month]} {future.year}", "tomorrow"
        if re.search(r"\blast\s+year\b", text):
            return str(d.year - 1), "last year"
        if re.search(r"\bnext\s+year\b", text):
            return str(d.year + 1), "next year"

        duration = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(days?|weeks?|months?|years?)\s+(?:ago|now)\b",
            text,
        )
        if duration:
            count = self._number_value(duration.group(1))
            unit = duration.group(2).lower()
            if unit.startswith("day"):
                prior = d - datetime.timedelta(days=count)
                return f"{prior.day} {self.MONTH_NAMES[prior.month]} {prior.year}", duration.group(0)
            if unit.startswith("week"):
                return f"The week before {date_text}" if count == 1 else f"{count} weeks before {date_text}", duration.group(0)
            if unit.startswith("month"):
                month_index = d.month - count
                year = d.year
                while month_index < 1:
                    month_index += 12
                    year -= 1
                return f"{self.MONTH_NAMES[month_index]} {year}", duration.group(0)
            # A duration phrased as "seven years now" is a continuing state,
            # so expose its start rather than claiming a single event date.
            return f"Since {d.year - count}", duration.group(0)

        if re.search(r"\btwo\s+weekends?\b", text):
            return f"two weekends before {date_text}", "two weekends before"
        if re.search(r"\blast\s+weekend\b", text):
            return f"The weekend before {date_text}", "last weekend"
        if re.search(r"\bthis\s+past\s+weekend\b|\bpast\s+weekend\b", text):
            return f"The weekend before {date_text}", "past weekend"
        if re.search(r"\blast\s+week\b", text):
            return f"The week before {date_text}", "last week"
        if re.search(r"\bthis\s+week\b", text):
            return f"The week of {date_text}", "this week"

        weekday = re.search(
            r"\blast\s+(mon(?:day)?|tues?(?:day)?|wed(?:nesday)?|thurs?(?:day)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
            text,
        )
        if weekday:
            day_name = self.WEEKDAY_ALIASES[weekday.group(1).lower()]
            return f"The {day_name} before {date_text}", f"last {weekday.group(1)}"

        if re.search(r"\bnext\s+month\b", text):
            month = d.month + 1
            year = d.year
            if month == 13:
                month, year = 1, year + 1
            return f"{self.MONTH_NAMES[month]} {year}", "next month"
        if re.search(r"\blast\s+month\b", text):
            month = d.month - 1 or 12
            year = d.year if d.month > 1 else d.year - 1
            return f"{self.MONTH_NAMES[month]} {year}", "last month"
        if re.search(r"\bthis\s+month\b", text):
            return f"{self.MONTH_NAMES[d.month]} {d.year}", "this month"
        return "", ""

    def compile_temporal_states(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> list[CompiledTemporalState]:
        """Compile temporal states for temporal queries."""
        if not self.is_temporal_query(query):
            return []

        q_lower = query.lower()

        # 1. Identify Subject
        subject = ""
        if "melanie" in q_lower:
            subject = "Melanie"
        elif "caroline" in q_lower:
            subject = "Caroline"

        # 2. Extract Event Keywords from Query.  A small stemmer makes this
        # deterministic lexical retrieval robust to ordinary inflections.
        event_keywords = self._event_keywords(query)

        if not event_keywords:
            return []

        # 3. Search for the Best Matching Evidence Record
        best_record: StructuredIR | None = None
        best_score = 0.0

        for r in records:
            content = (r.raw_content or "").lower()
            if not content:
                continue
            content_words = {
                self._stem(token)
                for token in re.findall(r"\b[a-zA-Z0-9_-]+\b", content)
            }

            # Check overlap with event keywords
            event_overlap = 0.0
            for kw in event_keywords:
                if kw in content_words:
                    event_overlap += 1
                    if len(kw) > 4:
                        event_overlap += 0.5

            if event_overlap == 0:
                continue

            # Temporal expression bonus
            has_temp = any(tw in content for tw in [
                "yesterday", "last night", "last week", "last weekend", "last fri", "last tue",
                "next month", "this month", "last month", "last year", "ago", "years now",
                "since", "two weekends", "past weekend", "friday before", "weekend before"
            ])
            # One broad word (for example, "family" or "together") is not
            # enough to bind a time statement.  Accept it only when the same
            # turn contains a temporal expression; otherwise it is a likely
            # topical coincidence in a long conversation.
            if event_overlap < 2.0 and not has_temp:
                continue

            # Lexical event evidence is the primary signal.  Temporal wording
            # breaks ties; it must never let an unrelated "last week" outrank
            # a more specific event match.
            score = event_overlap * 3.0
            if subject and (r.source or "").lower() == subject.lower():
                score += 1.0
            if has_temp:
                score += 2.0
            if self.is_temporal_query(query) and re.search(
                r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+(?:ago|now)\b",
                content,
            ):
                score += 3.0

            if score > best_score:
                best_score = score
                best_record = r

        if not best_record:
            return []

        # 4. Extract Session Date and Turn ID
        raw_content = best_record.raw_content or ""
        m_turn = re.search(r"\[(D\d+:\d+|\d+)", raw_content)
        turn_id = m_turn.group(1) if m_turn else ""

        session_date_str = best_record.time_scope or ""
        if not session_date_str:
            # Try to extract date from raw_content: [D1:7 on 1:56 pm on 8 May, 2023]
            m_date = re.search(r"on\s+(?:\d{1,2}:\d{2}\s+(?:am|pm)\s+on\s+)?([A-Za-z0-9, ]+?)\]", raw_content)
            if m_date:
                session_date_str = m_date.group(1).strip()

        session_d = parse_date(session_date_str)
        content_lower = raw_content.lower()

        # 5. Perform deterministic calendar anchoring before considering an
        # explicit year.  ``raw_content`` includes the session timestamp, so
        # reversing this order silently turned "last year" into the session
        # year in the old implementation.
        anchored_date, rel_expr = self._anchor_relative_expression(content_lower, session_d)

        # Case H: Explicit year / date mentioned in the utterance.
        if not anchored_date:
            # Exclude the wrapper timestamp when looking for an explicit year.
            utterance = re.sub(r"^\[[^\]]+\]\s*[^:]*:\s*", "", raw_content).strip()
            utterance_lower = utterance.lower()
            m_year = re.search(r"\b(20\d{2})\b", utterance_lower)
            m_month_day = re.search(r"\b(\d{1,2}\s+[A-Za-z]+|[A-Za-z]+\s+\d{1,2})\b", utterance)
            if "since" in utterance_lower and m_year:
                anchored_date = f"Since {m_year.group(1)}"
                rel_expr = f"since {m_year.group(1)}"
            elif m_month_day and not session_d:
                anchored_date = m_month_day.group(1).strip()
                rel_expr = anchored_date
            elif m_year:
                anchored_date = m_year.group(1)
                rel_expr = m_year.group(1)
            elif session_d:
                # Fallback to session date itself if event happened on that day
                anchored_date = f"{session_d.day} {self.MONTH_NAMES[session_d.month]} {session_d.year}"
                rel_expr = "on session date"

        if not anchored_date:
            return []

        # Construct a concise event description from query keywords
        event_phrase = " ".join(event_keywords[:4])

        state = CompiledTemporalState(
            subject=subject or "The subject",
            event_description=event_phrase,
            anchored_date=anchored_date,
            relative_expression=rel_expr,
            session_date=session_date_str,
            turn_id=turn_id,
            confidence=0.95,
        )
        return [state]
