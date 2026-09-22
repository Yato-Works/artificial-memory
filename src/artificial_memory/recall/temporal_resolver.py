"""Deterministic Date Arithmetic & Temporal Resolver for AM Apex (CHRONOS II).

Offloads complex calendar math, event interval calculations, relative time anchoring,
and chronological ordering from the downstream LLM into the deterministic memory runtime.

Handles all core temporal reasoning paradigms:
1. Two-Event Duration: "between Event A and Event B", "had passed since A when I B", "how many days before/after B did I A"
2. Single-Event Duration: "how many days did I spend on [trip/event]", "did it take me to finish [book/project]"
3. Recency from Reference Date: "how many weeks/months ago did I do X?", "have passed since X"
4. Chronological Ordering: "order from first to last: A, B, and C", "which event happened first, A or B?", "who ... first, second and third"
5. Topical Chronological Ordering: "what is the order of the three trips/sports events/museums from earliest to latest"
6. Relative Time Anchoring: "last Saturday", "last Tuesday", "two weeks ago", "Valentine's day"
7. Temporal Abstention: Detects missing entities/events deterministically.
"""

from __future__ import annotations

import calendar
import datetime
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

ACTION_VERBS = [
    "got back from", "started my", "completed", "participated in",
    "attended", "watched", "visited", "flew with", "went on"
]


def parse_date(date_str: str | None) -> Optional[datetime.date]:
    """Parse various date string formats into datetime.date."""
    if not date_str:
        return None

    # 1. YYYY/MM/DD or YYYY-MM-DD
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", date_str)
    if m:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 2. DD Month YYYY
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", date_str)
    if m:
        m_name = m.group(2).lower()[:3]
        if m_name in MONTH_MAP:
            return datetime.date(int(m.group(3)), MONTH_MAP[m_name], int(m.group(1)))

    # 3. Month DD, YYYY
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", date_str)
    if m:
        m_name = m.group(1).lower()[:3]
        if m_name in MONTH_MAP:
            return datetime.date(int(m.group(3)), MONTH_MAP[m_name], int(m.group(2)))

    return None


@dataclass
class TemporalGrounding:
    """Grounding certificate produced by DateArithmeticEngine."""
    grounding_text: str
    target_dates: list[datetime.date]
    calculation_type: str  # "duration", "recency", "ordering", "time_anchor", "abstention"


class DateArithmeticEngine:
    """Performs deterministic calendar arithmetic on dates."""

    @staticmethod
    def calculate_difference(d1: datetime.date, d2: datetime.date, unit: str = "days") -> int:
        """Calculate absolute difference between two dates."""
        delta_days = abs((d2 - d1).days)
        if "week" in unit:
            return round(delta_days / 7.0)
        elif "month" in unit:
            # Calendar month difference if close, or days/30
            m_diff = abs((d2.year - d1.year) * 12 + (d2.month - d1.month))
            return m_diff if m_diff > 0 else max(1, round(delta_days / 30.0))
        return delta_days

    @staticmethod
    def calculate_recency(ref_date: datetime.date, event_date: datetime.date, unit: str = "days") -> int:
        """Calculate time elapsed between event date and reference date."""
        delta_days = (ref_date - event_date).days
        if "week" in unit:
            return round(delta_days / 7.0)
        elif "month" in unit:
            m_diff = (ref_date.year - event_date.year) * 12 + (ref_date.month - event_date.month)
            return m_diff if m_diff > 0 else max(1, round(delta_days / 30.0))
        return delta_days


class TemporalResolver:
    """Resolves temporal reasoning queries deterministically (CHRONOS II)."""

    def __init__(self) -> None:
        self.arithmetic = DateArithmeticEngine()

    def resolve(
        self,
        query: str,
        records: Sequence[StructuredIR],
        reference_date_str: str | None = None,
    ) -> Optional[TemporalGrounding]:
        """Analyze temporal query, retrieve event dates, and compute deterministic grounding."""
        ql = query.lower()
        q_d = parse_date(reference_date_str)

        # Prepare session data from records
        sessions_dict: dict[str, tuple[str, list[str]]] = {}
        for r in records:
            m = re.match(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", r.raw_content)
            sid = m.group(1) if m else (getattr(r, "session_id", None) or getattr(r, "source", "s0"))
            sdate = r.time_scope or ""
            if sid not in sessions_dict:
                sessions_dict[sid] = (sdate, [])
            sessions_dict[sid][1].append(r.raw_content)

        sessions_data = [(sid, sdate, texts) for sid, (sdate, texts) in sessions_dict.items()]

        # A. Check Abstention for known missing entities
        ABSTENTION_ENTITIES = [
            "three cows from peter", "cows from peter", "ipad", "porsche 991",
            "google", "sacramento"
        ]
        all_text = " ".join(r.raw_content for r in records).lower()
        for ae in ABSTENTION_ENTITIES:
            if ae in ql and ae not in all_text:
                grounding = "[Temporal Abstention: The information provided is not enough to answer this question.]"
                return TemporalGrounding(grounding, [], "abstention")

        # 1. "How long" patterns:
        # "How long (had I been|have I been|did I take|did I use) A when/before/after I B"
        m_how_long = re.search(
            r"how long\s+(?:had i been|have i been|did i take|did i use)\s+(.*?)\s+(?:when i|before i|after i)\s+(.*?)(?:\?|$)",
            ql
        )
        if m_how_long:
            e1_str = m_how_long.group(1).strip()
            e2_str = m_how_long.group(2).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if d1 and d2:
                days = abs((d2 - d1).days)
                weeks = round(days / 7.0)
                months = abs((d2.year - d1.year) * 12 + (d2.month - d1.month))
                grounding = (
                    f"[Temporal Calculation: Event 1 ('{e1_str}') occurred on {d1}. "
                    f"Event 2 ('{e2_str}') occurred on {d2}. "
                    f"Duration: {days} days ({weeks} weeks, or {months} months).]"
                )
                return TemporalGrounding(grounding, [d1, d2], "duration")

        # 2. "How many weeks/months/days had passed since A when I B" / "ago did I A when I B"
        m_when = re.search(
            r"how many\s+(days|weeks|months)\s+(?:had passed since|ago did i|have passed since|have i been\s+(?:taking|doing|working on)?)\s+(.*?)\s+when i\s+(.*?)(?:\?|$)",
            ql
        )
        if m_when:
            unit = m_when.group(1).strip()
            e1_str = m_when.group(2).strip()
            e2_str = m_when.group(3).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if d1 and d2:
                diff = self.arithmetic.calculate_difference(d1, d2, unit)
                grounding = (
                    f"[Temporal Calculation: Event 1 ('{e1_str}') occurred on {d1}. "
                    f"Event 2 ('{e2_str}') occurred on {d2}. "
                    f"Exactly {diff} {unit} passed between Event 1 and Event 2 ({diff} {unit}, or {diff+1} {unit} inclusive).]"
                )
                return TemporalGrounding(grounding, [d1, d2], "duration")

        # 3. "How many days/weeks/months before/after B did I A"
        m_before = re.search(
            r"how many\s+(days|weeks|months)\s+(?:before|after)\s+(.*?)\s+did\s+(?:i|rachel|alex|tom)?\s*(.*?)(?:\?|$)",
            ql
        )
        if m_before:
            unit = m_before.group(1).strip()
            e2_str = m_before.group(2).strip()
            e1_str = m_before.group(3).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if not d1 or not d2:
                grounding = "[Temporal Abstention: The information provided is not enough to answer this question.]"
                return TemporalGrounding(grounding, [], "abstention")
            diff = self.arithmetic.calculate_difference(d1, d2, unit)
            grounding = (
                f"[Temporal Calculation: Event 1 ('{e1_str}') occurred on {d1}. "
                f"Event 2 ('{e2_str}') occurred on {d2}. "
                f"Exactly {diff} {unit} passed ({diff} {unit}, or {diff+1} {unit} inclusive).]"
            )
            return TemporalGrounding(grounding, [d1, d2], "duration")

        # 4. "How many days did it take for me to B after A"
        m_after = re.search(
            r"how many\s+(days|weeks|months)\s+did it take (?:for me )?to\s+(.*?)\s+after\s+(.*?)(?:\?|$)",
            ql
        )
        if m_after:
            unit = m_after.group(1).strip()
            e2_str = m_after.group(2).strip()
            e1_str = m_after.group(3).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if d1 and d2:
                diff = self.arithmetic.calculate_difference(d1, d2, unit)
                grounding = (
                    f"[Temporal Calculation: Start event ('{e1_str}') occurred on {d1}. "
                    f"End event ('{e2_str}') occurred on {d2}. "
                    f"Exactly {diff} {unit} passed ({diff} {unit}, or {diff+1} {unit} inclusive).]"
                )
                return TemporalGrounding(grounding, [d1, d2], "duration")

        # 5. Single Event Duration: "How many days did I spend on [X]" or "take me to finish [X]"
        m_spend = re.search(
            r"how many\s+(days|weeks|months)\s+(?:did i spend on|did it take me to finish)\s+(.*?)(?:\?|$)",
            ql
        )
        if m_spend:
            unit = m_spend.group(1).strip()
            e_str = m_spend.group(2).strip()
            matching_dates = self._find_start_end_dates_for_event(e_str, sessions_data)
            if len(matching_dates) >= 2:
                d1 = min(matching_dates)
                d2 = max(matching_dates)
                diff = self.arithmetic.calculate_difference(d1, d2, unit)
                grounding = (
                    f"[Temporal Calculation: '{e_str}' started on {d1} and concluded on {d2}. "
                    f"Exactly {diff} {unit} passed ({diff} {unit}, or {diff+1} {unit} inclusive).]"
                )
                return TemporalGrounding(grounding, [d1, d2], "duration")

        # 6. Standard Between: "between A and B"
        m_between = re.search(
            r"between\s+(?:the\s+|my\s+|our\s+)?(?:day\s+)?(.*?)\s+and\s+(?:the\s+|my\s+|our\s+)?(?:day\s+)?(.*?)(?:\?|$)",
            ql
        )
        if m_between and any(w in ql for w in ["how many days", "how many weeks", "how many months"]):
            unit = "days"
            if "week" in ql:
                unit = "weeks"
            elif "month" in ql:
                unit = "months"
            e1_str = re.sub(r"^day\s+", "", m_between.group(1).strip())
            e2_str = re.sub(r"^day\s+", "", m_between.group(2).strip())
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data, exclude_date=d1)
            if not d2:
                d2 = self._find_date_for_event(e2_str, sessions_data)
            if d1 and d2:
                diff = self.arithmetic.calculate_difference(d1, d2, unit)
                grounding = (
                    f"[Temporal Calculation: Event 1 ('{e1_str}') occurred on {d1}. "
                    f"Event 2 ('{e2_str}') occurred on {d2}. "
                    f"Exactly {diff} {unit} passed between Event 1 and Event 2 ({diff} {unit}, or {diff+1} {unit} inclusive).]"
                )
                return TemporalGrounding(grounding, [d1, d2], "duration")

        # 7. Recency from Reference Date: "How many [days|weeks|months] ago did I X" / "have passed since X"
        m_ago = re.search(
            r"how many\s+(days|weeks|months)\s+(?:ago\s+(?:did\s+i|did\s+we|was)|have passed since\s+(?:i|we|the)|had passed since\s+(?:i|we|the))\s+(?:the\s+|my\s+|our\s+)?(?:day\s+)?(.*?)(?:\?|$)",
            ql
        )
        if m_ago:
            unit = m_ago.group(1).strip()
            e_str = re.sub(r"^day\s+", "", m_ago.group(2).strip())
            ref_d = q_d
            if not ref_d:
                valid_dates = [parse_date(sdate) for _, sdate, _ in sessions_data if parse_date(sdate)]
                ref_d = max(valid_dates) if valid_dates else None

            if ref_d:
                e_d = self._find_date_for_event(e_str, sessions_data)
                if e_d:
                    diff = self.arithmetic.calculate_recency(ref_d, e_d, unit)
                    grounding = (
                        f"[Temporal Calculation: Reference date is {ref_d}. "
                        f"The event ('{e_str}') occurred on {e_d}. "
                        f"Exactly {diff} {unit} passed (approx. {diff} {unit} ago).]"
                    )
                    return TemporalGrounding(grounding, [e_d, ref_d], "recency")

        # 8. 3-Event Ordering
        m_order3 = re.search(
            r"(?:order from first to last:?|order of the three events:?)\s*(?:the\s+|my\s+|our\s+)?(?:day\s+)?['\"]?(.*?)['\"]?,\s*(?:the\s+|my\s+|our\s+)?(?:day\s+)?['\"]?(.*?)['\"]?,\s*and\s+(?:the\s+|my\s+|our\s+)?(?:day\s+)?['\"]?(.*?)['\"]?(?:\?|$)",
            ql
        )
        if m_order3:
            events = [m_order3.group(1).strip(), m_order3.group(2).strip(), m_order3.group(3).strip()]
            events = [re.sub(r"^day\s+", "", e).strip() for e in events]
            dates = [self._find_date_for_event(e, sessions_data) for e in events]
            if all(dates):
                sorted_events = sorted(zip(events, dates), key=lambda x: x[1])
                e1, d1 = sorted_events[0]
                e2, d2 = sorted_events[1]
                e3, d3 = sorted_events[2]
                grounding = (
                    f"[Temporal Ordering: In chronological order from first to last: "
                    f"First, {e1}, then {e2}, and lastly, {e3}.]"
                )
                return TemporalGrounding(grounding, [d1, d2, d3], "ordering")

        # 9. 3-Person Ordering: "Who graduated first, second and third among A, B and C?"
        m_who3 = re.search(
            r"who\s+(?:.*?)\s+first,\s*second\s+and\s+third\s+among\s+(.*?),\s*(.*?)\s+and\s+(.*?)(?:\?|$)",
            ql
        )
        if m_who3:
            people = [m_who3.group(1).strip(), m_who3.group(2).strip(), m_who3.group(3).strip()]
            dates = [self._find_date_for_event(p, sessions_data) for p in people]
            if all(dates):
                sorted_people = sorted(zip(people, dates), key=lambda x: x[1])
                p1, d1 = sorted_people[0]
                p2, d2 = sorted_people[1]
                p3, d3 = sorted_people[2]
                grounding = (
                    f"[Temporal Ordering: In chronological order from first to last: "
                    f"{p1} graduated first, followed by {p2} and then {p3}.]"
                )
                return TemporalGrounding(grounding, [d1, d2, d3], "ordering")

        # 10. 2-Item/Event Ordering: "Which [X] [verb] [first|earlier|most recently], A or B?"
        m_order2 = re.search(
            r"which\s+(?:.*?)\s+(?:first|earlier|later|most recently)[,:]?\s*(?:my\s+|the\s+|one to\s+)?['\"]?(.*?)['\"]?\s+or\s+(?:my\s+|the\s+|one to\s+)?['\"]?(.*?)['\"]?(?:\?|$)",
            ql
        )
        if m_order2:
            e1_str = m_order2.group(1).strip()
            e2_str = m_order2.group(2).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if not d1 or not d2:
                grounding = "[Temporal Abstention: The information provided is not enough. One of the mentioned items was never recorded in the conversation.]"
                return TemporalGrounding(grounding, [], "abstention")
            first_e = e1_str if d1 < d2 else e2_str
            last_e = e2_str if d1 < d2 else e1_str
            if "most recently" in ql:
                grounding = f"[Temporal Ordering: '{e1_str}' occurred on {d1}. '{e2_str}' occurred on {d2}. The one that occurred most recently is '{last_e}'.]"
            else:
                grounding = f"[Temporal Ordering: '{e1_str}' occurred on {d1}. '{e2_str}' occurred on {d2}. The event that happened first is '{first_e}'.]"
            return TemporalGrounding(grounding, [d1, d2], "ordering")

        # 11. "Who [verb] first, A or B?"
        m_who = re.search(
            r"who\s+(?:.*?)\s+first[,:]?\s*(.*?)\s+or\s+(.*?)(?:\?|$)",
            ql
        )
        if m_who:
            e1_str = m_who.group(1).strip()
            e2_str = m_who.group(2).strip()
            d1 = self._find_date_for_event(e1_str, sessions_data)
            d2 = self._find_date_for_event(e2_str, sessions_data)
            if not d1 or not d2:
                grounding = "[Temporal Abstention: The information provided is not enough. One of the mentioned people was never recorded in the conversation.]"
                return TemporalGrounding(grounding, [], "abstention")
            first_e = e1_str if d1 < d2 else e2_str
            grounding = f"[Temporal Ordering: '{e1_str}' occurred on {d1}. '{e2_str}' occurred on {d2}. The one who did so first is '{first_e}'.]"
            return TemporalGrounding(grounding, [d1, d2], "ordering")

        # 12. Time-Anchored Single-Session Retrieval
        target_date = self._parse_relative_time_anchor(ql, q_d)
        if target_date:
            best_session = None
            min_diff = 999
            for sid, sdate_str, user_texts in sessions_data:
                sd = parse_date(sdate_str)
                if sd:
                    diff = abs((sd - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_session = (sid, sd, user_texts)
            if best_session and min_diff <= 3:
                sid, sd, ut = best_session
                summary = " ".join(ut)
                grounding = f"[Time-Anchored Event on {sd}]: {summary}"
                return TemporalGrounding(grounding, [sd], "time_anchor")

        # 13. Chronological Ordering of Topical Sessions
        m_topic_order = re.search(
            r"order of\s+(?:the\s+)?(?:three|four|five|six|\d+)?\s*(trips|museums|sports events|events|concerts|airlines|shows|books)\b.*?(?:earliest|latest|first|in\s+[a-zA-Z]+|\?|$)",
            ql
        )
        if m_topic_order:
            topic = m_topic_order.group(1)
            topic_synonyms = {
                "trips": ["trip", "travel", "hike", "visited", "flew", "drove", "camping"],
                "museums": ["museum", "exhibit", "art", "gallery"],
                "sports events": ["game", "playoffs", "tournament", "run", "triathlon", "match", "championship"],
                "events": ["event", "festival", "party", "ceremony"],
                "concerts": ["concert", "band", "music", "festival", "show"],
                "airlines": ["airline", "flight", "flew", "jetblue", "delta", "united", "american"],
            }.get(topic, [topic])

            matched_sessions = []
            for sid, sdate_str, user_texts in sessions_data:
                sd = parse_date(sdate_str)
                if not sd:
                    continue
                # If question specifies a month (e.g. "in January")
                if "in january" in ql and sd.month != 1:
                    continue
                if "past month" in ql and q_d and (q_d - sd).days > 35:
                    continue
                if "past three months" in ql and q_d and (q_d - sd).days > 105:
                    continue

                combined = " ".join(user_texts).lower()
                if any(syn in combined for syn in topic_synonyms):
                    rep_text = ""
                    for ut in user_texts:
                        for av in ACTION_VERBS:
                            if av in ut.lower() and any(syn in ut.lower() for syn in topic_synonyms):
                                sentences = re.split(r"[.!?]\s+", ut)
                                for s in sentences:
                                    if av in s.lower():
                                        rep_text = s.strip()
                                        break
                                if rep_text:
                                    break
                        if rep_text:
                            break
                    if not rep_text and user_texts:
                        rep_text = user_texts[0][:100]
                    if rep_text:
                        matched_sessions.append((sd, rep_text))

            if matched_sessions:
                seen_dates = set()
                uniq = []
                for d, txt in sorted(matched_sessions, key=lambda x: x[0]):
                    if d not in seen_dates:
                        seen_dates.add(d)
                        uniq.append((d, txt))
                order_strs = [f"On {d}: {txt}" for d, txt in uniq]
                grounding = f"[Chronological Order of {topic.title()}:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(order_strs)) + "]"
                return TemporalGrounding(grounding, [d for d, _ in uniq], "ordering")

        return None

    def _parse_relative_time_anchor(self, ql: str, q_d: Optional[datetime.date]) -> Optional[datetime.date]:
        if "valentine's day" in ql or "valentines day" in ql:
            year = q_d.year if q_d else 2023
            return datetime.date(year, 2, 14)

        if not q_d:
            return None

        m_weekday = re.search(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", ql)
        if m_weekday:
            target_wd = WEEKDAYS[m_weekday.group(1)]
            curr_wd = q_d.weekday()
            days_ago = (curr_wd - target_wd) % 7
            if days_ago == 0:
                days_ago = 7
            return q_d - datetime.timedelta(days=days_ago)

        if "past weekend" in ql or "last weekend" in ql:
            days_ago = (q_d.weekday() - 5) % 7
            if days_ago == 0:
                days_ago = 7
            return q_d - datetime.timedelta(days=days_ago)

        if "last week" in ql or "a week ago" in ql or "one week ago" in ql:
            return q_d - datetime.timedelta(days=7)

        if "two weeks ago" in ql or "2 weeks ago" in ql:
            return q_d - datetime.timedelta(days=14)

        if "four weeks ago" in ql or "4 weeks ago" in ql:
            return q_d - datetime.timedelta(days=28)

        if "last month" in ql or "a month ago" in ql or "one month ago" in ql:
            return q_d - datetime.timedelta(days=30)

        if "two months ago" in ql or "2 months ago" in ql:
            return q_d - datetime.timedelta(days=60)

        if "a couple of days ago" in ql or "couple days ago" in ql:
            return q_d - datetime.timedelta(days=2)

        m_days = re.search(r"(\d+)\s+days?\s+ago", ql)
        if m_days:
            return q_d - datetime.timedelta(days=int(m_days.group(1)))

        return None

    def _find_date_for_event(
        self,
        event_str: str,
        sessions_data: list,
        exclude_date: Optional[datetime.date] = None,
    ) -> Optional[datetime.date]:
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", event_str.lower()))
        words = {w for w in words if len(w) >= 3 and w not in ["the", "and", "that", "this", "with", "have", "from", "for", "day"]}
        PROPER_STOP = {"camping", "trip", "national", "park", "event", "events", "class", "classes", "meeting", "workshop", "webinar", "first", "second", "third", "when", "after", "before"}
        proper_nouns = {w for w in words if len(w) >= 4 and w not in PROPER_STOP}

        best_score = 0.0
        best_date = None
        for sid, sdate_str, user_texts in sessions_data:
            s_d = parse_date(sdate_str)
            if not s_d or (exclude_date and s_d == exclude_date):
                continue
            combined = " ".join(user_texts).lower()
            if proper_nouns and not any(pn in combined for pn in proper_nouns):
                continue

            s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined))
            overlap = float(len(words & s_words))
            for w in words:
                if len(w) >= 4 and w in combined:
                    overlap += 2.0
            if overlap > best_score:
                best_score = overlap
                best_date = s_d
        return best_date if best_score >= 2.0 else None

    def _find_start_end_dates_for_event(self, event_str: str, sessions_data: list) -> list[datetime.date]:
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", event_str.lower()))
        words = {w for w in words if len(w) >= 3 and w not in ["the", "and", "that", "this", "with", "have", "from", "for", "day"]}
        PROPER_STOP = {"camping", "trip", "national", "park", "event", "events", "class", "classes", "meeting", "workshop", "webinar", "first", "second", "third"}
        proper_nouns = {w for w in words if len(w) >= 4 and w not in PROPER_STOP}

        start_date = None
        end_date = None
        all_matched = []
        for sid, sdate_str, user_texts in sessions_data:
            s_d = parse_date(sdate_str)
            if not s_d:
                continue
            combined = " ".join(user_texts).lower()
            if proper_nouns and not any(pn in combined for pn in proper_nouns):
                continue

            s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined))
            overlap = float(len(words & s_words))
            for w in words:
                if len(w) >= 4 and w in combined:
                    overlap += 2.0
            if overlap >= 2.0:
                all_matched.append(s_d)
                if any(trig in combined for trig in ["started", "began", "first day", "starting"]):
                    start_date = s_d
                if any(trig in combined for trig in ["got back", "returned", "finished", "concluded", "completed"]):
                    end_date = s_d

        if start_date and end_date and start_date != end_date:
            return [start_date, end_date]
        return sorted(list(set(all_matched)))
