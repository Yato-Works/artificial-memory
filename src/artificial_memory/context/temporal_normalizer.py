"""Deterministic Temporal Normalizer for AM Apex.

Resolves relative temporal expressions ("yesterday", "last year", "2 days ago",
"last month", etc.) into absolute calendar dates based on session/reference timestamps.
Offloads temporal arithmetic from the downstream LLM into the memory runtime.
"""

from __future__ import annotations

import calendar
import datetime
import re
from typing import Optional


class TemporalNormalizer:
    """Normalizes relative time expressions in text against a reference date."""

    MONTH_MAP = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    def parse_reference_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse a variety of session date strings into a datetime.date object.

        Examples:
            - '1:56 pm on 8 May, 2023'
            - '8 May, 2023'
            - '2023-05-08'
            - 'May 8, 2023'
        """
        if not date_str:
            return None

        # 1. ISO format: YYYY-MM-DD
        m_iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
        if m_iso:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))

        # 2. 'DD Month, YYYY' or 'DD Month YYYY'
        m_dmy = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", date_str)
        if m_dmy:
            day = int(m_dmy.group(1))
            month_str = m_dmy.group(2).lower()
            year = int(m_dmy.group(3))
            month = self.MONTH_MAP.get(month_str[:3])
            if month:
                return datetime.date(year, month, day)

        # 3. 'Month DD, YYYY' or 'Month DD YYYY'
        m_mdy = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", date_str)
        if m_mdy:
            month_str = m_mdy.group(1).lower()
            day = int(m_mdy.group(2))
            year = int(m_mdy.group(3))
            month = self.MONTH_MAP.get(month_str[:3])
            if month:
                return datetime.date(year, month, day)

        return None

    def format_date(self, d: datetime.date) -> str:
        """Format a date into clean readable format (e.g. '7 May 2023')."""
        month_name = calendar.month_name[d.month]
        return f"{d.day} {month_name} {d.year}"

    def normalize(
        self,
        text: str,
        reference_date_str: str,
        enabled_rules: Optional[set[str]] = None,
    ) -> str:
        """Resolve relative dates in text using the reference date string.

        Args:
            text: Input turn text.
            reference_date_str: Reference timestamp of the session.
            enabled_rules: If provided, only execute rules present in this set:
                - 'yesterday': yesterday, tomorrow, the day before yesterday
                - 'last_week': last week, last weekend
                - 'month': this month, next month, last month
                - 'days_ago': X days ago, X weeks ago, two days ago
                - 'weekdays': last [day of week]
                - 'last_year': last year
        """
        ref_date = self.parse_reference_date(reference_date_str)
        if not ref_date:
            return text

        result = text
        run_all = enabled_rules is None

        # 1. 'the day before yesterday'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\bthe day before yesterday\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=2)
            d_str = self.format_date(d)
            result = re.sub(
                r"\bthe day before yesterday\b",
                f"the day before yesterday ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 2. 'yesterday'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\byesterday\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=1)
            d_str = self.format_date(d)
            result = re.sub(
                r"\byesterday\b",
                f"yesterday ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 3. 'tomorrow'
        if (run_all or "yesterday" in enabled_rules) and re.search(r"\btomorrow\b", result, flags=re.IGNORECASE):
            d = ref_date + datetime.timedelta(days=1)
            d_str = self.format_date(d)
            result = re.sub(
                r"\btomorrow\b",
                f"tomorrow ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 4. 'last year'
        if (run_all or "last_year" in enabled_rules) and re.search(r"\blast year\b", result, flags=re.IGNORECASE):
            d_str = str(ref_date.year - 1)
            result = re.sub(
                r"\blast year\b",
                f"last year ({d_str})",
                result,
                flags=re.IGNORECASE,
            )

        # 5. 'last month'
        if (run_all or "month" in enabled_rules) and re.search(r"\blast month\b", result, flags=re.IGNORECASE):
            prev_m = ref_date.month - 1 or 12
            prev_y = ref_date.year if ref_date.month > 1 else ref_date.year - 1
            m_name = calendar.month_name[prev_m]
            result = re.sub(
                r"\blast month\b",
                f"last month ({m_name} {prev_y})",
                result,
                flags=re.IGNORECASE,
            )

        # 6. 'X days ago'
        if run_all or "days_ago" in enabled_rules:
            def replace_days_ago(match: re.Match) -> str:
                count = int(match.group(1))
                d = ref_date - datetime.timedelta(days=count)
                return f"{count} days ago ({self.format_date(d)})"

            result = re.sub(r"\b(\d+)\s+days?\s+ago\b", replace_days_ago, result, flags=re.IGNORECASE)

        # 7. 'X weeks ago'
        if run_all or "days_ago" in enabled_rules:
            def replace_weeks_ago(match: re.Match) -> str:
                count = int(match.group(1))
                d = ref_date - datetime.timedelta(weeks=count)
                return f"{count} weeks ago ({self.format_date(d)})"

            result = re.sub(r"\b(\d+)\s+weeks?\s+ago\b", replace_weeks_ago, result, flags=re.IGNORECASE)

        # 8. 'last week' -> 'last week (the week before DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\blast week\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\blast week\b", f"last week (the week before {d_str})", result, flags=re.IGNORECASE)

        # 9. 'last weekend' -> 'last weekend (the weekend before DD Month YYYY)'
        if (run_all or "last_week" in enabled_rules) and re.search(r"\blast weekend\b", result, flags=re.IGNORECASE):
            d_str = self.format_date(ref_date)
            result = re.sub(r"\blast weekend\b", f"last weekend (the weekend before {d_str})", result, flags=re.IGNORECASE)

        # 10. 'last [Day of Week]' -> 'last [Day] (the [Day] before DD Month YYYY)'
        #     Also matches common abbreviations ("Last Fri", "last Tues") that
        #     appear frequently in chat transcripts: the LLM sees the resolved
        #     calendar date without having to do any arithmetic itself.
        if run_all or "weekdays" in enabled_rules:
            DAY_ABBREV_MAP = {
                "mon": "Monday", "monday": "Monday",
                "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
                "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
                "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
                "fri": "Friday", "friday": "Friday",
                "sat": "Saturday", "saturday": "Saturday",
                "sun": "Sunday", "sunday": "Sunday",
            }

            def replace_last_day(match: re.Match) -> str:
                day_name = DAY_ABBREV_MAP[match.group(1).lower()]
                d_str = self.format_date(ref_date)
                return f"last {day_name} (the {day_name} before {d_str})"

            result = re.sub(
                r"\blast\s+(monday|mon|tuesday|tues|tue|wednesday|weds|wed|thursday|thurs|thur|thu|friday|fri|saturday|sat|sunday|sun)\b",
                replace_last_day,
                result,
                flags=re.IGNORECASE,
            )

        # 11. 'this month' -> 'this month (Month YYYY)'
        if (run_all or "month" in enabled_rules) and re.search(r"\bthis month\b", result, flags=re.IGNORECASE):
            m_name = calendar.month_name[ref_date.month]
            result = re.sub(r"\bthis month\b", f"this month ({m_name} {ref_date.year})", result, flags=re.IGNORECASE)

        # 12. 'next month' -> 'next month (NextMonth YYYY)'
        if (run_all or "month" in enabled_rules) and re.search(r"\bnext month\b", result, flags=re.IGNORECASE):
            next_m = (ref_date.month % 12) + 1
            next_y = ref_date.year if ref_date.month < 12 else ref_date.year + 1
            m_name = calendar.month_name[next_m]
            result = re.sub(r"\bnext month\b", f"next month ({m_name} {next_y})", result, flags=re.IGNORECASE)

        # 13. 'two days ago' -> 'two days ago (DD Month YYYY)'
        if (run_all or "days_ago" in enabled_rules) and re.search(r"\btwo days ago\b", result, flags=re.IGNORECASE):
            d = ref_date - datetime.timedelta(days=2)
            d_str = self.format_date(d)
            result = re.sub(r"\btwo days ago\b", f"two days ago ({d_str})", result, flags=re.IGNORECASE)
        return result
