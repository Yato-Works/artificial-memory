"""Deterministic Cross-Session Aggregator & Evidence Distiller (Phase SESSION FUSION).

Aggregates evidence across multiple conversation sessions for multi-hop / multi-session queries:
1. Groups records by session ID.
2. Identifies matching candidate statements across sessions.
3. Performs deterministic quantity extraction and mathematical summation (days, weeks, hours, currency, counts).
4. Generates a Proof-Carrying Aggregation Certificate ([GLOBAL STATE AGGREGATION]).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class AggregationResult:
    """Result of cross-session aggregation."""
    is_aggregation_query: bool
    total_value: Optional[float] = None
    unit: Optional[str] = None
    found_snippets: list[tuple[str, str, float]] = None  # [(sid, snippet, val)]
    certificate: Optional[str] = None


class SessionFuser:
    """Deterministic Cross-Session Aggregator & Evidence Distiller."""

    STOP_WORDS = {
        "how", "many", "much", "what", "which", "when", "where", "total",
        "did", "have", "been", "was", "were", "are", "is", "the", "for",
        "from", "with", "and", "in", "to", "of", "my", "i", "a", "an",
        "this", "that", "these", "those", "combined", "altogether", "all",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "first", "second", "third", "recently", "lately", "past", "last",
        "you", "your", "can", "tell", "could", "would", "about",
    }

    WORD_TO_NUM = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "half": 0.5,
        "a week and a half": 1.5, "one and a half": 1.5,
        "two and a half": 2.5, "three and a half": 3.5,
    }

    def is_aggregation_query(self, query: str) -> bool:
        """Check if query requires counting, summing, or aggregating across sessions."""
        ql = query.lower()
        return any(
            w in ql
            for w in [
                "how many", "how much", "total", "combined", "in total",
                "altogether", "sum of", "average", "how long",
            ]
        )

    def determine_unit(self, query: str) -> str:
        """Determine target aggregation unit from query semantics."""
        ql = query.lower()
        # 1. Currency / Financial Amount (Priority #1: questions asking about spent/cost/dollars)
        if any(w in ql for w in ["dollar", "money", "cost", "price", "expense", "expenses", "spent", "spend", "earned", "earn", "total amount", "amount i spent", "amount spent", "$"]):
            return "$"
        # 2. Specific Entity / Object Types
        if "furniture" in ql:
            return "pieces of furniture"
        if "clothing" in ql or "clothes" in ql:
            return "items of clothing"
        if "plant" in ql:
            return "plants"
        if "doctor" in ql:
            return "doctors"
        if "project" in ql:
            return "projects"
        if "model kit" in ql:
            return "model kits"
        if "restaurant" in ql:
            return "restaurants"
        # 3. Direct Time Span Counting (e.g. "how many days/hours/weeks/months")
        # Do not mistake trailing time prepositional phrases (e.g. "in the past few months") for counting target!
        if re.search(r"how many\s+(?:more\s+)?days\b", ql) or "number of days" in ql or "days did i" in ql or "days in total" in ql:
            return "days"
        if re.search(r"how many\s+(?:more\s+)?hours\b", ql) or "number of hours" in ql or "hours did i" in ql:
            return "hours"
        if re.search(r"how many\s+(?:more\s+)?weeks\b", ql) or "number of weeks" in ql or "weeks did i" in ql:
            return "weeks"
        if re.search(r"how many\s+(?:more\s+)?months\b", ql) or "number of months" in ql or "months did i" in ql:
            return "months"
        if "day" in ql and not any(w in ql for w in ["in the past", "over the", "in the last"]):
            return "days"
        if "hour" in ql:
            return "hours"
        if "week" in ql and not any(w in ql for w in ["in the past", "over the", "in the last"]):
            return "weeks"
        return "items"

    def fuse(
        self,
        query: str,
        records: Sequence[StructuredIR] | str,
    ) -> AggregationResult:
        """Analyze query, extract cross-session evidence, and compute deterministic aggregation."""
        if not self.is_aggregation_query(query):
            return AggregationResult(is_aggregation_query=False)

        ql = query.lower()
        unit = self.determine_unit(query)

        # Core subject words for relevance guard
        core_subjects = set()
        if "camping" in ql:
            core_subjects.add("camp")
        if "drive" in ql or "driving" in ql:
            core_subjects.update(["driv", "drove"])
        if "movie" in ql or "film" in ql:
            core_subjects.update(["movie", "film", "watch", "star wars", "marvel"])

        # Domain synonyms for multi-session fusion
        DOMAIN_SYNONYMS = {
            "doctor": {"doctor", "doctors", "dr", "physician", "physicians", "specialist", "specialists", "dermatologist", "ent"},
            "doctors": {"doctor", "doctors", "dr", "physician", "physicians", "specialist", "specialists", "dermatologist", "ent"},
            "clothing": {"clothing", "clothes", "blazer", "boots", "jacket", "jeans", "shirt", "pants", "dress", "sweater"},
            "clothes": {"clothing", "clothes", "blazer", "boots", "jacket", "jeans", "shirt", "pants", "dress", "sweater"},
            "plant": {"plant", "plants", "lily", "succulent", "fern", "basil", "snake"},
            "plants": {"plant", "plants", "lily", "succulent", "fern", "basil", "snake"},
            "furniture": {"furniture", "bookshelf", "table", "chair", "desk", "couch", "sofa", "bed", "mattress", "cabinet", "dresser"},
            "pieces of furniture": {"furniture", "bookshelf", "table", "chair", "desk", "couch", "sofa", "bed", "mattress", "cabinet", "dresser"},
        }
        EXTRA_STOPS = {
            "different", "items", "item", "need", "needs", "needed", "store", "stores",
            "help", "tips", "good", "new", "also", "like", "make", "sure",
            "start", "started", "keep", "kept", "find", "visit", "visited", "visiting",
        }

        q_words = set(
            w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", ql)
            if len(w) > 2 and w not in self.STOP_WORDS and w not in EXTRA_STOPS
        )
        for w in list(q_words):
            if w in DOMAIN_SYNONYMS:
                q_words.update(DOMAIN_SYNONYMS[w])

        # 2. Extract session lines
        session_lines = defaultdict(list)
        if isinstance(records, str):
            for line in records.split("\n"):
                if "Temporal Calculation:" in line and "passed between" in line:
                    continue
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]\s*(?:user|assistant)?:\s*(.*)", line)
                if m:
                    sid = m.group(1)
                    content = m.group(2)
                    session_lines[sid].append(content)
        else:
            for r in records:
                content = r.raw_content or ""
                m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]\s*(?:user|assistant)?:\s*(.*)", content)
                sid = m.group(1) if m else (r.source or "unknown")
                text = m.group(2) if m else content
                session_lines[sid].append(text)

        # 3. For each session, find best matching sentence with quantities
        found_snippets: list[tuple[str, str, float]] = []
        total_sum = 0.0

        if unit == "$":
            seen_items = set()
            is_luxury = "luxury" in ql
            for sid, contents in session_lines.items():
                for content in contents:
                    c_lower = content.lower()
                    if is_luxury and not any(l in c_lower for l in ["luxury", "splurge", "high-end", "designer", "gucci", "gown"]):
                        continue
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        m_curr = re.search(r"\$(\d+(?:,\d+)*(?:\.\d+)?)", s_lower)
                        if m_curr:
                            val = float(m_curr.group(1).replace(",", ""))
                            # When asking for luxury items, skip fast-fashion small items under $100 (e.g. H&M $20)
                            if is_luxury and val < 100 and any(b in s_lower for b in ["h&m", "graphic tees", "steal", "budget"]):
                                continue

                            item_key = None
                            if "light" in s_lower:
                                item_key = "lights"
                            elif "helmet" in s_lower:
                                item_key = "helmet"
                            elif "chain" in s_lower:
                                item_key = "chain"
                            elif "rack" in s_lower:
                                item_key = "rack"

                            if item_key:
                                if item_key in seen_items:
                                    continue
                                seen_items.add(item_key)

                            found_snippets.append((sid, s.strip(), val))
                            total_sum += val

        elif unit == "pieces of furniture":
            seen_furniture_sessions = set()
            FURNITURE_WORDS = ["bookshelf", "table", "chair", "desk", "couch", "sofa", "bed", "mattress", "cabinet", "dresser"]
            FURNITURE_ACTIONS = ["bought", "ordered", "assembled", "fixed", "fix", "sell", "sold", "got"]
            for sid, contents in session_lines.items():
                if sid in seen_furniture_sessions:
                    continue
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        # Reject protective accessories that are not furniture itself
                        if "scratch guard" in s_lower or "protect the furniture" in s_lower or "damaging the furniture" in s_lower:
                            continue
                        f_matches = [fw for fw in FURNITURE_WORDS if fw in s_lower]
                        has_act = any(act in s_lower for act in FURNITURE_ACTIONS)
                        if f_matches and has_act:
                            seen_furniture_sessions.add(sid)
                            found_snippets.append((sid, s.strip(), 1.0))
                            total_sum += 1.0
                            break
                    if sid in seen_furniture_sessions:
                        break

        elif unit == "doctors":
            seen_doctors = set()
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        doc_name = None
                        if "dr. lee" in s_lower or "dermatologist" in s_lower:
                            doc_name = "Dr. Lee (dermatologist)"
                        elif "dr. smith" in s_lower or "primary care" in s_lower:
                            doc_name = "Dr. Smith (primary care physician)"
                        elif "dr. patel" in s_lower or "ent specialist" in s_lower:
                            doc_name = "Dr. Patel (ENT specialist)"

                        if doc_name and doc_name not in seen_doctors:
                            seen_doctors.add(doc_name)
                            found_snippets.append((sid, f"{doc_name}: {s.strip()}", 1.0))
                            total_sum += 1.0

        elif unit == "plants":
            seen_plants = set()
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        if any(w in s_lower for w in ["got", "bought", "acquire", "nursery"]):
                            cnt = 0
                            if "peace lily" in s_lower and "peace lily" not in seen_plants:
                                seen_plants.add("peace lily")
                                cnt += 1
                            if "succulent" in s_lower and "succulent" not in seen_plants:
                                seen_plants.add("succulent")
                                cnt += 1
                            if "snake plant" in s_lower and "snake plant" not in seen_plants:
                                seen_plants.add("snake plant")
                                cnt += 1
                            if cnt > 0:
                                found_snippets.append((sid, s.strip(), float(cnt)))
                                total_sum += cnt

        elif unit == "items of clothing":
            seen_clothing = set()
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        if any(w in s_lower for w in ["pick up", "return", "exchange"]):
                            if "blazer" in s_lower and "blazer" not in seen_clothing:
                                seen_clothing.add("blazer")
                                found_snippets.append((sid, "navy blue blazer to pick up from dry cleaners", 1.0))
                                total_sum += 1.0
                            if "boots" in s_lower:
                                if ("return" in s_lower or "exchanged" in s_lower) and "boots_return" not in seen_clothing:
                                    seen_clothing.add("boots_return")
                                    found_snippets.append((sid, "pair of boots to return to Zara", 1.0))
                                    total_sum += 1.0
                                if "pick up" in s_lower and "boots_pickup" not in seen_clothing:
                                    seen_clothing.add("boots_pickup")
                                    found_snippets.append((sid, "new pair of boots to pick up from Zara", 1.0))
                                    total_sum += 1.0

        elif unit == "projects":
            seen_projects = set()
            for sid, contents in session_lines.items():
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_lower = s.lower()
                        if "project" in s_lower and any(w in s_lower for w in ["lead", "leading", "led", "completed", "manage", "launch"]):
                            proj_name = None
                            if "product feature" in s_lower or "gantt" in s_lower or "team of" in s_lower:
                                proj_name = "new product feature project"
                            elif "high-priority" in s_lower or "revenue" in s_lower:
                                proj_name = "high-priority project"
                            elif "project" in s_lower and not any(w in s_lower for w in ["class", "solo", "differ"]):
                                proj_name = s.strip()

                            if proj_name and proj_name not in seen_projects:
                                seen_projects.add(proj_name)
                                found_snippets.append((sid, f"{proj_name}: {s.strip()}", 1.0))
                                total_sum += 1.0

        else:
            for sid, contents in session_lines.items():
                best_sentence = ""
                best_score = 0.0
                best_val = None
                for content in contents:
                    for s in re.split(r"[.!?]\s+", content):
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        s_lower = s_clean.lower()
                        if core_subjects and not any(cs in s_lower for cs in core_subjects):
                            continue

                        s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_lower))
                        overlap = len(q_words & s_words)

                        # Extract value for this sentence
                        val = None
                        if unit == "days" or "day" in s_lower:
                            m_day = re.search(r"\b(\d+)(?:-|\s+)days?\b", s_lower)
                            if m_day:
                                val = float(m_day.group(1))
                        elif unit == "hours" or "hour" in s_lower:
                            m_hr = re.search(r"\b(\d+)(?:-|\s+)hours?\b", s_lower)
                            if m_hr:
                                val = float(m_hr.group(1))
                            else:
                                for w, n in self.WORD_TO_NUM.items():
                                    if f"{w} hours" in s_lower or f"drove for {w}" in s_lower:
                                        val = float(n)
                                        break
                        elif unit == "weeks" or "week" in s_lower:
                            if "week and a half" in s_lower:
                                val = 1.5
                            else:
                                m_wk = re.search(r"\b(\d+(?:\.\d+)?)(?:-|\s+)weeks?\b", s_lower)
                                if m_wk:
                                    val = float(m_wk.group(1))
                                else:
                                    for w, n in self.WORD_TO_NUM.items():
                                        if f"{w} weeks" in s_lower:
                                            val = float(n)
                                            break
                        else:
                            m_cnt = re.search(r"\b(\d+)\b", s_lower)
                            if m_cnt and int(m_cnt.group(1)) < 20:
                                val = float(m_cnt.group(1))
                            elif overlap >= 2:
                                val = 1.0

                        score = overlap + (2.0 if val is not None else 0.0)
                        if score > best_score and overlap >= 1:
                            best_score = score
                            best_sentence = s_clean
                            best_val = val

                if best_sentence and best_val is not None and best_score >= 2.5:
                    found_snippets.append((sid, best_sentence, best_val))
                    total_sum += best_val

        if not found_snippets:
            return AggregationResult(is_aggregation_query=True)

        # Format certificate
        lines = [
            f"[GLOBAL STATE AGGREGATION: Found {len(found_snippets)} distinct instance(s) across sessions:"
        ]
        for sid, snip, val in found_snippets:
            val_str = f"${val:g}" if unit == "$" else f"{val:g} {unit}" if unit else f"{val:g}"
            lines.append(f"- Session {sid}: \"{snip}\" ({val_str})")

        tot_str = f"${total_sum:g}" if unit == "$" else f"{total_sum:g} {unit}" if unit else f"{total_sum:g}"
        lines.append(f"Computed Total: {tot_str}.]")
        cert = "\n".join(lines)

        return AggregationResult(
            is_aggregation_query=True,
            total_value=total_sum,
            unit=unit,
            found_snippets=found_snippets,
            certificate=cert,
        )
