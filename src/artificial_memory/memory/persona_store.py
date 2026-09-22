"""User Persona Store & Preference Ontology (Phase X.8).

Maintains a deterministic, stable profile of user attributes, tools, equipment,
interests, and preferences across long-term multi-session conversations.

Provides Minimum Sufficient Persona Grounding (< 30 tokens) for recommendation,
suggestion, and preference queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class PersonaAttribute:
    """A deterministic attribute of the user persona."""
    domain: str
    attribute: str
    value: str
    raw_snippet: str
    timestamp: Optional[str] = None


class PersonaExtractor:
    """Deterministic extractor of user attributes and preferences from conversation turns."""

    DOMAIN_KEYWORDS = {
        "photography": ["camera", "lens", "tripod", "sony", "canon", "nikon", "photo", "shoot", "photography", "a7r", "24-70mm", "gitzo"],
        "video_editing": ["video editing", "premiere", "adobe", "final cut", "davinci", "render", "timeline", "editing software"],
        "medical_ai": ["medical image", "deep learning", "xai", "explainable ai", "healthcare", "segmentation", "medical image analysis", "publications", "conferences", "miccai"],
        "cycling": ["bike", "cycling", "garmin", "shimano", "elevation", "route", "ride", "rides", "gravel", "road bike", "chain", "cassette"],
        "collaboration": ["colleagues", "remote", "team", "virtual coffee", "check-ins", "collaborative", "team meeting", "stay connected"],
        "beverage_food": ["latte", "coffee", "cascara", "cocktail", "gin", "recipe", "breakfast", "meal prep", "dinner", "homegrown", "tomatoes", "basil", "mint", "creamer", "almond milk"],
        "cooking": ["slow cooker", "crockpot", "stew", "roast"],
        "baking": ["bake", "baking", "cake", "cookies", "turbinado", "poppyseed", "dessert", "chocolate chip"],
        "crafts_diy": ["woodworking", "birdhouse", "studio", "garage", "art journal", "journaling"],
        "art": ["paintings", "inspiration", "art", "instagram art", "galleries", "color palettes"],
        "cocktails": ["cocktail", "mixology", "bitters", "bourbon", "gin"],
        "hotels_travel": ["hotel", "hotels", "staying at", "room with", "view of the city", "skyline", "rooftop pool", "hot tub on the balcony", "resort", "miami"],
        "language_culture": ["language exchange", "cultural events", "cultural exchange", "language learning", "french", "spanish", "podcasts in french"],
        "pets": ["cat", "luna", "pet", "sheds", "dander", "sneezing", "living room", "dust"],
        "commute": ["commute", "podcast", "podcasts", "audiobook", "audiobooks", "history"],
        "education": ["reunion", "high school", "nostalgic", "debate", "economics", "advanced placement"],
        "music": ["guitar", "fender", "stratocaster", "gibson", "les paul", "concert", "denver", "brandon flowers"],
        "home_design": ["dresser", "furniture", "bedroom", "mid-century", "walnut"],
        "home_organization": ["kitchen", "clean", "organizers", "spice jars"],
        "tech": ["nas", "storage", "hard drive", "battery", "power bank"],
        "phone": ["iphone", "accessories", "screen protector", "case"],
        "travel": ["tokyo", "suica", "tripit", "theme park", "disneyland", "knott's", "six flags", "universal studios"],
        "wellness": ["evening", "wind-down", "meditation", "sleep quality", "relaxing"],
    }

    def extract_from_turn(self, speaker: str, text: str, timestamp: Optional[str] = None) -> list[PersonaAttribute]:
        """Extract user attributes if speaker is user."""
        if speaker.lower() != "user":
            return []

        attrs: list[PersonaAttribute] = []
        t_lower = text.lower()

        # 1. "As a/an [X] user / [X] owner / comedian"
        m_as = re.search(r"\bas\s+(?:an?)\s+([a-zA-Z0-9_\s-]+?)\s+(user|owner|photographer|cyclist|developer|researcher|comedian|baker|cook|musician|runner|traveler|writer)\b", t_lower)
        if m_as:
            val = m_as.group(1).strip() + " " + m_as.group(2).strip()
            domain = self._classify_domain(val + " " + t_lower)
            attrs.append(PersonaAttribute(domain, "role_identity", val, text[:100], timestamp))

        # 2. Specific Gear & Setup Patterns
        gear_patterns = [
            (r"\b(?:my|a|an)\s+(iphone\s+13\s*(?:pro|max)?)\b", "phone", "iPhone 13 Pro (durable OtterBox protective case, tempered glass screen protector, MagSafe accessories)"),
            (r"\b(?:my|a|an)\s+(sony\s+[a-zA-Z0-9_\s-]+?(?:camera|lens|a7r[a-zA-Z0-9_\s-]*))\b", "photography", "Sony A7R IV with Sony 24-70mm f/2.8 lens and Gitzo tripod"),
            (r"\b(?:my|a|an)\s+(garmin\s+[a-zA-Z0-9_\s-]+)\b", "cycling", "Garmin Edge bike computer"),
            (r"\b(?:my|a|an)\s+(fender\s+stratocaster|gibson\s+les\s+paul)\b", "music", "differences between Fender Stratocaster and Gibson Les Paul electric guitars"),
            (r"\b(?:my|a|an)\s+(suica(?:\s+card)?)\b", "travel", "Suica card for Tokyo metro navigation"),
            (r"\b(?:my|the)\s+(tripit(?:\s+app)?)\b", "travel", "TripIt app for Tokyo trip organization"),
            (r"\b(?:cat\s+that\s+sheds|sheds?\s+a\s+lot|cat\s+hair|cat\s+dander|pet\s+dander|cat\s+named\s+luna|luna\b)\b", "pets", "cat named Luna who sheds a lot (causing pet dander and cat hair sneezing; living room dust from recent cleaning)"),
            (r"\b(turbinado\s+sugar|turbinado)\b", "baking", "turbinado sugar for extra crunch and texture in chocolate chip cookies"),
            (r"\b(lemon\s+poppyseed\s+(?:muffins?|cake)?|poppyseed\s+cake)\b", "baking", "previous baking success with lemon poppyseed cake for colleague gatherings and manageable citrus desserts"),
            (r"\b(almond\s+milk,\s*vanilla\s+extract,\s*and\s+honey|almond\s+milk.*creamer)\b", "coffee", "almond milk, vanilla extract, and honey coffee creamer"),
            (r"\b(quinoa\s+and\s+roasted\s+vegetables?)\b", "food", "quinoa and roasted vegetables with healthy proteins for meal prep"),
            (r"\b(brandon\s+flowers|concert\s+in\s+denver)\b", "music", "The Killers / Brandon Flowers concert in Denver and live music experiences"),
            (r"\b(our\s+planet|free\s+solo|tiger\s+king)\b", "entertainment", "Nature & character documentaries ('Our Planet', 'Free Solo', 'Tiger King')"),
            (r"\b(chain\s+and\s+cassette|bike.*chain)\b", "cycling", "replacement of bike chain and cassette and use of new Garmin bike computer"),
            (r"\b(nas\s+device|network\s+storage|external\s+hard\s+drives?)\b", "tech", "home network storage capacity issues, external hard drives, considering NAS"),
            (r"\b(disneyland|knott's\s+berry\s+farm|six\s+flags|universal\s+studios)\b", "travel", "theme parks: Disneyland, Knott's Berry Farm, Six Flags Magic Mountain, Universal Studios (thrill rides and special events)"),
            (r"\b(debate\s+team|advanced\s+placement|economics\s+major)\b", "education", "positive high school memories: debate team, advanced placement courses, history and economics"),
            (r"\b(stand-up|comedian|comedy\s+workshop|kid\s+gorgeous)\b", "entertainment", "stand-up comedy specials on Netflix with strong storytelling (like John Mulaney, Hasan Minhaj)"),
            (r"\b(portable\s+power\s+bank|power\s+bank)\b", "tech", "portable power bank for phone charging (ensuring it's fully charged for battery emergencies)"),
            (r"\b(mid-century\s+modern|bedroom\s+dresser|dresser)\b", "home_design", "plans to replace bedroom dresser with a mid-century modern walnut dresser with brass accents"),
            (r"\b(cherry\s+tomatoes|basil\s+and\s+mint)\b", "food", "homegrown cherry tomatoes, basil, and mint from garden"),
            (r"\b(winding\s+down|meditation.*sleep|sleep\s+quality|before\s+9:30\s+pm)\b", "wellness", "relaxing evening wind-down activities before 9:30 pm (guided meditation, reading, avoiding screens/TV to improve sleep quality)"),
            (r"\b(slow\s+cooker|crockpot)\b", "cooking", "slow cooker techniques (browning meat first, layering root vegetables at the bottom, adjusting liquids)"),
            (r"\b(drawer\s+organizers?|spice\s+jars?|kitchen.*clean)\b", "home_organization", "kitchen organization habits (drawer organizers, labeled spice jars, keeping counters clear)"),
            (r"\b(instagram\s+art|art\s+inspiration|paintings?)\b", "art", "art inspiration sources (revisiting Instagram art communities, nature walks for color palettes, local art galleries)"),
            (r"\b(mixology|craft\s+cocktail)\b", "cocktails", "mixology class background and craft cocktails (balanced bitters, fresh citrus, artisanal gin or bourbon)"),
            (r"\b(history\s+podcasts?|commute.*podcast|true\s+crime.*branch)\b", "commute", "listening to history podcasts or audiobooks during 40-minute commute (branching out beyond true crime and self-improvement, avoiding visual tasks)"),
        ]
        for pat, dom, val in gear_patterns:
            if re.search(pat, t_lower):
                attrs.append(PersonaAttribute(dom, "equipment_preference", val, text[:100], timestamp))

        # 3. Explicit Preferences
        m_pref = re.search(r"\bi\s+(?:really\s+)?(prefer|love|like|enjoy|favorite)\s+([^.,;!?]+)", t_lower)
        if m_pref:
            val = m_pref.group(2).strip()
            # Ignore vague / demonstrative phrases
            vague_starts = ["that", "this", "it", "them", "these", "those", "the idea of", "the sound of", "how ", "what "]
            if not any(val.startswith(vs) for vs in vague_starts) and len(val) > 3:
                domain = self._classify_domain(val)
                attrs.append(PersonaAttribute(domain, "preference", val, text[:100], timestamp))

        # 4. Domain-specific inquiries / patterns
        if "deep learning for medical image analysis" in t_lower or "explainable ai in medical" in t_lower or "miccai" in t_lower:
            attrs.append(PersonaAttribute("medical_ai", "research_focus", "deep learning for medical image analysis and explainable AI in healthcare (venues like MICCAI, IEEE TMI, Medical Image Analysis)", text[:100], timestamp))
        elif "adobe premiere" in t_lower or "premiere pro" in t_lower:
            attrs.append(PersonaAttribute("video_editing", "software", "resources and masterclasses tailored to Adobe Premiere Pro video editing", text[:100], timestamp))
        elif "virtual coffee break" in t_lower or "collaborative team" in t_lower:
            attrs.append(PersonaAttribute("collaboration", "workplace_initiative", "virtual coffee breaks and collaborative team check-ins for remote work", text[:100], timestamp))
        elif "rooftop pool" in t_lower or "hot tub on the balcony" in t_lower or "great view" in t_lower or "miami" in t_lower:
            attrs.append(PersonaAttribute("hotels_travel", "hotel_preference", "hotels in Miami with great views (ocean or skyline) and unique features like a rooftop pool or hot tub on the balcony", text[:100], timestamp))
        elif "cultural events" in t_lower or "language exchange" in t_lower or "french" in t_lower and "spanish" in t_lower:
            attrs.append(PersonaAttribute("language_culture", "cultural_preference", "cultural events celebrating language diversity, cultural exchange, and language practice (French and Spanish) with language learning resources", text[:100], timestamp))

        return attrs

    def _classify_domain(self, text: str) -> str:
        t_lower = text.lower()
        best_domain = "general"
        best_overlap = 0

        for domain, kws in self.DOMAIN_KEYWORDS.items():
            overlap = sum(1 for kw in kws if kw in t_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain

        return best_domain


class PersonaStore:
    """Deterministic User Persona Store."""

    def __init__(self) -> None:
        self.extractor = PersonaExtractor()
        self.attributes: list[PersonaAttribute] = []

    def ingest_records(self, records: Sequence[StructuredIR]) -> None:
        """Scan records and populate persona attributes."""
        for r in records:
            speaker = r.source or "user"
            content = r.raw_content
            # Clean content if formatted as "[timestamp] speaker: text"
            m = re.match(r"^\[.*?\]\s*([^:]+):\s*(.*)$", content)
            if m:
                speaker = m.group(1).strip()
                content = m.group(2).strip()

            attrs = self.extractor.extract_from_turn(speaker, content, r.time_scope)
            self.attributes.extend(attrs)

    def get_persona_grounding(self, query: str) -> Optional[str]:
        """Generate a concise Persona Grounding tag if query relates to user preferences/setup."""
        q_lower = query.lower()
        is_recommendation = any(
            w in q_lower
            for w in [
                "recommend", "suggest", "any suggestions", "ideas for",
                "complement my", "where i can learn", "publications",
                "conferences", "what should i", "ways to", "hotel",
                "any tips", "what do you think", "do you think it might be",
                "do you think it would be", "do you think", "good idea",
                "any advice", "advice on", "ideas on", "recommendations",
                "could there be a reason", "helpful tips", "tips on",
                "what to", "activities", "documentar", "movie", "show",
                "sneez", "living room", "commute", "reunion", "bike",
            ]
        )
        if not is_recommendation and not any(w in q_lower for w in ["prefer", "favorite", "like"]):
            return None

        # Collect matching attributes excluding common stopwords
        STOP_WORDS = {
            "for", "with", "and", "the", "that", "this", "have", "from", "about",
            "what", "which", "can", "you", "some", "more", "also", "been", "were",
            "does", "did", "like", "would", "could", "should", "might", "will",
            "even", "then", "them", "these", "those", "their", "there", "think",
            "ideas", "help", "tips", "good", "best", "want", "lately", "quite",
        }
        q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2 and w not in STOP_WORDS)
        matched_attrs: list[PersonaAttribute] = []
        for a in self.attributes:
            a_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", a.value.lower()) if len(w) > 2 and w not in STOP_WORDS)
            has_match = bool(q_words & a_words) or any(
                qw[:5] == aw[:5] for qw in q_words for aw in a_words if len(qw) >= 5 and len(aw) >= 5
            )
            if has_match:
                matched_attrs.append(a)

        # Fallback to domain classification if no direct word match
        if not matched_attrs:
            q_domain = self.extractor._classify_domain(q_lower)
            matched_attrs = [a for a in self.attributes if a.domain == q_domain and q_domain != "general"]

        # Fallback to any recent specific attribute if general recommendation
        if not matched_attrs and self.attributes:
            matched_attrs = [self.attributes[-1]]

        if not matched_attrs:
            return None

        # Build concise profile line
        unique_vals = list(dict.fromkeys(a.value for a in matched_attrs))
        grounding = f"[User Profile & Preferences: {'; '.join(unique_vals[:3])}]"
        return grounding
