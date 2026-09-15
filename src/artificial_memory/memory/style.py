from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Conversation


@dataclass
class ConversationStyle:
    """Represents the style characteristics of a conversation."""
    tone: str = "neutral"  # casual, formal, excited, frustrated, etc.
    formality: float = 0.5  # 0.0 = very casual, 1.0 = very formal
    verbosity: float = 0.5  # 0.0 = concise, 1.0 = verbose
    fillers: list[str] = field(default_factory=list)
    hesitation_markers: list[str] = field(default_factory=list)
    sentence_patterns: dict[str, int] = field(default_factory=dict)
    emotional_markers: dict[str, int] = field(default_factory=dict)
    typical_greetings: list[str] = field(default_factory=list)
    typical_closings: list[str] = field(default_factory=list)
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tone": self.tone,
            "formality": self.formality,
            "verbosity": self.verbosity,
            "fillers": self.fillers,
            "hesitation_markers": self.hesitation_markers,
            "sentence_patterns": self.sentence_patterns,
            "emotional_markers": self.emotional_markers,
            "typical_greetings": self.typical_greetings,
            "typical_closings": self.typical_closings,
            "avg_sentence_length": self.avg_sentence_length,
            "vocabulary_richness": self.vocabulary_richness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationStyle:
        return cls(**data)


@dataclass
class StyleProfile:
    """Aggregated style profile for a user/topic."""
    topic_id: int
    user_style: ConversationStyle = field(default_factory=ConversationStyle)
    assistant_style: ConversationStyle = field(default_factory=ConversationStyle)
    combined_style: ConversationStyle = field(default_factory=ConversationStyle)
    sample_count: int = 0
    updated_at: datetime = field(default_factory=datetime.now)

    def to_metadata(self) -> dict:
        return {
            "user_style": self.user_style.to_dict(),
            "assistant_style": self.assistant_style.to_dict(),
            "combined_style": self.combined_style.to_dict(),
            "sample_count": self.sample_count,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_metadata(cls, topic_id: int, metadata: dict) -> StyleProfile:
        profile = cls(topic_id=topic_id)
        if "user_style" in metadata:
            profile.user_style = ConversationStyle.from_dict(metadata["user_style"])
        if "assistant_style" in metadata:
            profile.assistant_style = ConversationStyle.from_dict(metadata["assistant_style"])
        if "combined_style" in metadata:
            profile.combined_style = ConversationStyle.from_dict(metadata["combined_style"])
        profile.sample_count = metadata.get("sample_count", 0)
        if "updated_at" in metadata:
            profile.updated_at = datetime.fromisoformat(metadata["updated_at"])
        return profile


class StyleExtractor:
    """Extracts style features from conversation text."""

    # Filler word patterns
    FILLER_PATTERNS = [
        r'\b(いや[〜～]?|えーっと|えーと|あの[〜～]?|うーん|まぁ|なんか|ちょっと|一応|一応|一応)\b',
        r'\b(だと思う|気がする|と思う|でしょうか|ですよね|じゃないですか)\b',
        r'\b(普通に|結構|まあ|わりと|そこそこ)\b',
        r'(w{2,}|笑|草|www)',
        r'\b(um|uh|er|ah|hmm|well|like|you know|basically|actually|literally)\b',
    ]

    # Hesitation markers
    HESITATION_PATTERNS = [
        r'\b(um|uh|er|ah|hmm|well|let me think|let me see)\b',
        r'\b(いや|えー|あの|うーん|まぁ)\b',
        r'(\.\.\.|…|。。)',
    ]

    # Emotional markers
    EMOTIONAL_PATTERNS = {
        "excited": [r'[!！]{2,}', r'わーい|やったー|すごい|amazing|awesome|great'],
        "frustrated": [r'[?？]{2,}', r'なぜ|どうして|why|困った|むずかしい|difficult|impossible'],
        "uncertain": [r'かも|かもしれない|maybe|perhaps|probably|not sure|分からない|unknown'],
        "confident": [r'確実|certain|definitely|absolutely|間違いなく|sure'],
        "polite": [r'です|ます|please|thank you|ありがとうございます|失礼'],
        "casual": [r'だね|だよ|だわ|じゃん|だよね|yeah|ok|okay|sure|cool|nice'],
    }

    # Greeting patterns
    GREETING_PATTERNS = [
        r'^(こんにちは|こんばんは|おはよう|hello|hi|hey|やあ|よう)',
        r'(よろしく|please|thanks|thanks!)',
    ]

    # Closing patterns
    CLOSING_PATTERNS = [
        r'(では|それじゃ|また|bye|goodbye|see you|またね|じゃあね)',
        r'(ありがとうございました|thank you|thanks|お疲れ様)',
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.filler_regex = [re.compile(p, re.IGNORECASE) for p in self.FILLER_PATTERNS]
        self.hesitation_regex = [re.compile(p, re.IGNORECASE) for p in self.HESITATION_PATTERNS]
        self.emotional_regex = {
            emotion: [re.compile(p, re.IGNORECASE) for p in patterns]
            for emotion, patterns in self.EMOTIONAL_PATTERNS.items()
        }
        self.greeting_regex = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self.closing_regex = [re.compile(p, re.IGNORECASE) for p in self.CLOSING_PATTERNS]

    def extract_style(self, text: str, role: str = "user") -> ConversationStyle:
        """Extract style features from text."""
        style = ConversationStyle()

        # Extract fillers
        fillers = []
        for regex in self.filler_regex:
            matches = regex.findall(text)
            fillers.extend(matches)
        style.fillers = list(set(fillers))[:20]  # Limit

        # Extract hesitation markers
        hesitations = []
        for regex in self.hesitation_regex:
            matches = regex.findall(text)
            hesitations.extend(matches)
        style.hesitation_markers = list(set(hesitations))[:20]

        # Detect emotional tone
        emotion_scores = {}
        for emotion, patterns in self.emotional_regex.items():
            score = sum(1 for pattern in patterns if pattern.search(text))
            emotion_scores[emotion] = score

        style.emotional_markers = emotion_scores

        # Determine primary tone
        if emotion_scores:
            style.tone = max(emotion_scores, key=emotion_scores.get)
        else:
            style.tone = "neutral"

        # Detect formality
        polite_count = sum(1 for p in self.emotional_regex["polite"] if p.search(text))
        casual_count = sum(1 for p in self.emotional_regex["casual"] if p.search(text))
        total = polite_count + casual_count
        if total > 0:
            style.formality = polite_count / total
        else:
            style.formality = 0.5

        # Extract greetings and closings
        for regex in self.greeting_regex:
            matches = regex.findall(text)
            style.typical_greetings.extend(matches)

        for regex in self.closing_regex:
            matches = regex.findall(text)
            style.typical_closings.extend(matches)

        style.typical_greetings = list(set(style.typical_greetings))[:10]
        style.typical_closings = list(set(style.typical_closings))[:10]

        # Calculate verbosity (average sentence length)
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            style.avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)

        # Calculate vocabulary richness (unique words / total words)
        words = re.findall(r'\w+', text.lower())
        if words:
            style.vocabulary_richness = len(set(words)) / len(words)

        return style

    def merge_styles(self, style_a: ConversationStyle, style_b: ConversationStyle,
                     weight_a: float = 0.5) -> ConversationStyle:
        """Merge two styles with given weights."""
        weight_b = 1.0 - weight_a

        merged = ConversationStyle()

        # Merge tone (take dominant)
        if style_a.emotional_markers and style_b.emotional_markers:
            all_emotions = set(style_a.emotional_markers.keys()) | set(style_b.emotional_markers.keys())
            merged_scores = {}
            for emotion in all_emotions:
                merged_scores[emotion] = (
                    style_a.emotional_markers.get(emotion, 0) * weight_a +
                    style_b.emotional_markers.get(emotion, 0) * weight_b
                )
            merged.emotional_markers = merged_scores
            if merged_scores:
                merged.tone = max(merged_scores, key=merged_scores.get)

        # Merge fillers
        merged.fillers = list(set(style_a.fillers + style_b.fillers))[:20]
        merged.hesitation_markers = list(set(style_a.hesitation_markers + style_b.hesitation_markers))[:20]

        # Merge emotional markers
        for emotion in set(style_a.emotional_markers.keys()) | set(style_b.emotional_markers.keys()):
            merged.emotional_markers[emotion] = (
                style_a.emotional_markers.get(emotion, 0) * weight_a +
                style_b.emotional_markers.get(emotion, 0) * weight_b
            )

        # Merge greetings/closings
        merged.typical_greetings = list(set(style_a.typical_greetings + style_b.typical_greetings))[:10]
        merged.typical_closings = list(set(style_a.typical_closings + style_b.typical_closings))[:10]

        # Weighted averages for numeric fields
        merged.formality = style_a.formality * weight_a + style_b.formality * weight_b
        merged.verbosity = style_a.verbosity * weight_a + style_b.verbosity * weight_b
        merged.avg_sentence_length = style_a.avg_sentence_length * weight_a + style_b.avg_sentence_length * weight_b
        merged.vocabulary_richness = style_a.vocabulary_richness * weight_a + style_b.vocabulary_richness * weight_b

        return merged


class StyleEngine:
    """Engine for managing conversation styles."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.extractor = StyleExtractor()

    def analyze_conversation(self, conversation: Conversation) -> tuple[ConversationStyle, ConversationStyle]:
        """Analyze a conversation and extract user/assistant styles."""
        messages = self.store.get_messages(conversation.id)

        user_texts = []
        assistant_texts = []

        for msg in messages:
            if msg.role.value == "user":
                user_texts.append(msg.content)
            elif msg.role.value == "assistant":
                assistant_texts.append(msg.content)

        user_style = ConversationStyle()
        assistant_style = ConversationStyle()

        if user_texts:
            combined = " ".join(user_texts)
            user_style = self.extractor.extract_style(combined, "user")

        if assistant_texts:
            combined = " ".join(assistant_texts)
            assistant_style = self.extractor.extract_style(combined, "assistant")

        return user_style, assistant_style

    def build_topic_profile(self, topic_id: int) -> dict:
        """Build style profile for a topic."""
        conversations = self.store.list_conversations(topic_id=topic_id)

        all_user_styles = []
        all_assistant_styles = []

        for conv in conversations:
            if conv.status.value == "completed":
                user_style, assistant_style = self.analyze_conversation(conv)
                all_user_styles.append(user_style)
                all_assistant_styles.append(assistant_style)

        # Aggregate styles
        def aggregate_styles(styles: list[ConversationStyle]) -> ConversationStyle:
            if not styles:
                return ConversationStyle()

            merged = styles[0]
            for style in styles[1:]:
                merged = self.extractor.merge_styles(merged, style, 0.5)
            return merged

        user_profile = aggregate_styles(all_user_styles)
        assistant_profile = aggregate_styles(all_assistant_styles)

        # Combined profile
        combined = self.extractor.merge_styles(user_profile, assistant_profile, 0.5)

        return {
            "user_style": user_profile.to_dict(),
            "assistant_style": assistant_profile.to_dict(),
            "combined_style": combined.to_dict(),
            "sample_count": len(conversations),
            "updated_at": datetime.now().isoformat(),
        }

    def reconstruct_style(self, base_text: str, target_style: ConversationStyle) -> str:
        """Reconstruct text with target style characteristics."""
        # This is a simplified reconstruction - in practice would use more sophisticated NLP
        result = base_text

        # Add fillers if style has them
        if target_style.fillers:
            # Insert a filler at sentence boundaries occasionally
            sentences = re.split(r'([。！？.!?])', result)
            result_parts = []
            for i, part in enumerate(sentences):
                result_parts.append(part)
                if part in '。！？.!?' and target_style.fillers:
                    # 20% chance to add filler
                    import random
                    if random.random() < 0.2:
                        filler = random.choice(target_style.fillers)
                        result_parts.append(f" {filler}")
            result = "".join(result_parts)

        # Adjust formality
        # This is simplified - real implementation would be more sophisticated

        return result


def create_style_engine(store: MemoryStore) -> StyleEngine:
    """Factory function to create style engine."""
    return StyleEngine(store)
