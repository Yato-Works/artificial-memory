from __future__ import annotations

import re
from collections import Counter

from artificial_memory.core.interfaces import MemoryStore, TopicClassifier
from artificial_memory.core.models import Project, Topic

# Keywords for common topic categories
TOPIC_KEYWORDS = {
    "architecture": ["architecture", "architectural", "design pattern", "system design", "microservice", "monolith", "scalability"],
    "database": ["database", "sql", "nosql", "postgres", "mysql", "sqlite", "migration", "schema", "query", "index"],
    "api": ["api", "rest", "graphql", "grpc", "endpoint", "request", "response", "swagger", "openapi"],
    "frontend": ["frontend", "react", "vue", "svelte", "component", "ui", "ux", "css", "html", "typescript"],
    "backend": ["backend", "server", "service", "worker", "queue", "async", "concurrency", "thread"],
    "testing": ["test", "testing", "unit test", "integration test", "e2e", "mock", "pytest", "jest", "coverage"],
    "deployment": ["deploy", "deployment", "ci/cd", "pipeline", "docker", "kubernetes", "k8s", "helm", "terraform"],
    "security": ["security", "auth", "authentication", "authorization", "oauth", "jwt", "encryption", "vulnerability"],
    "performance": ["performance", "optimization", "latency", "throughput", "bottleneck", "profiling", "benchmark"],
    "ai/ml": ["machine learning", "ml", "ai", "model", "training", "inference", "llm", "embedding", "vector", "rag"],
    "project-management": ["project", "task", "sprint", "backlog", "kanban", "jira", "roadmap", "milestone"],
    "code-review": ["review", "pr", "pull request", "merge", "diff", "comment", "approval", "change"],
    "debugging": ["debug", "bug", "error", "exception", "stack trace", "log", "issue", "fix", "crash"],
    "refactoring": ["refactor", "cleanup", "technical debt", "legacy", "modernize", "rewrite", "improve"],
    "documentation": ["document", "readme", "docstring", "comment", "wiki", "specification", "adr"],
}


class RuleBasedTopicClassifier:
    """Rule-based topic classifier for MVP."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self._keyword_weights = self._build_keyword_weights()

    def _build_keyword_weights(self) -> dict[str, dict[str, float]]:
        """Build weighted keyword dictionary for each category."""
        weights = {}
        for category, keywords in TOPIC_KEYWORDS.items():
            weights[category] = {}
            for i, kw in enumerate(keywords):
                # Earlier keywords get higher weight
                weights[category][kw.lower()] = 1.0 - (i * 0.05)
        return weights

    def classify(self, text: str, existing_topics: list[Topic]) -> tuple[Topic | None, float]:
        """Classify text into existing topics."""
        if not existing_topics:
            return None, 0.0

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        best_topic = None
        best_score = 0.0

        for topic in existing_topics:
            score = self._score_topic(text_lower, words, topic)
            if score > best_score:
                best_score = score
                best_topic = topic

        # Threshold for classification
        if best_score >= 0.3:
            return best_topic, min(best_score, 1.0)

        return None, 0.0

    def _score_topic(self, text: str, words: set[str], topic: Topic) -> float:
        """Score how well text matches a topic."""
        score = 0.0
        topic_name_lower = topic.name.lower()
        topic_path_lower = topic.path.lower()

        # Direct name/path match
        if topic_name_lower in text:
            score += 0.5
        if topic_path_lower in text:
            score += 0.3

        # Keyword matching based on topic path
        category = self._infer_category_from_path(topic.path)
        if category and category in self._keyword_weights:
            for kw, weight in self._keyword_weights[category].items():
                if kw in words:
                    score += weight * 0.1

        # Check for partial word matches in topic name
        topic_words = set(re.findall(r'\b\w+\b', topic_name_lower))
        overlap = words & topic_words
        if overlap:
            score += len(overlap) * 0.15

        return min(score, 1.0)

    def _infer_category_from_path(self, path: str) -> str | None:
        """Infer category from topic path."""
        path_lower = path.lower()
        for category in TOPIC_KEYWORDS:
            if category.replace("-", " ") in path_lower or category in path_lower:
                return category
        return None

    def suggest_new_topic(self, text: str, project_name: str) -> Topic:
        """Suggest a new topic based on text content."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        best_category = None
        best_score = 0.0

        for category, keywords in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_category = category

        if not best_category:
            best_category = "general"

        # Generate topic name from top keywords
        topic_words = []
        if best_category in self._keyword_weights:
            for kw in sorted(self._keyword_weights[best_category].keys(),
                           key=lambda k: -self._keyword_weights[best_category][k])[:3]:
                if kw in words:
                    topic_words.append(kw)

        if not topic_words:
            topic_words = [best_category]

        topic_name = "-".join(topic_words[:2]).title()
        path = f"Projects/{project_name}/{topic_name}"

        project = self.store.get_project_by_name(project_name)
        if not project:
            project = Project(name=project_name, display_name=project_name)
            project = self.store.create_project(project)

        return Topic(
            project_id=project.id,
            name=topic_name,
            path=path,
            description=f"Auto-created from conversation about {best_category}",
        )

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[tuple[str, float]]:
        """Extract keywords from text with scores."""
        text_lower = text.lower()
        words = re.findall(r'\b\w{3,}\b', text_lower)

        # Filter stop words
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "has", "had", "was", "were", "been", "have", "has", "this", "that", "with", "from", "they", "will", "would", "could", "should", "what", "when", "where", "which", "who", "how", "why", "your", "our", "their", "there", "here", "then", "than", "into", "onto", "upon"}
        filtered = [w for w in words if w not in stop_words and len(w) > 2]

        counter = Counter(filtered)
        total = len(filtered) or 1

        return [(word, count / total) for word, count in counter.most_common(max_keywords)]


def create_topic_classifier(store: MemoryStore) -> TopicClassifier:
    """Factory function to create topic classifier."""
    return RuleBasedTopicClassifier(store)
