"""Leakage Guard tests for the Memory Arena (Phase 8.2).

Per Benchmark_Plan.txt §5 "Leakage Boundary" (Spec Freeze):

- Ground truth / forbidden terms / abstention flags must NEVER reach the
  LLM prompt. They are evaluation-only.
- The generation API signature is answer(question_text, context) — a question
  object (which carries the answers) cannot be passed by design.
- These tests plant a sentinel into the ground truth of real dataset
  questions and assert the sentinel never appears in any recorded prompt.
"""

import inspect

import pytest

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    build_dataset,
)
from artificial_memory.research.benchmarks.llm import (
    ABSTENTION_TEXT,
    FROZEN_MODEL,
    FROZEN_SEED,
    FROZEN_TEMPERATURE,
    LLMAnswer,
    OllamaAnswerer,
    frozen_config_sha256,
)
from artificial_memory.research.benchmarks.players import (
    AMv020Player,
    Mem0StylePlayer,
    MemGPTStylePlayer,
    MemoryBankStylePlayer,
    SimpleRAGPlayer,
    create_controlled_players,
    get_shared_answerer,
)

SENTINEL = "GROUND_TRUTH_SENTINEL_9f83a"


class RecordingAnswerer:
    """Test double: records every prompt, returns a fixed answer.

    Mimics OllamaAnswerer's public surface (answer/extract) but never
    touches a real LLM.
    """

    def __init__(self) -> None:
        self.recorded: list[list[dict[str, str]]] = []
        self.extract_recorded: list[str] = []

    def answer(self, question_text: str, context: str) -> LLMAnswer:
        # Same frozen prompt construction as the real adapter, so the
        # recorded messages are what a real run would send.
        real = OllamaAnswerer()
        messages = real.build_answer_messages(question_text, context)
        real.close()
        self.recorded.append(messages)
        return LLMAnswer(
            text="Recorded stub answer",
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            messages=messages,
        )

    def extract(self, exchange_text: str) -> LLMAnswer:
        self.extract_recorded.append(exchange_text)
        return LLMAnswer(
            text="The user uses ripgrep for search.",
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def make_players() -> list:
    stub = RecordingAnswerer()
    players = [
        SimpleRAGPlayer({"context_budget": 2000}, answerer=stub),
        MemoryBankStylePlayer({"context_budget": 2000}, answerer=stub),
        MemGPTStylePlayer({"context_budget": 2000}, answerer=stub),
        Mem0StylePlayer({"context_budget": 2000}, answerer=stub),
    ]
    return players, stub


class TestGenerationAPIShape:
    def test_answer_signature_accepts_only_text_and_context(self):
        params = list(inspect.signature(OllamaAnswerer.answer).parameters)
        assert params == ["self", "question_text", "context"]

    def test_no_generation_api_accepts_question_object(self):
        # The old leakage temptation: _generate_answer(question, ...) — gone.
        for cls in (SimpleRAGPlayer, MemoryBankStylePlayer, MemGPTStylePlayer, Mem0StylePlayer, AMv020Player):
            assert not hasattr(cls, "_generate_mock_answer"), cls.__name__
            for name, value in vars(cls).items():
                if name.startswith("_generate") and callable(value):
                    params = list(inspect.signature(value).parameters)
                    assert "question" not in params, (cls.__name__, name, params)


class TestFrozenConfig:
    def test_frozen_values(self):
        assert FROZEN_MODEL == "phi4-mini:latest"
        assert FROZEN_TEMPERATURE == 0.0
        assert FROZEN_SEED == 42

    def test_config_hash_is_stable(self):
        assert frozen_config_sha256() == frozen_config_sha256()
        assert len(frozen_config_sha256()) == 64

    def test_create_players_share_one_answerer(self):
        players = create_controlled_players({"context_budget": 2000})
        answerers = {id(p._answerer) for p in players}
        assert len(answerers) == 1
        assert isinstance(players[0]._answerer, OllamaAnswerer)
        assert get_shared_answerer() is players[0]._answerer


class TestLeakageGuard:
    @pytest.fixture
    def seeded_dataset(self):
        dataset = build_dataset()
        questions = [
            q for q in dataset.questions
            if q.category is not ArenaCategory.ABSTENTION
        ][:2]
        for q in questions:
            object.__setattr__(q, "ground_truth", ((SENTINEL,),))
        return questions

    def test_controlled_players_never_leak_ground_truth(self, seeded_dataset):
        players, stub = make_players()
        for player in players:
            player.ingest([])  # minimal init
        for player in players:
            answer = player._answer_with_budget(seeded_dataset[0], 2000)
            assert answer is not None
        assert stub.recorded, "no prompts were recorded"
        for messages in stub.recorded:
            blob = "\n".join(m["content"] for m in messages)
            assert SENTINEL not in blob

    def test_extraction_never_receives_ground_truth(self):
        stub = RecordingAnswerer()
        mem0 = Mem0StylePlayer({"context_budget": 2000}, answerer=stub)
        ds = build_dataset()
        mem0.ingest([ds.scenarios[0]])
        assert stub.extract_recorded, "no extraction prompts recorded"
        for text in stub.extract_recorded:
            assert SENTINEL not in text

    def test_prompt_contains_question_and_context_only(self, seeded_dataset):
        stub = RecordingAnswerer()
        rag = SimpleRAGPlayer({"context_budget": 2000}, answerer=stub)
        rag.ingest([])
        rag._answer_with_budget(seeded_dataset[0], 2000)
        messages = stub.recorded[-1]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "ONLY the provided conversation context" in messages[0]["content"]
        user_blob = messages[1]["content"]
        assert "Question:" in user_blob
        assert "Conversation context:" in user_blob

    def test_abstention_text_is_part_of_frozen_prompt(self):
        answerer = OllamaAnswerer()
        messages = answerer.build_answer_messages("Q?", "context")
        answerer.close()
        assert ABSTENTION_TEXT in messages[0]["content"]

