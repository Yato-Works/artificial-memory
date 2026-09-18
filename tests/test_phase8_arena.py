"""Tests for Phase 8.1 Arena Core."""


import pytest

from artificial_memory.research.benchmarks.arena import (
    ArenaCategory,
    ArenaQuestion,
    build_dataset,
)
from artificial_memory.research.benchmarks.llm import LLMAnswer
from artificial_memory.research.benchmarks.player import (
    Answer,
    ControlledPlayer,
    CostReport,
    RealSystemPlayer,
)
from artificial_memory.research.benchmarks.players import (
    AMv020Player,
    Mem0StylePlayer,
    MemGPTStylePlayer,
    MemoryBankStylePlayer,
    SimpleRAGPlayer,
    create_controlled_players,
)
from artificial_memory.research.benchmarks.runner import (
    ANSWER_REPEATS,
    CONTROLLED_BUDGET,
    LATENCY_REPEATS,
    ArenaRunner,
    RunConfig,
    create_arena_manifest,
)


class MockPlayer(ControlledPlayer):
    """Minimal controlled player for testing."""

    def __init__(self, name: str = "Mock", config: dict | None = None):
        super().__init__(name, config)
        self.ingested = False

    def ingest(self, scenarios) -> None:
        self.ingested = True
        self._update_costs(self.costs.add_write_cost(calls=1, prompt_tokens=100, completion_tokens=50))

    def _answer_with_budget(self, question: ArenaQuestion, budget: int) -> Answer:
        return Answer(
            text=f"Mock answer for {question.question_id}",
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=min(100, budget),
            retrieval_latency_ms=10.0,
            context_build_latency_ms=5.0,
            answer_latency_ms=50.0,
        )


class MockRealPlayer(RealSystemPlayer):
    """Minimal real-system player for testing."""

    def __init__(self, name: str = "MockReal", config: dict | None = None):
        super().__init__(name, config)
        self.ingested = False

    def ingest(self, scenarios) -> None:
        self.ingested = True

    def _answer_native(self, question: ArenaQuestion) -> Answer:
        return Answer(
            text=f"Real answer for {question.question_id}",
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=1500,  # Native behavior, no budget
            retrieval_latency_ms=20.0,
            context_build_latency_ms=10.0,
            answer_latency_ms=100.0,
        )


# ==================== Arena Dataset Tests ====================

class TestArenaDataset:
    def test_build_dataset_deterministic(self):
        ds1 = build_dataset()
        ds2 = build_dataset()
        assert ds1.content_hash() == ds2.content_hash()
        assert ds1.question_count == 200
        assert len(ds1.scenarios) > 0

    def test_dataset_categories(self):
        ds = build_dataset()
        counts = ds.category_counts()
        for cat in ArenaCategory:
            assert counts[cat.value] == 20

    def test_question_structure(self):
        ds = build_dataset()
        q = ds.questions[0]
        assert q.question_id.startswith(q.category.value)
        assert q.ground_truth is not None
        assert hasattr(q, 'forbidden')
        assert hasattr(q, 'expected_abstention')

    def test_freeze_verification(self):
        ds = build_dataset()
        # Just verify the hash computation works consistently
        assert ds.content_hash() == build_dataset().content_hash()
        # Verify the content hash matches expected format
        assert len(ds.content_hash()) == 64  # SHA256 hex


# ==================== CostReport Tests ====================

class TestCostReport:
    def test_empty_report(self):
        report = CostReport.empty("TestPlayer", 2000)
        assert report.player_name == "TestPlayer"
        assert report.context_budget_tokens == 2000
        assert report.total_tokens == 0

    def test_add_write_cost(self):
        report = CostReport.empty("TestPlayer")
        report = report.add_write_cost(calls=2, prompt_tokens=500, completion_tokens=200, latency_ms=100)
        assert report.write_llm_calls == 2
        assert report.write_llm_prompt_tokens == 500
        assert report.write_llm_completion_tokens == 200
        assert report.write_latency_ms == 100
        assert report.total_llm_calls == 2
        assert report.total_prompt_tokens == 500
        assert report.total_completion_tokens == 200
        assert report.total_tokens == 700

    def test_non_llm_write_records_no_llm_calls(self):
        """Embedding-only / deterministic writes must not inflate write_llm_calls."""
        report = CostReport.empty("TestPlayer")
        report = report.add_write_cost(calls=0, latency_ms=12.5)
        assert report.write_llm_calls == 0
        assert report.write_llm_prompt_tokens == 0
        assert report.write_llm_completion_tokens == 0
        assert report.write_latency_ms == 12.5
        assert report.total_llm_calls == 0

    def test_add_read_cost(self):
        report = CostReport.empty("TestPlayer")
        report = report.add_write_cost(calls=1, prompt_tokens=100, completion_tokens=50)
        report = report.add_read_cost(
            retrieval_calls=1, retrieval_latency_ms=10,
            context_calls=1, context_latency_ms=5,
            answer_calls=1, answer_prompt_tokens=200, answer_completion_tokens=100, answer_latency_ms=50,
            context_tokens_used=500
        )
        assert report.retrieval_calls == 1
        assert report.retrieval_latency_ms == 10
        assert report.context_construction_calls == 1
        assert report.answer_llm_calls == 1
        assert report.answer_llm_prompt_tokens == 200
        assert report.context_tokens_used == 500
        assert report.budget_violations == 0

    def test_budget_violation_tracking(self):
        report = CostReport.empty("TestPlayer", 2000)
        report = report.add_read_cost(context_tokens_used=2500)
        assert report.budget_violations == 1
        assert report.context_tokens_used == 2500


# ==================== Player Protocol Tests ====================

class TestPlayerProtocol:
    def test_controlled_player_budget_enforcement(self):
        player = MockPlayer()
        assert isinstance(player, ControlledPlayer)
        assert player.config.get('context_budget', CONTROLLED_BUDGET) == CONTROLLED_BUDGET

    def test_real_player_no_budget_enforcement(self):
        player = MockRealPlayer()
        assert isinstance(player, RealSystemPlayer)

    def test_cost_tracking_reset(self):
        player = MockPlayer()
        player.ingest([])
        assert player.costs.write_llm_calls == 1
        player.reset_costs()
        assert player.costs.write_llm_calls == 0


# ==================== Controlled Players Tests ====================

class TestControlledPlayers:
    @pytest.fixture
    def base_config(self):
        return {"context_budget": CONTROLLED_BUDGET, "llm": "test-model"}

    def test_create_all_players(self, base_config):
        players = create_controlled_players(base_config)
        assert len(players) == 5
        names = [p.name for p in players]
        assert "Simple RAG" in names
        assert "MemoryBank-style" in names
        assert "MemGPT-style" in names
        assert "Mem0-style" in names
        assert "AM v0.2.0" in names

    def test_simple_rag_ingest(self, base_config):
        player = SimpleRAGPlayer(base_config)
        from artificial_memory.research.benchmarks.arena import build_dataset
        ds = build_dataset()
        player.ingest(list(ds.scenarios))
        assert player._store is not None

    def test_memorybank_ingest(self, base_config):
        player = MemoryBankStylePlayer(base_config)
        from artificial_memory.research.benchmarks.arena import build_dataset
        ds = build_dataset()
        player.ingest(list(ds.scenarios))
        assert player._store is not None

    def test_memgpt_ingest(self, base_config):
        player = MemGPTStylePlayer(base_config)
        from artificial_memory.research.benchmarks.arena import build_dataset
        ds = build_dataset()
        player.ingest(list(ds.scenarios))
        assert player._store is not None
        assert len(player._main_context) > 0

    def test_mem0_ingest(self, base_config):
        player = Mem0StylePlayer(base_config)
        from artificial_memory.research.benchmarks.arena import build_dataset
        ds = build_dataset()
        player.ingest(list(ds.scenarios))
        assert player._store is not None

    def test_am_ingest(self, base_config):
        player = AMv020Player(base_config)
        from artificial_memory.research.benchmarks.arena import build_dataset
        ds = build_dataset()
        player.ingest(list(ds.scenarios))
        assert player._runtime is not None

    def test_non_llm_players_report_zero_write_llm_calls(self, base_config):
        """Write-side metric truthfulness (Benchmark_Plan.txt §6).

        Players that never call an LLM on the write path must report
        write_llm_calls == 0 even though write latency is still recorded.
        """
        scenario = build_dataset().scenarios[0]
        for player_cls in (
            SimpleRAGPlayer,
            MemoryBankStylePlayer,
            MemGPTStylePlayer,
            Mem0StylePlayer,
            AMv020Player,
        ):
            player = player_cls(base_config)
            player.ingest([scenario])
            assert player.costs.write_llm_calls == 0, player.name
            assert player.costs.write_llm_prompt_tokens == 0, player.name
            assert player.costs.write_llm_completion_tokens == 0, player.name
            assert player.costs.write_latency_ms >= 0.0, player.name

    def test_player_index_dirs_are_isolated_and_cleaned(self, base_config):
        """Each ingest gets its own index dir; all are removed at process exit."""
        from artificial_memory.research.benchmarks import players as players_module

        first = players_module._isolated_index_dir("Simple RAG")
        second = players_module._isolated_index_dir("Simple RAG")
        assert first != second
        assert first.exists() and second.exists()

        players_module._cleanup_isolated_index_dirs()
        assert not first.exists()
        assert not second.exists()

    def test_vector_retrieval_reaches_target_fact(self, base_config):
        """Regression (Phase 8.3.4): baselines must retrieve via the vector index.

        The first E2E run answered everything with ``I don't know`` because the
        players queried ``BasicRecallEngine`` — which ignores the vector index
        entirely (insertion-order candidates + word overlap) — while the
        dataset's target fact ranked #1 (0.808 cosine) in the very index the
        players had built. A baseline must never be silently crippled by
        harness plumbing, so this guard asserts dense retrieval actually
        surfaces the planted fact for the easiest factual question.
        """
        ds = build_dataset()
        q = ds.by_id("factual_recall-00")
        assert q is not None
        for player_cls in (SimpleRAGPlayer, MemoryBankStylePlayer, MemGPTStylePlayer):
            player = player_cls(base_config)
            player.ingest(list(ds.scenarios))
            memories = player._vector_retrieve(q.question, k=5)
            texts = [m.content.lower() for m in memories]
            assert any("ripgrep" in t and "atlas" in t for t in texts), player.name

    def test_mem0_write_llm_cost_is_measured(self, base_config):
        """Mem0-style LLM extraction must report real write-side tokens/calls."""
        scenario = build_dataset().scenarios[0]

        class StubExtractionAnswerer:
            def __init__(self) -> None:
                self.extract_calls = 0

            def extract(self, exchange_text: str) -> LLMAnswer:
                self.extract_calls += 1
                return LLMAnswer(
                    text="The user uses ripgrep.",
                    latency_ms=5.0,
                    prompt_tokens=40,
                    completion_tokens=8,
                    total_tokens=48,
                    messages=[],
                )

        answerer = StubExtractionAnswerer()
        player = Mem0StylePlayer(base_config, answerer=answerer)
        player.ingest([scenario])

        assert answerer.extract_calls == len(scenario.turns)
        assert player.costs.write_llm_calls == len(scenario.turns)
        assert player.costs.write_llm_prompt_tokens == 40 * len(scenario.turns)
        assert player.costs.write_llm_completion_tokens == 8 * len(scenario.turns)
        assert player.costs.total_completion_tokens == 8 * len(scenario.turns)


# ==================== Arena Runner Tests ====================

class TestArenaRunner:
    @pytest.fixture
    def dataset(self):
        return build_dataset()

    @pytest.fixture
    def controlled_players(self):
        return create_controlled_players({"context_budget": CONTROLLED_BUDGET})

    def test_run_config_creation(self, dataset, controlled_players):
        config = RunConfig(
            arena_class="controlled",
            players=controlled_players[:2],  # Test with 2 players
            dataset=dataset,
            answer_repeats=1,  # Fast test
            latency_repeats=2,
            output_dir="test_results",
        )
        assert config.arena_class == "controlled"
        assert len(config.players) == 2

    def test_manifest_creation(self, dataset, controlled_players):
        manifest = create_arena_manifest("controlled", controlled_players[:1], dataset)
        assert manifest.name == "memory-arena-controlled"
        assert manifest.experiment_type.value == "benchmark"
        # Players are in benchmark_parameters
        assert "players" in manifest.parameters["benchmark_parameters"]
        assert manifest.parameters["benchmark_parameters"]["context_budget"] == CONTROLLED_BUDGET
        assert manifest.parameters["benchmark_parameters"]["answer_repeats"] == ANSWER_REPEATS
        assert manifest.parameters["benchmark_parameters"]["latency_repeats"] == LATENCY_REPEATS

    def test_runner_executes(self, dataset, controlled_players, tmp_path):
        # Use only 1 player, 1 question for fast test
        player = controlled_players[0]
        config = RunConfig(
            arena_class="controlled",
            players=[player],
            dataset=dataset,
            answer_repeats=1,
            latency_repeats=1,
            output_dir=str(tmp_path),
        )
        runner = ArenaRunner(config)
        result = runner.run()

        assert result.run_id.startswith("controlled_")
        assert result.arena_class == "controlled"
        assert result.dataset_hash == dataset.content_hash()
        assert player.name in result.player_results
        pr = result.player_results[player.name]
        assert len(pr.question_results) == 200  # All questions answered
        assert pr.final_costs is not None

        # Check output files
        run_dir = tmp_path / "controlled" / result.run_id
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "latency.csv").exists()
        assert (run_dir / "tokens.csv").exists()
        assert (run_dir / "traces.jsonl").exists()
        assert (run_dir / "environment.txt").exists()
        assert (run_dir / "result.json").exists()

    def test_failure_recording(self, dataset, tmp_path):
        """Test that failures are recorded to failures.jsonl"""
        player = MockPlayer("FailingPlayer")
        # Make ingest fail
        def failing_ingest(scenarios):
            raise ValueError("Simulated failure")
        player.ingest = failing_ingest

        config = RunConfig(
            arena_class="controlled",
            players=[player],
            dataset=dataset,
            answer_repeats=1,
            latency_repeats=1,
            output_dir=str(tmp_path),
        )
        runner = ArenaRunner(config)
        result = runner.run()

        assert len(result.failures) == 1
        assert result.failures[0]["player"] == "FailingPlayer"
        assert result.failures[0]["stage"] == "ingest"
        assert "Simulated failure" in result.failures[0]["error"]

        # A player whose ingest failed must not be asked any questions:
        # answering from an empty store would record meaningless quality.
        assert "FailingPlayer" not in result.player_results

        # Check failures.jsonl exists and has content
        failures_path = tmp_path / "controlled" / result.run_id / "failures.jsonl"
        assert failures_path.exists()
        content = failures_path.read_text()
        assert "FailingPlayer" in content
        assert "ingest" in content

    def test_runner_does_not_double_count_answer_costs(self, dataset, tmp_path):
        """Regression: the runner must not re-record the read cost the player
        already recorded internally, or answer_llm_calls doubles
        (10 questions -> 20 calls) and §6 totals become lies.

        Note: ingest failures must skip the player (no questions asked).
        """
        class CostRecordingPlayer(MockPlayer):
            def _answer_with_budget(self, question, budget):
                ans = super()._answer_with_budget(question, budget)
                self._update_costs(self.costs.add_read_cost(
                    retrieval_calls=1, retrieval_latency_ms=1.0,
                    context_calls=1, context_latency_ms=1.0,
                    answer_calls=1, answer_prompt_tokens=10,
                    answer_completion_tokens=5, answer_latency_ms=1.0,
                    context_tokens_used=ans.context_tokens_used,
                ))
                return ans

        player = CostRecordingPlayer()
        config = RunConfig(
            arena_class="controlled",
            players=[player],
            dataset=dataset,
            answer_repeats=1,
            latency_repeats=1,
            output_dir=str(tmp_path),
            question_ids=[q.question_id for q in dataset.questions[:2]],
        )
        result = ArenaRunner(config).run()
        costs = result.player_results["Mock"].final_costs
        assert costs is not None
        assert costs.answer_llm_calls == 2  # one per question, not 2x
        assert costs.answer_llm_prompt_tokens == 20
        assert costs.answer_llm_completion_tokens == 10


# ==================== Repeat Protocol Tests ====================

class TestRepeatProtocol:
    def test_answer_repeats_count(self):
        """Verify answer repeats = 3 per §9."""
        assert ANSWER_REPEATS == 3

    def test_latency_repeats_count(self):
        """Verify latency repeats >= 5 per §9."""
        assert LATENCY_REPEATS >= 5

    def test_runner_respects_repeats(self, tmp_path):
        from artificial_memory.research.benchmarks.arena import build_dataset
        dataset = build_dataset()
        player = MockPlayer()
        config = RunConfig(
            arena_class="controlled",
            players=[player],
            dataset=dataset,
            answer_repeats=3,
            latency_repeats=5,
            output_dir=str(tmp_path),
        )
        runner = ArenaRunner(config)
        result = runner.run()

        pr = result.player_results[player.name]
        # Pick a question
        qid = list(pr.question_results.keys())[0]
        qr = pr.question_results[qid]

        # Should have 3 answer repeats
        assert len(qr.answers) == 3
        # Should have 5 latency samples (3 answer + 2 extra)
        assert len(qr.latency_samples) == 5

        # All answers should have run_index 0, 1, 2
        run_indices = sorted(a.run_index for a in qr.answers)
        assert run_indices == [0, 1, 2]


# ==================== Budget Enforcement Tests ====================

class TestBudgetEnforcement:
    def test_controlled_budget_constant(self):
        assert CONTROLLED_BUDGET == 2000

    def test_budget_violation_recorded(self, tmp_path):
        from artificial_memory.research.benchmarks.arena import build_dataset
        dataset = build_dataset()

        class OverBudgetPlayer(ControlledPlayer):
            def __init__(self):
                super().__init__("OverBudget", {"context_budget": 2000})
            def ingest(self, scenarios): pass
            def _answer_with_budget(self, question, budget):
                return Answer(
                    text="x" * 3000,  # 3000 tokens > 2000 budget
                    question_id=question.question_id,
                    player_name=self.name,
                    run_index=0,
                    context_tokens_used=3000,
                )

        player = OverBudgetPlayer()
        config = RunConfig(
            arena_class="controlled",
            players=[player],
            dataset=dataset,
            answer_repeats=1,
            latency_repeats=1,
            output_dir=str(tmp_path),
        )
        runner = ArenaRunner(config)
        result = runner.run()

        pr = result.player_results[player.name]
        # Enforcement is independent of the player's own cost recording: the
        # violation must be flagged, tokens accounting stays the player's duty.
        assert pr.final_costs.budget_violations >= 1


# ==================== Real-System Arena Tests ====================

class TestRealSystemArena:
    def test_real_player_no_budget_enforcement(self, tmp_path):
        from artificial_memory.research.benchmarks.arena import build_dataset
        dataset = build_dataset()

        player = MockRealPlayer()
        config = RunConfig(
            arena_class="real",
            players=[player],
            dataset=dataset,
            answer_repeats=1,
            latency_repeats=1,
            output_dir=str(tmp_path),
        )
        runner = ArenaRunner(config)
        result = runner.run()

        assert result.arena_class == "real"
        pr = result.player_results[player.name]
        # Real players can use more than 2000 tokens
        qr = list(pr.question_results.values())[0]
        ans = qr.answers[0]
        assert ans.context_tokens_used == 1500  # Native behavior

    def test_real_manifest_context_budget_native(self):
        from artificial_memory.research.benchmarks.arena import build_dataset
        dataset = build_dataset()
        player = MockRealPlayer()
        manifest = create_arena_manifest("real", [player], dataset)
        # context_budget is in benchmark_parameters
        assert manifest.parameters["benchmark_parameters"]["context_budget"] == "native"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
