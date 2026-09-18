"""Live smoke test: one real frozen-LLM answer through Ollama (phi4-mini:latest).

Run manually: python tests/smoke_llm_answer.py
Not collected by pytest (filename does not start with test_).
"""
import sys

from artificial_memory.research.benchmarks.arena import ArenaCategory, build_dataset
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, frozen_config_sha256
from artificial_memory.research.benchmarks.players import SimpleRAGPlayer

sys.stdout.reconfigure(encoding="utf-8")

print("config_sha256:", frozen_config_sha256()[:16], "...")

answerer = OllamaAnswerer()
llm = answerer.answer(
    "Which search tool did the user configure for project atlas?",
    "[user] I set up ripgrep as the search backend for the atlas project.\n"
    "[assistant] Got it, ripgrep it is.",
)
print("latency_ms:", round(llm.latency_ms, 1))
print("tokens:", llm.prompt_tokens, "prompt +", llm.completion_tokens, "completion")
print("answer:", llm.text)

# End-to-end: real player + real answerer on one question
ds = build_dataset()
q = ds.by_category(ArenaCategory.FACTUAL_RECALL)[0]
player = SimpleRAGPlayer({"context_budget": 2000}, answerer=answerer)
player.ingest([ds.scenarios[0], ds.scenarios[1]])
answer = player._answer_with_budget(q, 2000)
print("\nquestion:", q.question)
print("answer:", answer.text)
print("context_tokens:", answer.context_tokens_used, "| llm tokens:", llm.total_tokens)
print("write cost:", player.costs.write_llm_calls, "calls /", player.costs.write_llm_prompt_tokens, "pt /", player.costs.write_llm_completion_tokens, "ct")
answerer.close()
