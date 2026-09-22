"""Fixed LLM Adapter for the Memory Arena (Phase 8.2).

Per Benchmark_Plan.txt §5 "Fixed LLM Configuration" (Spec Freeze):

- One thin shared layer; ALL five Controlled players answer through the same
  instance with the same model / generation settings, so that measured
  differences are memory / retrieval strategy differences, never LLM
  differences.
- Generation signature is ``answer(question_text, context)`` ONLY. Ground
  truth and other question metadata can never reach the prompt by design
  (Leakage Boundary).
- The frozen configuration (model, temperature, seed, prompts, max tokens)
  is hashed into ``config_sha256`` / ``prompt_sha256`` for the run manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ==================== Frozen configuration (Spec Freeze) ====================

FROZEN_BASE_URL = "http://localhost:11434"
# Frozen model: phi4-mini:latest (non-thinking, local).
# Decision evidence (probe, 2026-09-18): qwen3:4b on this Ollama install
# ignores both the `think: false` flag and the /no_think soft switch — it
# emits 300-500 reasoning tokens in `content` per call (~26 s), which would
# poison forbidden-term scoring and make 3,000+ calls impractical.
# phi4-mini answers directly ("ripgrep", 3 completion tokens). Recorded in
# Benchmark_Plan.txt §5 Fixed LLM Configuration.
FROZEN_MODEL = "phi4-mini:latest"
FROZEN_TEMPERATURE = 0.0
FROZEN_SEED = 42
FROZEN_ANSWER_MAX_TOKENS = 256
FROZEN_EXTRACTION_MAX_TOKENS = 256

# phi4-mini is a non-thinking model; the Ollama `think` flag is NOT sent
# (sending it to a non-thinking model is rejected by some Ollama versions).
FROZEN_THINK: bool | None = None

ABSTENTION_TEXT = "I don't know."

# Official-protocol abstention surface for the LoCoMo arena.
#
# The pinned official harness (third_party/benchmarks/locomo/task_eval/evaluation.py)
# credits a category-5 (adversarial) question ONLY when the prediction literally
# contains "no information available" or "not mentioned"; every other refusal
# wording ("I don't know.", "No", "None") scores 0 even though the memory system
# behaved correctly.  Measured offline on the directives_v4 run (1986 Q,
# scripts/ab_official_abstention.py): aligning the surface moves the official F1
# from 47.69% to 50.16% (+2.47pp) with zero change to AM's dev matcher.
#
# Keep ABSTENTION_TEXT itself untouched so already-published arenas (e.g.
# LongMemEval) keep their frozen prompt hash and stay comparable.
OFFICIAL_ABSTENTION_TEXT = "No information available (not mentioned in the conversation)."

OFFICIAL_ABSTENTION_MARKERS = ("no information available", "not mentioned")

FROZEN_ANSWER_SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided conversation context.\n"
    "Rules:\n"
    "- Reply with the final answer only. Never show your reasoning, thinking"
    " steps, or any preamble.\n"
    f"- If the context does not contain the answer, reply exactly: {ABSTENTION_TEXT}\n"
    "- Never invent information that is not in the context.\n"
    "- Answer in English with one or two short sentences.\n"
)

FROZEN_ANSWER_USER_TEMPLATE = (
    "Conversation context:\n"
    "{context}\n"
    "\n"
    "Question: {question_text}\n"
    "Answer:\n"
)

# Mem0-style single-pass ADD-only extraction prompt (write-side, frozen).
FROZEN_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable long-term memories from a conversation exchange.\n"
    "Rules:\n"
    "- Output one memory per line, as a short self-contained factual sentence.\n"
    "- Only state facts explicitly present in the conversation.\n"
    "- Do not merge, rephrase, rank, or annotate. One fact, one line.\n"
    "- If nothing is worth remembering, output nothing.\n"
    "- Reply with the extracted lines only. Never show your reasoning.\n"
)

FROZEN_EXTRACTION_USER_TEMPLATE = (
    "Conversation exchange:\n"
    "{exchange}\n"
    "\n"
    "Extracted memories:\n"
)


def sha256_of(text: str) -> str:
    """Stable hash for config / prompt freeze recording."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


FROZEN_PROMPT_HASHES = {
    "answer_system_prompt_sha256": sha256_of(FROZEN_ANSWER_SYSTEM_PROMPT),
    "answer_user_template_sha256": sha256_of(FROZEN_ANSWER_USER_TEMPLATE),
    "extraction_system_prompt_sha256": sha256_of(FROZEN_EXTRACTION_SYSTEM_PROMPT),
    "extraction_user_template_sha256": sha256_of(FROZEN_EXTRACTION_USER_TEMPLATE),
}


def frozen_config_sha256() -> str:
    """Hash of the whole frozen LLM configuration (for the manifest)."""
    canonical = json.dumps(
        {
            "base_url": FROZEN_BASE_URL,
            "model": FROZEN_MODEL,
            "temperature": FROZEN_TEMPERATURE,
            "seed": FROZEN_SEED,
            "answer_max_tokens": FROZEN_ANSWER_MAX_TOKENS,
            "extraction_max_tokens": FROZEN_EXTRACTION_MAX_TOKENS,
            "think": FROZEN_THINK,
            "prompts": FROZEN_PROMPT_HASHES,
        },
        sort_keys=True,
    )
    return sha256_of(canonical)


# ==================== Results ====================

@dataclass
class LLMAnswer:
    """Result of one frozen-config LLM call."""
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Full prompt payload, recorded for traces / leakage assertions.
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "messages": self.messages,
        }

# ==================== Adapter ====================

class OllamaAnswerer:
    """The single frozen LLM adapter shared by all Controlled players.

    Two call shapes, both with the frozen configuration:

    - ``answer(question_text, context)`` — read-side answer generation.
      The signature intentionally accepts only the question TEXT and the
      retrieved context; a question object (which carries ground truth)
      cannot be passed by accident.
    - ``extract(exchange_text)`` — write-side single-pass memory extraction
      (e.g. Mem0-style ADD-only ingestion). Token usage is reported back so
      players can record mandatory write-side cost.
    """

    def __init__(
        self,
        base_url: str = FROZEN_BASE_URL,
        model: str = FROZEN_MODEL,
        timeout_seconds: float = 120.0,
        answer_adapter: Any | None = None,
        num_ctx: int | None = None,
    ):
        # Spec Freeze: model / temperature / seed / max tokens are constants
        # above and are NOT constructor-overridable.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_seconds)
        # Optional benchmark-only post-processing of the raw model text.  Left at
        # ``None`` in production so the frozen protocol is bit-for-bit unchanged.
        self.answer_adapter = answer_adapter
        # A/B-only context window cap.  ``None`` (production) leaves Ollama's own
        # default untouched.  Needed because large-tag models (e.g. 7B on an 8 GB
        # RTX 3050) request a 32K KV cache by default and the server answers
        # HTTP 500 once VRAM is exhausted, which silently kills a full run.
        self.num_ctx = num_ctx

    # ---------- prompt construction (frozen) ----------

    def build_answer_messages(self, question_text: str, context: str) -> list[dict[str, str]]:
        user_content = FROZEN_ANSWER_USER_TEMPLATE.format(
            context=context, question_text=question_text
        )
        return [
            {"role": "system", "content": FROZEN_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def build_extraction_messages(self, exchange_text: str) -> list[dict[str, str]]:
        user_content = FROZEN_EXTRACTION_USER_TEMPLATE.format(exchange=exchange_text)
        return [
            {"role": "system", "content": FROZEN_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ---------- frozen generation call ----------

    def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> LLMAnswer:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": FROZEN_TEMPERATURE,
                "seed": FROZEN_SEED,
                "num_predict": max_tokens,
            },
        }
        if FROZEN_THINK is not None:
            payload["think"] = FROZEN_THINK
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        start = time.perf_counter()
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
        text = data.get("message", {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return LLMAnswer(
            text=text.strip(),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            messages=messages,
            raw_response=data,
        )

    # ---------- public API (Leakage Boundary by signature) ----------

    def answer(self, question_text: str, context: str) -> LLMAnswer:
        """Generate the final answer from retrieved context ONLY."""
        messages = self.build_answer_messages(question_text, context)
        return self._chat(messages, FROZEN_ANSWER_MAX_TOKENS)

    def extract(self, exchange_text: str) -> LLMAnswer:
        """Single-pass write-side memory extraction (frozen prompt)."""
        messages = self.build_extraction_messages(exchange_text)
        return self._chat(messages, FROZEN_EXTRACTION_MAX_TOKENS)

    def close(self) -> None:
        self._client.close()


__all__ = [
    "ABSTENTION_TEXT",
    "FROZEN_ANSWER_MAX_TOKENS",
    "FROZEN_ANSWER_SYSTEM_PROMPT",
    "FROZEN_ANSWER_USER_TEMPLATE",
    "FROZEN_BASE_URL",
    "FROZEN_EXTRACTION_MAX_TOKENS",
    "FROZEN_EXTRACTION_SYSTEM_PROMPT",
    "FROZEN_EXTRACTION_USER_TEMPLATE",
    "FROZEN_MODEL",
    "FROZEN_PROMPT_HASHES",
    "FROZEN_SEED",
    "FROZEN_TEMPERATURE",
    "FROZEN_THINK",
    "LLMAnswer",
    "OllamaAnswerer",
    "frozen_config_sha256",
    "sha256_of",
]
