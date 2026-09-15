from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import httpx

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.metrics.collector import get_metrics_collector


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ) -> AsyncGenerator[str, None] | str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        think: bool | None = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        # For thinking models (qwen3, deepseek-r1, ...) Ollama puts the chain
        # of thought in a separate field and leaves ``content`` empty unless
        # thinking is explicitly disabled. None = don't send the flag.
        self.think = think
        self.client = httpx.AsyncClient(timeout=120.0)

    def _payload(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload


    async def _generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation."""
        payload = self._payload(messages, temperature, max_tokens)
        payload["stream"] = True

        async with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue

    async def _generate_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Non-streaming generation."""
        payload = self._payload(messages, temperature, max_tokens)
        payload["stream"] = False

        response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ) -> AsyncGenerator[str, None] | str:
        if stream:
            raise ValueError("Use generate_stream() for streaming")
        return await self._generate_sync(messages, temperature, max_tokens)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._generate_stream(messages, temperature, max_tokens):
            yield chunk

    async def count_tokens(self, text: str) -> int:
        # Approximate: 1 token ≈ 4 characters for English
        return len(text) // 4


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )

    async def _generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    async def _generate_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Non-streaming generation."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ) -> AsyncGenerator[str, None] | str:
        if stream:
            raise ValueError("Use generate_stream() for streaming")
        return await self._generate_sync(messages, temperature, max_tokens)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._generate_stream(messages, temperature, max_tokens):
            yield chunk

    async def count_tokens(self, text: str) -> int:
        # Use tiktoken if available, otherwise approximate
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4


class LLMManager:
    """Manager for LLM providers with Context Runtime integration."""

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: RecallEngine,
        context_builder,
        default_provider: str = "ollama",
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.context_builder = context_builder
        self.providers: dict[str, LLMProvider] = {}
        self.default_provider = default_provider
        self.collector = get_metrics_collector()

    def add_provider(self, name: str, provider: LLMProvider):
        """Add an LLM provider."""
        self.providers[name] = provider

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """Get a provider by name, or default."""
        name = name or self.default_provider
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self.providers.keys())}")
        return self.providers[name]

    async def chat(
        self,
        user_message: str,
        topic_path: str | None = None,
        conversation_id: int | None = None,
        provider_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
        use_context: bool = True,
        recall_level: int = 2,
    ) -> dict[str, Any]:
        """Send a message and get a response with full context."""
        provider = self.get_provider(provider_name)

        # Get context if enabled
        context = ""
        topic_id = None
        conversation = None

        if use_context:
            if topic_path:
                # Resolve topic
                from artificial_memory.storage.conversation_logger import ConversationManager
                conv_manager = ConversationManager(self.store)
                topic = conv_manager.get_or_create_topic(topic_path)
                topic_id = topic.id

            if conversation_id:
                conversation = self.store.get_conversation(conversation_id)
                if conversation:
                    topic_id = conversation.topic_id

            # Build context
            if topic_id:
                current_memories = self.store.get_memories(
                    topic_id=topic_id,
                    is_current=True,
                    status="active"
                )
                context = self.context_builder.build_context(
                    user_message, topic_id, max_tokens=4000, current_memories=current_memories
                )

        # Build messages
        messages = []

        # System prompt with context
        if use_context and context:
            system_prompt = self.context_builder.build_system_prompt(
                topic_path or "General"
            ) + "\n\nRelevant Context:\n" + context
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant with access to a cognitive memory system."
            })

        # Add conversation history if available
        if conversation:
            messages_hist = self.store.get_messages(conversation.id, limit=20)
            for msg in messages_hist[-10:]:  # Last 10 messages
                messages.append({"role": msg.role.value, "content": msg.content})

        # Add user message
        messages.append({"role": "user", "content": user_message})

        # Generate response
        datetime.now()
        response_text = ""

        if stream:
            # Streaming response - return async generator
            async def stream_response() -> AsyncGenerator[str, None]:
                async for chunk in self.get_provider().generate_stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    yield chunk
            return {"stream": stream_response()}
        else:
            response_text = await self.get_provider().generate(
                messages, temperature=temperature, max_tokens=max_tokens, stream=False
            )

        # Log token usage
        provider = self.get_provider()
        input_tokens = sum(provider.count_tokens(m["content"]) for m in messages)
        output_tokens = provider.count_tokens(response_text)

        from artificial_memory.core.models import TokenUsage
        usage = TokenUsage(
            conversation_id=conversation_id,
            operation="chat",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=getattr(provider, "model", "unknown"),
        )
        self.store.log_token_usage(usage)

        # Log to metrics
        self.collector.record_context(type("ContextMetrics", (), {
            "raw_tokens": input_tokens,
            "effective_tokens": input_tokens + output_tokens,
            "parts_count": len(messages),
            "timestamp": datetime.now(),
        })())

        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": getattr(provider, "model", "unknown"),
            "context_used": use_context,
            "topic_id": topic_id,
        }

    async def chat_stream(
        self,
        user_message: str,
        topic_path: str | None = None,
        conversation_id: int | None = None,
        provider_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        use_context: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        provider = self.get_provider(provider_name)

        # Build context and messages (same as chat but streaming)
        context = ""
        topic_id = None
        conversation = None

        if use_context:
            if topic_path:
                from artificial_memory.storage.conversation_logger import ConversationManager
                conv_manager = ConversationManager(self.store)
                topic = conv_manager.get_or_create_topic(topic_path)
                topic_id = topic.id

            if conversation_id:
                conversation = self.store.get_conversation(conversation_id)
                if conversation:
                    topic_id = conversation.topic_id

            if topic_id:
                current_memories = self.store.get_memories(
                    topic_id=topic_id, is_current=True, status="active"
                )
                context = self.context_builder.build_context(
                    user_message, topic_id, max_tokens=4000, current_memories=current_memories
                )

        messages = []
        if use_context and context:
            system_prompt = self.context_builder.build_system_prompt(
                topic_path or "General"
            ) + "\n\nRelevant Context:\n" + context
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful AI assistant with access to a cognitive memory system."})

        if conversation:
            messages_hist = self.store.get_messages(conversation.id, limit=20)
            for msg in messages_hist[-10:]:
                messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        async for chunk in provider.generate(messages, stream=True):
            yield chunk


def create_llm_manager(
    store: MemoryStore,
    recall_engine: RecallEngine,
    context_builder,
    default_provider: str = "ollama",
) -> LLMManager:
    """Factory function to create LLM manager."""
    return LLMManager(store, recall_engine, context_builder, default_provider)
