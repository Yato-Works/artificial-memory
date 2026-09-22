"""Competitor Memory Systems Adapters (Phase 1).

Exposes Controlled Arena adapters for Mem0, MemGPT, MemoryBank, and Simple RAG.
All adapters are governed by their respective benchmark_config/*.yaml files.
"""

from __future__ import annotations

from competitors.mem0.adapter import Mem0Adapter
from competitors.memgpt.adapter import MemGPTAdapter
from competitors.memorybank.adapter import MemoryBankAdapter
from competitors.simple_rag.adapter import SimpleRAGAdapter

__all__ = [
    "Mem0Adapter",
    "MemGPTAdapter",
    "MemoryBankAdapter",
    "SimpleRAGAdapter",
]
