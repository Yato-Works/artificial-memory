#!/usr/bin/env python3
import sys, time, httpx
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, FROZEN_MODEL

answerer = OllamaAnswerer(model=FROZEN_MODEL, timeout_seconds=300.0)
print(f"Model: {answerer.model}, URL: {answerer.base_url}")

# Simple test
t0 = time.time()
ans = answerer.answer("What is the capital of France?", "Paris is the capital of France.")
print(f"Time: {time.time()-t0:.1f}s")
print(f"Answer: {ans.text[:80]}")
print(f"Tokens: prompt={ans.prompt_tokens} completion={ans.completion_tokens}")
