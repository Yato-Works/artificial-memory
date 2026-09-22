"""Cross-LLM Model Invariance Benchmark (Phase 6-C).

Compares No Memory vs. Simple RAG vs. Artificial Memory (AM)
across multiple model scales (e.g. 1.5B, 3.8B, 7B) using local Ollama.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.context.compiler import CognitiveContextCompiler
from artificial_memory.research.benchmarks.adversarial_builder import AdversarialBuilder
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


@dataclass
class ModelEvaluationRow:
    model_name: str
    no_memory_acc: float
    rag_acc: float
    am_acc: float


class CrossModelBenchmark:
    """Executes cross-model evaluation across parameter scales."""

    def __init__(
        self,
        models: list[str] | None = None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.models = models or ["qwen2.5:1.5b", "phi4-mini:latest", "qwen2.5-coder:7b"]
        self.ollama_url = ollama_url
        self.extractor = UniversalIRExtractor()
        self.compiler = CognitiveContextCompiler()

    def run(self) -> list[ModelEvaluationRow]:
        """Run evaluation on the Adversarial Suite for all models."""
        builder = AdversarialBuilder()
        dataset = builder.build_adversarial_suite()
        scenarios = dataset.scenarios
        questions = dataset.questions

        # Flatten dialogue text for RAG and AM
        all_dialogue_lines: list[str] = []
        for s in scenarios:
            for t in s.turns:
                all_dialogue_lines.append(t.content)

        # Compile AM StructuredIR records
        am_records = []
        for line in all_dialogue_lines:
            am_records.extend(self.extractor.extract(line, default_source="user"))

        results: list[ModelEvaluationRow] = []

        for model in self.models:
            print(f"\n=======================================================")
            print(f"  Evaluating Model: {model}")
            print(f"=======================================================")
            llm = OllamaAnswerer(model=model, base_url=self.ollama_url)

            no_mem_correct = 0
            rag_correct = 0
            am_correct = 0
            total = len(questions)

            for q in questions:
                query = q.question
                # ground_truth is tuple of tuples, e.g. (("ClickHouse",),)
                ground_truth = q.ground_truth[0][0].lower() if q.ground_truth and q.ground_truth[0] else ""
                eval_type = "exact_match"

                # 1. No Memory Condition
                no_mem_ans = llm.answer(question_text=query, context="")
                if self._check_match(no_mem_ans.text, ground_truth, eval_type):
                    no_mem_correct += 1

                # 2. Simple RAG Condition (naive keyword context)
                rag_context = "\n".join(all_dialogue_lines[:6])
                rag_ans = llm.answer(question_text=query, context=rag_context)
                if self._check_match(rag_ans.text, ground_truth, eval_type):
                    rag_correct += 1

                # 3. Artificial Memory Condition (Compiled Context IR)
                compiled = self.compiler.compile(query=query, records=am_records)
                am_ans = llm.answer(question_text=query, context=compiled.text)
                if self._check_match(am_ans.text, ground_truth, eval_type):
                    am_correct += 1

            row = ModelEvaluationRow(
                model_name=model,
                no_memory_acc=no_mem_correct / total,
                rag_acc=rag_correct / total,
                am_acc=am_correct / total,
            )
            results.append(row)
            print(f"Results for {model}: No Mem={row.no_memory_acc:.1%}, RAG={row.rag_acc:.1%}, AM={row.am_acc:.1%}")

        return results

    def _check_match(self, response: str, ground_truth: str, eval_type: str) -> bool:
        resp_lower = response.lower()
        if eval_type == "exact_match" or eval_type == "substring":
            return ground_truth in resp_lower
        elif eval_type == "boolean":
            if ground_truth == "true" or ground_truth == "yes":
                return "yes" in resp_lower or "true" in resp_lower
            return "no" in resp_lower or "false" in resp_lower
        return ground_truth in resp_lower


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Model Memory Benchmark")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "phi4-mini:latest"])
    args = parser.parse_args()

    bench = CrossModelBenchmark(models=args.models)
    rows = bench.run()

    print("\n" + "=" * 60)
    print("  Cross-LLM Memory Invariance Benchmark Results")
    print("=" * 60)
    print(f"{'Model Scale':<20} | {'No Memory':<12} | {'Simple RAG':<12} | {'Artificial Memory':<18}")
    print("-" * 60)
    for r in rows:
        print(f"{r.model_name:<20} | {r.no_memory_acc:<12.1%} | {r.rag_acc:<12.1%} | {r.am_acc:<18.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
