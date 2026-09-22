import argparse
import yaml
import json
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from artificial_memory.runtime.facade import ArtificialMemoryRuntime, RuntimeConfig

@dataclass
class ApexResult:
    benchmark: str
    scale: Optional[str]
    question_id: str
    is_correct: bool
    predicted_answer: str
    ground_truth: str
    tokens_used: int
    latency_ms: float
    evidence_recall: float = 0.0
    provenance_valid: bool = False

class ApexRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path('C:/Users/smily/artificial_memory/benchmark/results/apex_frozen')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        rt_config = RuntimeConfig(
            ollama_model=self.config['global'].get('default_model', 'llama3.1'),
            default_provider=self.config['global'].get('model_provider', 'ollama'),
        )
        rt_config.retrieval_strategy = self.config['global'].get('retrieval_strategy', 'classic')
        
        self.runtime = ArtificialMemoryRuntime(config=rt_config)

    async def run_benchmark(self, bench_name: str, scale: Optional[str] = None, mode: str = 'full'):
        print(f"🚀 Starting AM Apex Evaluation: {bench_name} ({scale if scale else 'N/A'}) | Mode: {mode}")
        
        if bench_name == 'beam':
            results = await self._eval_beam(scale, mode)
        elif bench_name == 'locomo':
            results = await self._eval_locomo(mode)
        elif bench_name == 'longmemeval':
            results = await self._eval_longmem(mode)
        elif bench_name == 'perma':
            results = await self._eval_perma(mode)
        else:
            raise ValueError(f"Unsupported benchmark: {bench_name}")

        self._save_results(bench_name, scale, results)
        self._print_summary(bench_name, results)

    async def _eval_beam(self, scale: str, mode: str) -> List[ApexResult]:
        if not scale:
            raise ValueError("Scale (100K, 500K, 1M, 10M) is required for BEAM")
        
        dataset_path = Path(f'C:/Users/smily/artificial_memory/datasets/official/beam/{scale}')
        results = []
        
        chat_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        if mode == 'smoke':
            chat_dirs = chat_dirs[:2] 
            
        for chat_dir in chat_dirs:
            chat_id = chat_dir.name
            print(f"Processing chat {chat_id}...")
            
            topic_path = f"BEAM/{scale}/{chat_id}"
            self.runtime.conversation_manager.start_conversation(topic_path)
            
            chat_json_path = chat_dir / 'chat.json'
            if not chat_json_path.exists():
                continue
                
            with open(chat_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # BEAM structure: List[Batch] -> Batch['turns'] -> List[TurnGroup] -> List[Message]
                for batch in data:
                    turns_groups = batch.get('turns', [])
                    for turn_group in turns_groups:
                        for msg in turn_group:
                            role = "user" if msg.get('role', '').lower() == 'user' else "assistant"
                            content = msg.get('content', '')
                            if role == "user":
                                self.runtime.conversation_manager.log_user(content)
                            else:
                                self.runtime.conversation_manager.log_assistant(content)
            
            self.runtime.conversation_manager.end_conversation()
            
            # Answer probing questions
            pq_path = chat_dir / 'probing_questions' / 'probing_questions.json'
            if not pq_path.exists():
                pq_path = chat_dir / 'probing_questions.json'
                
            if not pq_path.exists():
                continue
                
            with open(pq_path, 'r', encoding='utf-8') as f:
                pq_data = json.load(f)
                
                # pq_data is a dict of categories: { 'abstention': [...], 'information_extraction': [...] }
                all_questions = []
                for cat_name, cat_questions in pq_data.items():
                    if isinstance(cat_questions, list):
                        all_questions.extend(cat_questions)
                
                if mode == 'smoke':
                    all_questions = all_questions[:5]

                for q in all_questions:
                    # Ground truth can be 'answer', 'ideal_answer', or 'ideal_response'
                    gt = q.get('answer') or q.get('ideal_answer') or q.get('ideal_response', '')
                    
                    start_time = time.time()
                    res = await self.runtime.chat(
                        message=q['question'],
                        topic=topic_path,
                        temperature=self.config['global']['temperature'],
                        use_context=True
                    )
                    latency = (time.time() - start_time) * 1000
                    
                    results.append(ApexResult(
                        benchmark='beam',
                        scale=scale,
                        question_id=q.get('id', 'unknown'),
                        is_correct=self._judge(res.response, gt),
                        predicted_answer=res.response,
                        ground_truth=gt,
                        tokens_used=res.total_tokens,
                        latency_ms=latency
                    ))
        return results

    async def _eval_locomo(self, mode: str) -> List[ApexResult]:
        print("Evaluating LoCoMo (Stub)...")
        return []

    async def _eval_longmem(self, mode: str) -> List[ApexResult]:
        print("Evaluating LongMemEval (Stub)...")
        return []

    async def _eval_perma(self, mode: str) -> List[ApexResult]:
        print("Evaluating PERMA (Stub)...")
        return []

    def _judge(self, pred: str, gt: str) -> bool:
        if not gt: return False
        return gt.strip().lower() in pred.strip().lower()

    def _save_results(self, bench: str, scale: Optional[str], results: List[ApexResult]):
        filename = f"{bench}_{scale if scale else 'all'}_results.json"
        with open(self.results_dir / filename, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    def _print_summary(self, bench: str, results: List[ApexResult]):
        if not results: 
            print("No results collected.")
            return
        acc = sum(1 for r in results if r.is_correct) / len(results) * 100
        avg_tokens = sum(r.tokens_used for r in results) / len(results)
        avg_lat = sum(r.latency_ms for r in results) / len(results)
        
        print(f"\\n--- {bench.upper()} Apex Summary ---")
        print(f"Accuracy: {acc:.2f}%")
        print(f"Avg Tokens/Q: {avg_tokens:.1f}")
        print(f"Avg Latency: {avg_lat:.1f}ms")
        print(f"------------------------------\\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', required=True, choices=['locomo', 'longmemeval', 'beam', 'perma'])
    parser.add_argument('--scale', help='Scale for BEAM (100K, 500K, 1M, 10M)')
    parser.add_argument('--mode', default='full', choices=['smoke', 'full'])
    args = parser.parse_args()
    
    runner = ApexRunner('C:/Users/smily/artificial_memory/benchmark_config/apex_config.yaml')
    await runner.run_benchmark(args.bench, args.scale, args.mode)

if __name__ == '__main__':
    asyncio.run(main())
