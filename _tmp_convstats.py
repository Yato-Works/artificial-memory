import json
for i in range(10):
    d = json.load(open(f'benchmark_results/locomo10_runs/baseline_retrieval/conv_{i}_results.json', encoding='utf-8'))
    s = d['summary']
    print(i, f"ora={s['overall_oracle_recall']*100:.1f}% tok={s['mean_tokens']:.0f}")
