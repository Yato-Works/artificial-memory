# AM Apex official benchmark laboratory

This lab makes an AM score reproducible rather than merely impressive. It keeps
upstream code and benchmark payloads outside version control, while committing the
exact source revision and the freeze protocol in `benchmark_config/official_benchmarks.json`.

## Installed official sources

| Benchmark | Official code | Pinned revision | Data location / official source |
| --- | --- | --- | --- |
| LoCoMo | `third_party/benchmarks/locomo` | `3eb6f2c` | bundled `data/locomo10.json` |
| LongMemEval | `third_party/benchmarks/longmemeval-official` | `9e0b455` | `xiaowu0162/longmemeval-cleaned` |
| BEAM | `third_party/benchmarks/beam` | `b2da22e` | bundled `chats/100K`, `500K`, `1M`, `10M` |
| PersonaMem / v2 | `third_party/benchmarks/personamem` | `d07e6ad` | `bowen-upenn/PersonaMem`, `bowen-upenn/ImplicitPersona` |
| PERMA | `third_party/benchmarks/perma` | `d678640` | `ustclsc/PERMA` |

Run this before an experiment:

```powershell
python scripts/benchmarks/verify_official_benchmarks.py
```

An existing directory alone does not mean a large dataset is complete. The verifier
checks required files and, for multi-file releases such as PersonaMem-v2 and BEAM,
the expected lower bound on file counts.

If it reports missing Hugging Face payloads, obtain them using the official Hub CLI:

```powershell
# All standard data; BEAM is already included in its official checkout.
./scripts/benchmarks/bootstrap_official_benchmarks.ps1 -DownloadData

# Include PersonaMem's 128K/1M payloads and separate BEAM Hub snapshots as well.
./scripts/benchmarks/bootstrap_official_benchmarks.ps1 -DownloadData -IncludeLargeScale

# Separate environments prevent version conflicts among official projects.
./scripts/benchmarks/bootstrap_official_benchmarks.ps1 -CreateEvaluationEnvs
```

For the large multi-file PersonaMem-v2 and PERMA releases, authenticate before the
download to avoid Hugging Face anonymous rate limits, then rerun the same command:

```powershell
hf auth login
./scripts/benchmarks/bootstrap_official_benchmarks.ps1 -Benchmark personamem_v2,perma -DownloadData
```

The bootstrap uses one download worker for these payloads. Downloads are resumable;
the verifier deliberately refuses to label a partial PersonaMem-v2 snapshot ready.

The script never reads or writes API keys. Official LLM-as-judge evaluations require
their provider credentials to be supplied by the operator through the relevant
upstream environment configuration. LoCoMo's published dependency lock is a
Linux-only conda export, so exact upstream reproduction belongs in Linux/WSL; its
official dataset is available on this Windows checkout and AM's adapter can run here.
BEAM and PersonaMem's published locks also contain Linux-only CUDA packages. On
Windows, the bootstrap script installs an evaluation-only CPU-compatible variant:
it preserves every non-CUDA upstream pin and uses the platform's Torch wheel. Use
Linux/WSL for byte-identical GPU reproduction.

The BEAM scorer constructs its upstream judge client during import. Before invoking
it, set the provider configuration only in the active shell (never commit it):

```powershell
$env:OPENAI_API_KEY = "..."
# Configure the judge/model mapping in third_party/benchmarks/beam/src/llms_config.json
```

## Frozen evaluation contract

For every score that appears in a report, freeze all of the following together:

1. Dataset file and SHA-256; dataset release/version; upstream code commit.
2. AM code commit and a hash of its run configuration.
3. Reader model, judge model, prompt/template, temperature, seed, top-k, and context budget.
4. Complete answer file, scorer output, category denominators, evidence IDs, retrieval trace, and latency/tokens per question.

Use the manifest writer only after answer and result files exist:

```powershell
python scripts/benchmarks/capture_frozen_manifest.py `
  --benchmark longmemeval `
  --dataset datasets/official/longmemeval/longmemeval_oracle.json `
  --config benchmark_config/official_benchmarks.json `
  --answers benchmark_results/official/longmemeval_answers.jsonl `
  --result benchmark_results/official/longmemeval_scores.json `
  --model local-reader-id --judge judge-model-id --seed 42 --temperature 0 `
  --top-k 10 --context-token-budget 700
```

Never combine different dataset variants or judge configurations into one claimed
score. In particular, report LongMemEval `oracle`, `S`, and `M` separately; report
BEAM 100K/500K/1M/10M separately; and report each PersonaMem context scale
separately. A previous high score becomes a *candidate* result until this manifest
exists.

## Recommended expedition order

1. Freeze and reproduce LoCoMo and LongMemEval with their official data and scorer.
2. Run BEAM 100K, then 500K, 1M, and 10M in that order; record retrieval recall and p50/p95 latency at every scale.
3. Run PersonaMem 32K before the 128K/1M scales, then PersonaMem-v2.
4. Run PERMA MCQ smoke test, clean/noisy single-domain, multi-domain, and finally interactive protocol.

The AM Apex scorecard should contain accuracy, evidence recall, provenance validity,
mean context tokens, p50/p95 retrieval latency, and abstention precision/recall. Do
not call a result "state of the art" until its exact protocol matches the comparison.

## Official scorer entry points

These commands keep official code untouched and point it at the centrally pinned
data snapshot. Replace the answer/result paths with the output from the AM adapter.

```powershell
# LongMemEval: official QA evaluator (from the upstream checkout).
Push-Location third_party/benchmarks/longmemeval-official/src/evaluation
& "C:/absolute/path/.venv-benchmarks/longmemeval/Scripts/python.exe" evaluate_qa.py `
  judge-model-id C:/absolute/path/am_answers.jsonl `
  C:/absolute/path/datasets/official/longmemeval/longmemeval_oracle.json
Pop-Location

# BEAM: official LLM-as-a-judge evaluator. Its answers must preserve the upstream
# probing-question structure and include llm_response for every question.
Push-Location third_party/benchmarks/beam
& "C:/absolute/path/.venv-benchmarks/beam/Scripts/python.exe" -m src.evaluation.run_evaluation `
  --input_directory C:/absolute/path/am_beam_results/1M --chat_size 1M `
  --start_index 0 --end_index 35 --max_workers 1 --allowed_result_files am_answers.json
Pop-Location

# PersonaMem: upstream inference accepts explicit data paths, so no dataset copy is needed.
Push-Location third_party/benchmarks/personamem
& "C:/absolute/path/.venv-benchmarks/personamem/Scripts/python.exe" inference.py `
  --question_path C:/absolute/path/datasets/official/personamem/questions_32k.csv `
  --context_path C:/absolute/path/datasets/official/personamem/shared_contexts_32k.jsonl `
  --result_path C:/absolute/path/benchmark_results/official/personamem_32k.csv
Pop-Location
```

For BEAM and PersonaMem, substitute the absolute workspace path for
`C:/absolute/path`; quoted PowerShell paths avoid problems with spaces. PERMA uses
its upstream `code/src/evaluation.py` entry point after its `data` payload is
complete; start with its documented `--smoke_test` invocation.
