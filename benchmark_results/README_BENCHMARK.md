# Artificial Memory — README Benchmark

> Forget by compression. Recall by resolution. Reason with provenance.

| Metric | Diverse domains | Stress (near-duplicate) |
|---|---|---|
| Semantic recall accuracy@1 (vector / hybrid) | **66.7% / 79.2%** | 29.2% / 41.7% |
| Semantic recall accuracy@5 (vector / hybrid) | **100.0% / 100.0%** | 57.5% / 72.5% |
| Recall latency p50 / p95 | 27 / 32 ms | 26 / 28 ms |
| Token savings vs full-context | **69.6%** | 70.9% |
| Vector index build | 1191 ms | 1635 ms |
| Remember throughput | 531.6 writes/s | 551.5 writes/s |

*Seed 42, 120 synthetic memories and 50 queries per dataset.
"diverse" = memories from distinct domains (realistic retrieval), "stress" = near-identical templates (adversarial).
Python 3.11.15 on Windows 10 (Intel64 Family 6 Model 167 Stepping 1, GenuineIntel). Embeddings: all-MiniLM-L6-v2 (384-d, FAISS IndexFlatIP).*
