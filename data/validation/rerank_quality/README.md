# Task 6B — Rerank Quality Optimization

Overall: **PASS**

Dev winner: `hybrid_full_structured_256_384_rrf`
Final recommendation: **adopt hybrid_full_structured_256_384_rrf**
Benchmark execution HEAD: `af2cc26d5ae36c7c75d89c36ea4b0a1300f748f1`

The benchmark used the retained five-paper active corpus. It did not run PDF ingestion, MinerU, semantic enrichment, or LLM synthesis.

## Dev aggregation comparison

| Strategy | Anchor coverage | MRR | Hit@5 | Hit@10 | Mean latency (s) | Scoring count |
|---|---:|---:|---:|---:|---:|---:|
| full_chunk | 1.000 | 0.900 | 1.000 | 1.000 | 8.352 | 442 |
| legacy96_maxp | 1.000 | 0.820 | 1.000 | 1.000 | 11.489 | 2020 |
| structured_256_384_maxp | 1.000 | 0.870 | 1.000 | 1.000 | 9.982 | 1510 |
| structured_256_384_top2_mean | 1.000 | 0.917 | 0.900 | 1.000 | 9.970 | 1510 |
| hybrid_full_structured_256_384_rrf | 1.000 | 0.933 | 1.000 | 1.000 | 13.429 | 1952 |

## Selection

- Span-size candidates: hybrid_full_structured_192_288_rrf, hybrid_full_structured_256_384_rrf, hybrid_full_structured_384_576_rrf, structured_192_288_top2_mean, structured_256_384_top2_mean, structured_384_576_top2_mean
- Holdout compared: structured_256_384_maxp, hybrid_full_structured_256_384_rrf
- Holdout reversal: False
- Production changed: True

Gold anchors are exact substrings of active canonical parent Chunks. Expected-Work accuracy means the expected Work appears in the final canonical Evidence set.
