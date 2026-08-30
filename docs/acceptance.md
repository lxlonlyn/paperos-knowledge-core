# Chunk-first real-pipeline acceptance

The retrieval acceptance uses genuine academic PDFs already present in
`data/validation/corpus/papers/`. It does not create a second search corpus or
use synthetic retrieval fixtures, fixed embeddings, fixed reranker output, or
fixed LLM responses.

Run the single task entry point from the repository root:

```bash
conda run -n paperos python tests/validation/retrieval.py --rebuild
```

`--rebuild` isolates generated runtime data beneath
`data/validation/retrieval/output/runtime/` and executes the current chain:

```text
PDF
→ MinerU / configured parser
→ Canonical Document / Section / Element / ReferenceEntry
→ ChunkProjection and retrieval_text
→ canonical + scholarly Cognee graph mapping
→ optional semantic enrichment (disabled by default)
→ PAPEROS_CHUNKS vector index + SQLite FTS5 lexical index
→ lexical Chunk + vector Chunk retrieval
→ RRF → chunk_id dedup → rerank
→ optional local context / direct semantic relation expansion
→ whole-Chunk synthesis budget → canonical source-grounded Evidence
→ LLM answer
```

The default corpus selection is:

```text
volume_preserving.pdf
explicit_flows.pdf
nise.pdf
gaussian_splatting.pdf
```

## Acceptance gates

The run fails unless all of the following contracts hold:

1. Default search uses only lexical and vector Chunk discovery before RRF,
   mandatory `chunk_id` dedup, and rerank. Expansion is off by default.
2. Caller-supplied document/work IDs hard-filter the corpus. Natural-language
   query text does not infer a filter or select another search architecture.
3. Claim enrichment is off: the extraction prompt/schema has no Claim output,
   and newly generated Claim and ABOUT counts are both zero.
4. A real body citation resolves through `Chunk → ReferenceEntry → Work`,
   and the corresponding CITES relation records that body Chunk in
   `source_chunk_ids`.
5. Explicit local expansion starts from a first-stage hit and stays within the
   same document, region, and major section.
6. Direct semantic expansion starts from a seed Chunk, follows semantic objects
   grounded in that Chunk through one direct semantic relation, and resolves
   `relation.source_chunk_ids` to canonical Chunk candidates. CITES is validated
   as scholarly provenance, not used as ordinary semantic expansion. If the
   retained corpus has no genuine relation that can produce a case, the report
   records `NO_CASE`; it never fabricates a success.
7. Every final Evidence item satisfies the source-grounding invariant:
   `chunk_id` exists, `evidence.document_id == canonical_chunk.document_id`,
   and `evidence.text == canonical_chunk.text`.
8. At least one real query reaches final LLM synthesis, proving that the
   PDF-to-LLM chain completed.

The report includes the first-stage IDs, first rerank IDs, local/direct-semantic
expansion IDs, second rerank IDs, final selected IDs, and direct-relation
provenance needed for manual review. Citation/CITES checks remain separate
scholarly-provenance gates.

## Outputs

The entry point writes the human- and machine-reviewable reports to:

```text
data/validation/retrieval/output/acceptance.json
data/validation/retrieval/output/acceptance.md
```

The summary reports ingested paper and Chunk counts, Claim/ABOUT/CITES counts,
each acceptance status, real queries, retrieved Chunk IDs, final Evidence,
expansion traces, and whether the complete PDF-to-LLM chain ran.

Generated runtime and reports are replaceable validation artifacts. The PDFs
under `data/validation/corpus/` remain authoritative and must not be removed by
cleanup.

## Permanent Chunk-first contracts

Run the release base contracts independently of external services and without
pytest:

```bash
python tests/contract/test_runtime_query_contracts.py
python tests/contract/test_portable_data_paths.py
python tests/contract/test_citation_resolution.py
python tests/contract/test_document_regions.py
python tests/contract/test_chunk_boundaries.py
conda run -n paperos python tests/contract/test_cognee_retrieval_boundary.py
```

The real active-revision and pre-truncation vector-filter contracts additionally
require the Linux external Cognee/local-model boundary:

```bash
conda run -n paperos python tests/contract/test_active_canonical_revision.py
conda run -n paperos python tests/contract/test_query_filter_contracts.py
```

Together they enforce the production retrieval boundary, active-only lifecycle,
mandatory dedup, explicit hard filters, canonical evidence rehydration, trace
fields, and rejection of removed request fields such as `profile`.

This task intentionally has no previous-pipeline comparison, profile matrix,
ablation matrix, unique-rescue benchmark, or graph/Claim cost-benefit benchmark.
Those measurements are separate from proving that the one production
Chunk-first architecture is structurally correct and operational end to end.

## Structured reranking

Reranking scores rebuildable, snapshot-scoped `RerankSpan` ranges generated
from the canonical `SentenceUnit` structure during ChunkProjection construction.
The local model does not split strings at query time. Production ranking combines
the Full Chunk score with MaxP scores over structured spans targeting 256 tokens
with a 384-token hard maximum, using reciprocal rank fusion with `k = 60`.
The fused result resolves to the canonical parent Chunk. The authoritative
indexed and Evidence unit remains that canonical Chunk; reranking does not
rewrite Chunk text, retrieval text, embeddings, or Evidence provenance.

The retained Task 6B benchmark in
`data/validation/rerank_quality/output/benchmark.json` records the production
policy as `hybrid_full_structured_256_384_rrf` with `overall_status = PASS`.
Its historical `validated_head` remains the commit on which that benchmark
actually ran; release closure does not relabel it as current validation.
