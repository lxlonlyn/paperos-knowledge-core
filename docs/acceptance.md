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
→ Claim-disabled Cognee semantic enrichment
→ PAPEROS_CHUNKS vector index + SQLite FTS5 lexical index
→ lexical Chunk + vector Chunk retrieval
→ RRF → chunk_id dedup → rerank
→ optional post-hit local / citation / graph expansion
→ canonical source-grounded Evidence
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
6. Explicit graph expansion starts from a first-stage Chunk, follows bounded
   typed graph provenance, and returns only canonical Chunks. If the retained
   corpus has no genuine relation that can produce a case, the report records
   `NO_CASE`; it never fabricates a success.
7. Every final Evidence item satisfies the source-grounding invariant:
   `chunk_id` exists, `evidence.document_id == canonical_chunk.document_id`,
   and `evidence.text == canonical_chunk.text`.
8. At least one real query reaches final LLM synthesis, proving that the
   PDF-to-LLM chain completed.

The report includes the first-stage IDs, first rerank IDs, local/citation/graph
expansion IDs, second rerank IDs, final selected IDs, and graph traversal
provenance needed for manual review.

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

Run the fast contracts independently of external services:

```bash
conda run -n paperos python -m pytest tests/contract -q
conda run -n paperos python tests/contract/test_cognee_retrieval_boundary.py
```

They enforce the production retrieval boundary, mandatory dedup, explicit
expansion behavior, canonical evidence rehydration, trace fields, and rejection
of removed request fields such as `profile`.

This task intentionally has no previous-pipeline comparison, profile matrix,
ablation matrix, unique-rescue benchmark, or graph/Claim cost-benefit benchmark.
Those measurements are separate from proving that the one production
Chunk-first architecture is structurally correct and operational end to end.
