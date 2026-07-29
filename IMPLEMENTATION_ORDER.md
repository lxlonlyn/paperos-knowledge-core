# Implementation Order

## Purpose

This file defines the mandatory coding and acceptance order for Codex.

It does not describe the runtime architecture.

Codex must begin with the first gate and may write code for a later gate only after the current cumulative gate passes.

A later gate must consume real output produced by all previous gates.

## General rules

* Use genuine academic PDFs from the configured test corpus.
* Do not use fake, mock, stub, synthetic, generated, simplified, or prerecorded document inputs.
* Do not use manually constructed MinerU responses.
* Do not manually seed canonical snapshots, Cognee stores, indexes, embeddings, or answers.
* Do not implement future gates in parallel.
* Do not bypass an upstream failure by patching replacement data into a downstream module.
* Every acceptance run begins with the public PDF ingestion entry point.
* Continue to the next gate automatically after the current gate passes.
* Stop only when a real dependency, configuration value, model file, permission, or test PDF is unavailable.

## Gate 1 — PDF intake and source registration

Implement:

1. PaperOS configuration loading.
2. Data-directory resolution.
3. Runtime path models.
4. Public PDF ingestion entry point.
5. PDF header and MIME validation.
6. File-size validation.
7. SHA-256 calculation.
8. Stable SourceFile ID.
9. Immutable source-file storage.
10. Ingestion-job persistence.
11. Duplicate-file handling.
12. Actionable errors.

Acceptance path:

```
genuine PDF
→ public ingestion entry point
→ SourceFile
→ immutable raw PDF
→ ingestion-job record
```

Verify:

* the stored PDF checksum matches the supplied PDF;
* stable IDs are reproducible;
* duplicate ingestion is handled deterministically;
* runtime artifacts remain under the configured data directory;
* no parser, canonical, Cognee, or query objects are manually created.

Do not write Gate 2 code until Gate 1 passes.

## Gate 2 — MinerU parsing and raw artifacts

Implement:

1. MinerU provider interface.
2. MinerU Cloud provider.
3. Configurable MinerU HTTP provider.
4. PDF submission.
5. Asynchronous task polling.
6. Timeout handling.
7. Provider-error mapping.
8. Result retrieval.
9. Raw response validation.
10. ParseRun creation.
11. Parser-artifact manifest.
12. Immutable Markdown, content-list, model-output, asset, and metadata persistence.
13. Artifact checksums.

Gate 2 must consume the SourceFile created by Gate 1.

Acceptance path:

```
genuine PDF
→ Gate 1
→ live MinerU OCR
→ ParseRun
→ immutable parser artifacts
```

Verify:

* the PDF registered by Gate 1 is the file sent to MinerU;
* the test run uses the live configured MinerU service;
* no prerecorded MinerU output is used;
* all returned artifacts are persisted;
* the ParseRun references the correct SourceFile;
* parser metadata and checksums are recorded;
* persisted artifacts remain unchanged after writing.

Do not write Gate 3 code until the complete Gate 1–2 path passes.

## Gate 3 — Canonical transformation

Implement:

1. Provider-neutral MinerU result schemas.
2. MinerU-to-canonical mapper.
3. Unicode and whitespace normalization.
4. repeated header and footer cleanup.
5. duplicate-content cleanup.
6. document metadata mapping.
7. section hierarchy.
8. element classification.
9. paragraph and list handling.
10. formula handling.
11. figure, table, caption, and footnote handling.
12. reference-entry extraction.
13. structure-aware chunking.
14. page and source-span provenance.
15. stable canonical IDs.
16. schema and pipeline versioning.
17. canonical snapshot persistence.
18. expected-corpus structural validation.

Gate 3 must consume the parser artifacts created by Gate 2 during the same cumulative run.

Acceptance path:

```
genuine PDF
→ Gate 1
→ Gate 2
→ canonical transformation
→ canonical snapshot
```

Verify:

* no manually constructed parser response is used;
* the canonical snapshot references its SourceFile and ParseRun;
* sections, chunks, elements, and references have stable IDs;
* elements retain page and parser provenance;
* expected title and structural checks pass;
* missing optional OCR fields do not crash the pipeline;
* unknown provider fields do not leak into downstream canonical interfaces.

Do not write Gate 4 code until the complete Gate 1–3 path passes.

## Gate 4 — Cognee storage and derived indexes

Implement:

1. Cognee runtime configuration.
2. centralized DataPoint declarations.
3. canonical-to-DataPoint mapping.
4. document, section, chunk, element, and reference DataPoints.
5. entity, claim, summary, and concept-relation DataPoints.
6. typed graph relations.
7. provenance relations.
8. Cognee repository writes and reads.
9. local embedding-gateway client.
10. vector indexing.
11. SQLite FTS5 lexical indexing.
12. index manifests.
13. reference resolution.
14. citation relations.
15. semantic enrichment through DeepSeek.
16. document summaries.
17. consistency validation.
18. destructive rebuild of derived data.

Gate 4 must consume the canonical snapshot created by Gate 3 during the same cumulative run.

Acceptance path:

```
genuine PDF
→ Gate 1
→ Gate 2
→ Gate 3
→ Cognee and derived indexes
```

Verify:

* no canonical objects are manually seeded;
* Cognee records refer to canonical IDs or explicit versioned ID mappings;
* graph, vector, metadata, and lexical records refer to the same source objects;
* stored DataPoints can be read back;
* all inferred objects and relations have source provenance;
* citation relations originate from real ReferenceEntry objects;
* local embeddings are actually generated;
* deleting rebuildable stores and rebuilding them from preserved artifacts succeeds.

Do not write query code until the complete Gate 1–4 path passes.

## Gate 5 — Comprehensive research query

Implement:

1. query request models;
2. comprehensive, truth, and associative profiles;
3. query planning;
4. local query expansion;
5. lexical retrieval;
6. semantic chunk retrieval;
7. entity and claim retrieval;
8. graph traversal;
9. global-context retrieval;
10. confirmed-knowledge retrieval;
11. common candidate model;
12. stable-ID deduplication;
13. weighted reciprocal-rank fusion;
14. evidence backtracking;
15. local Qwen3 reranking;
16. document and section diversification;
17. DeepSeek answer synthesis;
18. source evidence formatting;
19. provenance formatting;
20. CLI and API query entry points.

Gate 5 must query data generated by the complete Gate 1–4 path during the same cumulative run.

Acceptance path:

```
genuine PDF corpus
→ Gate 1
→ Gate 2
→ Gate 3
→ Gate 4
→ comprehensive research query
→ source evidence and provenance
```

Verify:

* query tests do not seed their own chunks, vectors, nodes, edges, summaries, or answers;
* lexical, semantic, and graph channels use shared canonical IDs;
* local query expansion is executed;
* local reranking is executed;
* graph and summary candidates backtrack to source chunks;
* returned answers identify source documents;
* truth queries satisfy direct-evidence requirements;
* associative queries use multiple documents when required;
* comprehensive queries use the channels declared by the test corpus.

Do not write feedback and operational completion code until Gate 5 passes.

## Gate 6 — Feedback, improvement, and operations

Implement:

1. feedback API and CLI;
2. confirmation records;
3. rejection records;
4. correction records;
5. versioned improvement objects;
6. protection of original evidence;
7. Worker lifecycle;
8. health and status reporting;
9. document listing and inspection;
10. document reprocessing;
11. document deletion;
12. index rebuild commands;
13. full API integration;
14. repeated-query consistency checks.

Acceptance path:

```
genuine PDF corpus
→ complete ingestion
→ comprehensive query
→ feedback or correction
→ improvement
→ rebuild
→ repeated query
```

Verify:

* feedback never modifies the original PDF;
* feedback never modifies raw MinerU artifacts;
* feedback never overwrites source chunks;
* corrections are stored as versioned derived knowledge;
* rebuild preserves source evidence;
* repeated queries can distinguish source facts, structured relations, system inference, and user-confirmed knowledge.

## Final completion criteria

Codex may report completion only after all gates pass in order:

```
Gate 1
Gate 1–2
Gate 1–3
Gate 1–4
Gate 1–5
Gate 1–6
```

Passing isolated unit tests, adapter tests, or downstream tests is not sufficient.

The final report must include:

* commands executed;
* test corpus used;
* successful cumulative gates;
* created artifact paths;
* external services contacted;
* model files loaded;
* remaining known limitations;
* any destructive rebuild performed.
