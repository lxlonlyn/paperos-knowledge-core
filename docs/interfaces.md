# Interfaces

## Purpose

This document defines stable runtime and module boundaries for PaperOS Knowledge Core.

It describes the final interfaces between components.

Development and cumulative acceptance order is defined in `IMPLEMENTATION_ORDER.md`.

## Interface principles

* Modules exchange typed models rather than unvalidated dictionaries.
* Provider-specific fields are isolated inside provider adapters.
* Canonical models are the boundary between parsing and knowledge storage.
* Cognee, lexical indexing, and retrieval use the same canonical IDs.
* Missing dependencies cause explicit errors.
* External services must not be silently replaced.
* No interface may trigger automatic model, dataset, PDF, or dependency downloads.

## Configuration interface

## PaperOS configuration

PaperOS configuration is loaded from:

```
config/paperos.toml
```

Supported override order:

```
command-line option
→ environment variable
→ paperos.toml
→ default value
```

The main runtime data path is resolved through:

```
--data-dir
→ PAPEROS_DATA_DIR
→ project.data_dir
→ ~/paperos-knowledge-core/data
```

The configuration loader returns one validated PaperOS configuration object.

Modules must not parse TOML independently.

MinerU credentials may be persisted as `mineru_ocr.api_key` in the project-local
`config/paperos.toml`, which is excluded from Git. The environment variable named by
`mineru_ocr.api_key_env` overrides the persisted value when non-empty. Consumers must
use the validated secret field and must not serialize or log its plaintext value.

## Cognee configuration

Cognee configuration is loaded from:

```
.env
```

It contains:

* LLM provider;
* LLM model;
* LLM endpoint;
* LLM API key;
* embedding provider;
* embedding endpoint;
* embedding dimensions;
* relational-store configuration;
* vector-store configuration;
* graph-store configuration;
* Cognee runtime paths.

PaperOS-specific MinerU, retrieval, testing, and local-model-path settings must not be added to `.env`.

## Path interface

All runtime paths are provided by `DataPaths`.

Required properties:

* `root`
* `raw`
* `parsed`
* `canonical`
* `cognee`
* `indexes`
* `models`
* `jobs`
* `cache`
* `exports`
* `logs`
* `test_corpus`
* `test_runs`
* `tmp`

Other modules must not construct data-directory paths independently.

## Public ingestion interface

The ingestion service accepts one document ingestion request.

Logical input:

* local PDF path or uploaded PDF stream;
* dataset name;
* optional user metadata;
* optional parser options;
* optional reprocess flag.

Logical output:

* ingestion job ID;
* source file ID;
* current status;
* duplicate status;
* optional existing document ID.

The ingestion public entry point is responsible for orchestrating the complete ingestion operation.

Callers must not invoke internal normalization, Cognee, or index modules directly.

## Source registry interface

The source registry provides:

* PDF validation;
* SHA-256 calculation;
* SourceFile lookup;
* SourceFile registration;
* immutable source storage;
* duplicate detection;
* ingestion-job creation;
* ingestion-job status updates.

Expected operations:

```
register_source(...)
get_source(...)
find_source_by_sha256(...)
create_job(...)
update_job(...)
get_job(...)
```

The registry returns typed SourceFile and IngestionJob objects.

## MinerU OCR provider interface

Every MinerU provider implements a common interface.

Required operations:

```
health_check()
submit_pdf(...)
get_task_status(...)
fetch_result(...)
parse_sync(...)
```

`parse_sync` may report that synchronous parsing is unsupported.

Provider implementations include:

* MinerU Cloud;
* configurable MinerU HTTP service.

The provider interface accepts a registered SourceFile and validated parse options.

It returns a provider-neutral parse task or parse result.

## MinerU provider-neutral result

`MinerUParseResult` contains:

* provider name;
* provider task ID;
* provider status;
* request metadata;
* response metadata;
* Markdown artifact;
* content-list artifacts;
* model-output artifacts;
* returned assets;
* provider version when available;
* warnings;
* raw metadata.

Unknown provider fields may be retained in `raw_metadata`.

Only the MinerU adapter and canonical mapper may inspect provider-specific raw metadata.

## Parser-artifact repository interface

The parser-artifact repository persists immutable ParseRuns and ParserArtifacts.

Expected operations:

```
create_parse_run(...)
update_parse_run_status(...)
persist_artifact(...)
get_parse_run(...)
list_artifacts(...)
verify_artifact_checksums(...)
```

The repository must not parse, clean, classify, or chunk document content.

## Canonical mapping interface

The canonical mapper accepts:

* SourceFile;
* completed ParseRun;
* persisted ParserArtifacts;
* canonical-processing configuration.

It returns one complete CanonicalSnapshot containing:

* Document;
* sections;
* elements;
* chunks;
* reference entries;
* provenance mappings;
* transformation warnings;
* version metadata.

Expected operation:

```
build_canonical_snapshot(...)
```

The canonical mapper is the only boundary allowed to interpret MinerU content structures.

The mapper must not write to Cognee.

## Canonical repository interface

The canonical repository persists and retrieves complete canonical snapshots.

Expected operations:

```
save_snapshot(...)
get_snapshot(...)
get_document(...)
list_sections(...)
list_elements(...)
list_chunks(...)
list_references(...)
verify_snapshot(...)
delete_rebuildable_snapshot(...)
```

The repository must preserve schema and pipeline versions.

## Cognee repository interface

The Cognee repository accepts canonical and derived domain objects.

Expected operations:

```
upsert_document_graph(..., dataset_name, source, title)
upsert_datapoints(...)
get_datapoint(...)
get_document_graph(...)
delete_document_derived_data(...)
rebuild_document(...)
traverse_relations(...)
resolve_provenance(...)
```

All writes must use centralized DataPoint declarations.

The Cognee repository must not accept raw MinerU payloads.

Runtime retrieval uses the repository rather than its persisted manifests:

```
search_vectors(...)
traverse(..., depth, edge_types)
verify_vector_indexes(...)
vector_status(...)
verify_dataset_binding(...)
list_datasets(...)
```

`search_vectors` queries Cognee's configured vector engine and resolves every
hit through the Cognee graph node carrying the same deterministic ID.
`traverse` delegates typed multi-hop expansion to the configured Cognee graph
engine and returns canonical chunk provenance stored on graph nodes and edges.
Manifest and enrichment JSON files are not retrieval indexes.

`upsert_document_graph` must resolve or create an authorized Cognee Dataset,
register the immutable PDF as a Cognee Data item, start a Cognee PipelineRun,
and pass a `PipelineContext(user, dataset, data_item, pipeline_run_id, ...)` to
`add_data_points`. A successful return requires relational and graph provenance
readback. The selected dataset comes from the versioned CanonicalSnapshot, so a
CLI override survives rebuild.

## Cognee pipeline interface

The PaperOS Cognee pipeline receives one verified CanonicalSnapshot.

It performs:

* deterministic DataPoint mapping;
* graph relation creation;
* vector indexing;
* entity extraction;
* relation extraction;
* claim extraction;
* summary generation;
* provenance linking;
* consistency validation.

Expected operation:

```
ingest_canonical_snapshot(...)
```

The pipeline returns a typed ingestion report containing:

* written object IDs;
* written relation IDs;
* vector-index status;
* semantic-enrichment status;
* warnings;
* failures.

## Lexical-index interface

The lexical index exposes:

```
initialize()
upsert_chunks(...)
delete_document(...)
search(...)
rebuild(...)
status()
```

Indexed records reference canonical object IDs.

Search input contains:

* query text;
* dataset;
* optional document filters;
* optional section filters;
* candidate limit.

Search output contains lexical candidates using the common candidate contract.

## Embedding interface

Cognee exclusively owns DataPoint and query embeddings and accesses them through
an OpenAI-compatible endpoint. PaperOS does not persist embedding BLOBs or build
a parallel vector database.

Endpoint:

```
POST /v1/embeddings
```

The request includes:

* model;
* input;
* optional dimensions.

The response follows the OpenAI-compatible embedding shape.

The configured implementation loads the local EmbeddingGemma GGUF file.

The embedding client must not download a missing model.

## Reranking interface

Endpoint:

```
POST /v1/rerank
```

Logical request:

* query;
* candidate texts;
* candidate IDs;
* optional instruction;
* result limit.

Logical response:

* candidate ID;
* original index;
* relevance score;
* final rank.

Reranking must operate on real retrieved candidates.

## Query-expansion interface

Endpoint:

```
POST /v1/query-expansion
```

Logical request:

* original query;
* query profile;
* optional known entities;
* optional conversation context.

Logical response:

* lexical queries;
* semantic queries;
* entity queries;
* relation queries;
* HyDE text;
* optional time scope;
* optional required evidence type.

The query-expansion service must not write to the knowledge store.

## Local model gateway interface

Required routes:

```
GET  /health
GET  /v1/models
POST /v1/embeddings
POST /v1/rerank
POST /v1/query-expansion
```

`GET /health` reports:

* enabled models;
* configured paths;
* file-existence state;
* model-load state;
* last load error.

The gateway must validate enabled model paths before serving requests.

It must never download models.

## DeepSeek and LLM interface

Cognee owns the primary LLM configuration through `.env`.

The LLM adapter exposes typed operations for:

* structured extraction;
* summary generation;
* query planning;
* answer synthesis.

Expected operations:

```
generate_structured(...)
generate_text(...)
health_check(...)
```

The adapter must use the configured DeepSeek endpoint.

It must not silently switch to another provider.

Structured responses must be validated before storage.

## Query-plan interface

The query planner accepts:

* original query;
* retrieval profile;
* dataset;
* optional filters;
* optional conversation context.

It returns:

* original query;
* expanded queries;
* recognized entities;
* requested relations;
* time scope;
* graph depth;
* retrieval-channel budgets;
* required evidence type;
* answer language.

The planner does not retrieve or store knowledge.

## Retrieval-channel interface

Every retrieval channel implements a shared interface.

Expected operation:

```
retrieve(query_plan) -> list[RetrievalCandidate]
```

Supported channels include:

* lexical;
* semantic chunk;
* semantic entity;
* semantic claim;
* graph;
* global context;
* confirmed knowledge.

## Common retrieval candidate

Every retrieval channel returns a common candidate object.

Required fields:

* `object_id`
* `object_type`
* `channel`
* `text`
* `channel_score`
* `source_document_id`
* `provenance`

Optional fields:

* `source_section_id`
* `source_chunk_ids`
* `source_element_ids`
* `relation_path`
* `status`
* `metadata`

A candidate without source evidence must be labeled as inferred or global context.

## Candidate-fusion interface

Candidate fusion accepts ranked candidate lists from multiple channels.

It performs:

* stable-ID normalization;
* duplicate merging;
* weighted reciprocal-rank fusion;
* score metadata preservation.

Expected operation:

```
fuse(candidate_lists, profile) -> list[FusedCandidate]
```

Fusion must not create new knowledge objects.

## Evidence-backtracking interface

Evidence backtracking resolves a candidate to source evidence.

Expected operation:

```
resolve_evidence(candidate) -> EvidenceBundle
```

`EvidenceBundle` contains:

* source document;
* source sections;
* source chunks;
* source elements;
* page information;
* provenance chain;
* relation derivation where applicable.

Graph relations, summaries, claims, and insights must be backtracked before answer synthesis.

## Diversification interface

Diversification accepts reranked evidence candidates.

It applies:

* per-document limits;
* per-section limits;
* duplicate-evidence suppression;
* support/contradiction balancing;
* source diversity.

Expected operation:

```
diversify(candidates, profile) -> list[EvidenceCandidate]
```

## Answer-synthesis interface

Answer synthesis accepts:

* original query;
* retrieval profile;
* evidence bundles;
* confirmed knowledge;
* output-language requirement.

It returns:

* answer text;
* source citations;
* evidence IDs;
* source document IDs;
* provenance;
* distinctions between source fact, structured relation, inference, and user-confirmed knowledge;
* warnings for insufficient evidence.

The answer synthesizer must not present unsupported inference as source fact.

## Feedback interface

Feedback input contains:

* target object or answer ID;
* feedback type;
* optional replacement text;
* optional evidence IDs;
* optional comment.

Supported feedback types:

* accept;
* reject;
* correct;
* confirm;
* mark unsupported.

Feedback processing creates versioned Feedback and Correction objects.

It must not overwrite source artifacts or canonical source text.

## Improvement interface

Improvement accepts validated feedback and current knowledge objects.

It may create:

* confirmed claims;
* rejected claims;
* corrections;
* superseding insights;
* updated summaries;
* truth-weighting data.

It must preserve the provenance chain and previous versions.

## Job queue interface

The queue supports low-concurrency single-user operation.

Expected operations:

```
enqueue(...)
claim_next(...)
update_status(...)
mark_completed(...)
mark_failed(...)
get_job(...)
list_jobs(...)
```

Default worker concurrency is one.

The queue does not require distributed infrastructure.

## Health interface

Health checks report independently on:

* PaperOS application;
* data paths;
* MinerU;
* DeepSeek;
* local model gateway;
* embedding model;
* reranker model;
* query-expansion model;
* Cognee relational store;
* Cognee vector store;
* Cognee graph store;
* lexical index;
* job database.

Health output must distinguish:

* healthy;
* degraded;
* unavailable;
* misconfigured.

## Public HTTP API

Expected routes:

```
POST   /api/v1/ingest
GET    /api/v1/ingest/{job_id}
POST   /api/v1/query
GET    /api/v1/documents
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST   /api/v1/documents/{document_id}/reprocess
POST   /api/v1/feedback
GET    /api/v1/health
GET    /api/v1/datasets
GET    /api/v1/datasets/{dataset_id}/data
GET    /api/v1/datasets/{dataset_id}/graph
GET    /api/v1/visualize?dataset_id={dataset_id}
```

Authentication is not required.

## Failure contract

External and internal failures must use typed errors.

Required error categories include:

* configuration error;
* invalid PDF;
* missing source file;
* MinerU authentication error;
* MinerU quota error;
* MinerU timeout;
* MinerU parse failure;
* parser-artifact validation error;
* canonical mapping error;
* schema-version mismatch;
* missing local model;
* local-model load failure;
* Cognee write failure;
* lexical-index failure;
* DeepSeek failure;
* insufficient evidence;
* provenance validation failure.

Errors must include:

* stable error code;
* readable message;
* affected object or path;
* retryability;
* underlying provider information when safe.

## Compatibility and evolution

Interfaces are versioned during pre-stable development.

Breaking changes may rebuild canonical data, Cognee stores, and indexes.

Provider adapters, canonical models, and query interfaces must remain separated so that:

* MinerU providers may change;
* canonical fields may evolve;
* Cognee DataPoints may evolve;
* embedding dimensions may change;
* retrieval channels may change.

Original PDFs and raw parser artifacts remain the stable reconstruction boundary.
