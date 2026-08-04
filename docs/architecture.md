# Architecture

## Purpose

PaperOS Knowledge Core provides one unified research-paper knowledge backend.

The architecture separates:

* immutable source artifacts;
* parser artifacts;
* canonical document data;
* writable semantic knowledge;
* rebuildable indexes;
* query-time processing.

Development stages are defined outside this document. This document describes the final system and its runtime boundaries.

## System context

```
AstrBot / CLI / HTTP client
            |
            v
    PaperOS API and services
    ├── ingestion
    ├── query
    ├── feedback
    └── document management
            |
  ┌─────────┼──────────┐
  v         v          v
MinerU    Cognee     local model
adapter   adapter      gateway
  |         |          |
  v         v          v
parser    graph,      embedding,
output    vector,     reranking,
          metadata    expansion
            |
            v
        DeepSeek
```

## Architectural principles

### Unified knowledge representation

Cognee DataPoints and their provenance form the unified writable knowledge layer.

There must not be separate writable knowledge systems for MinerU, Markdown Wiki pages, lexical search, and Cognee.

### Shared canonical objects

All components use centrally defined models for:

* SourceFile;
* ParseRun;
* Document;
* Paper;
* Section;
* Chunk;
* Element;
* ReferenceEntry;
* Entity;
* Claim;
* ConceptRelation;
* Summary;
* ResearchInsight;
* Feedback and Correction.

Ingestion and query modules must not define independent versions of these objects.

### Stable identity

Canonical objects use stable IDs within a declared schema and ID-generation version.

Graph nodes, vectors, lexical rows, provenance links, and API responses refer to these IDs.

### Provenance-first design

Every derived object or relation must identify its source.

Possible provenance targets include:

* source PDF;
* parse run;
* page;
* section;
* chunk;
* element;
* reference entry;
* previous derived knowledge.

### Rebuildable derived data

Indexes and semantic derivatives may be removed and rebuilt.

Original PDFs and raw parser responses remain preserved.

## Runtime components

## Application layer

The application layer provides use-case orchestration.

Main services:

* ingestion service;
* research-query service;
* feedback service;
* document service;
* rebuild service;
* health and status service.

The application layer coordinates adapters but does not directly implement provider-specific APIs.

## MinerU adapter

The MinerU adapter is responsible for:

* submitting PDF files;
* polling asynchronous tasks;
* handling synchronous parsing when supported;
* downloading results;
* validating returned artifacts;
* exposing a provider-neutral parse result;
* preserving provider metadata.

MinerU-specific response fields must not leak into downstream modules.

Only the MinerU adapter and canonical mapper may interpret provider-specific structures.

## Canonical document processor

The canonical processor transforms raw parser artifacts into versioned PaperOS objects.

Responsibilities include:

* Unicode and whitespace normalization;
* repeated header and footer cleanup;
* section hierarchy construction;
* paragraph and list handling;
* formula preservation;
* figure, table, caption, and footnote handling;
* reference-entry extraction;
* structure-aware chunking;
* page and source-span mapping;
* canonical snapshot persistence.

The canonical processor must remain independent of Cognee storage details.

## Cognee adapter

The Cognee adapter maps canonical and derived objects into Cognee DataPoints.

Responsibilities include:

* DataPoint declarations;
* registration of the Cognee default User, Dataset, Data item, and PipelineRun;
* `PipelineContext` propagation for every DataPoint and custom edge write;
* identity and embeddable fields;
* deterministic writes;
* typed graph relations;
* provenance relations;
* vector indexing;
* semantic enrichment;
* summaries;
* entity, relation, and claim extraction;
* read and traversal operations;
* destructive rebuild of derived stores.

The Cognee adapter must not parse raw MinerU payloads.

Cognee is also the runtime structural retrieval layer. Query execution uses
Cognee/LanceDB to locate Chunk, Entity, Claim, Summary, ConceptRelation, and
Triplet DataPoints, resolves those hits to Cognee graph nodes, performs typed
multi-hop traversal in the configured graph engine, and backtracks graph edge
provenance to canonical chunks. Cognee manifests and enrichment JSON files are
rebuild/audit artifacts; retrieval must not scan them as query indexes.

The canonical snapshot stores the selected PaperOS dataset name. The adapter
creates or resolves that name through Cognee's authorized Dataset API, binds
one Cognee Data item to each immutable source PDF, and passes the resulting
context to `add_data_points`. Dataset/Data and pipeline-run rows are verified
after the graph write. PaperOS remains a single-user deployment and uses
Cognee's default user; this does not introduce a PaperOS authentication layer.

## Lexical index

The lexical index uses SQLite FTS5.

It stores searchable views of:

* chunk text;
* document title;
* section path;
* formulas represented as text;
* captions;
* reference text;
* selected metadata.

Every lexical row references the corresponding canonical object ID.

The lexical index is not an authoritative knowledge store.

## Local model gateway

The local model gateway is a repository-owned service.

It exposes:

```
GET  /health
GET  /v1/models
POST /v1/embeddings
POST /v1/rerank
POST /v1/query-expansion
```

It loads only manually provided local model files.

Enabled models:

* EmbeddingGemma for embeddings;
* Qwen3 Reranker for query-candidate ranking;
* QMD Query Expansion for lexical, semantic, entity, relation, and HyDE expansion.

The gateway must never download models.

## DeepSeek adapter

DeepSeek is the external generative LLM provider.

It is used for:

* Cognee semantic extraction;
* entity and relation extraction;
* claim extraction;
* summaries;
* query planning when required;
* grounded answer synthesis.

DeepSeek configuration is owned by Cognee through `.env`.

No local generative LLM process is required by the default deployment.

## API layer

The API layer exposes single-user HTTP endpoints.

It provides:

* ingestion submission and status;
* comprehensive query;
* document listing and inspection;
* document deletion and reprocessing;
* feedback;
* health and dependency status.

Authentication and multi-user access control are outside the project scope.

## Worker

The Worker executes serialized long-running jobs.

Responsibilities include:

* MinerU task polling;
* artifact persistence;
* canonical processing;
* Cognee writes;
* index updates;
* semantic enrichment;
* feedback improvement;
* rebuild operations.

The default concurrency is one.

## Data layers

## Source layer

Stored under:

```
DATA_DIR/raw/
```

Contains immutable original PDF files.

Primary object:

* SourceFile.

## Parser-artifact layer

Stored under:

```
DATA_DIR/parsed/
```

Contains immutable MinerU outputs and parse manifests.

Primary objects:

* ParseRun;
* ParserArtifact.

## Canonical layer

Stored under:

```
DATA_DIR/canonical/
```

Contains versioned normalized document snapshots.

Primary objects:

* Document;
* Paper;
* Section;
* Chunk;
* Element;
* ReferenceEntry;
* CanonicalSnapshot.

## Semantic knowledge layer

Stored through Cognee under:

```
DATA_DIR/cognee/
```

Contains:

* canonical DataPoints;
* semantic entities;
* claims;
* concept relations;
* summaries;
* research insights;
* feedback-derived knowledge;
* graph and provenance relations.

This is the unified writable knowledge layer.

## Derived-index layer

Stored under:

```
DATA_DIR/indexes/
```

Contains:

* SQLite FTS index;
* index manifests;
* schema and model metadata.

It does not contain a second vector database. Vector indexes managed by Cognee
under `DATA_DIR/cognee/vector/` are the sole semantic vector projection.

## Model layer

Stored under:

```
DATA_DIR/models/
```

Contains manually provided model files.

## Test layer

Stored under:

```
DATA_DIR/test-corpus/
DATA_DIR/test-runs/
```

Contains user-supplied academic PDFs, manually verified expectations, query requirements, and isolated test execution artifacts.

## Ingestion data flow

```
PDF
→ file validation
→ SHA-256 and stable source registration
→ immutable source storage
→ MinerU parsing
→ immutable parser-artifact storage
→ canonical mapping
→ cleaning and classification
→ sections, elements, chunks, and references
→ canonical snapshot
→ Cognee DataPoints
→ graph and vector writes
→ lexical indexing
→ semantic enrichment
→ consistency validation
```

## Query data flow

```
question
→ query planning
→ query expansion
→ parallel retrieval
   ├── SQLite FTS lexical chunks
   ├── Cognee vector Chunk/Entity/Claim/Summary/Triplet hits
   ├── Cognee Entity and Claim hits
   ├── Cognee typed multi-hop graph traversal
   ├── global summaries
   └── confirmed knowledge
→ candidate normalization
→ stable-ID deduplication
→ weighted rank fusion
→ evidence backtracking
→ local reranking
→ source diversification
→ DeepSeek synthesis
→ answer and provenance
```

## Candidate contract

All retrieval channels return a common candidate shape containing:

* object ID;
* object type;
* source document ID;
* source section ID when applicable;
* source chunk IDs;
* candidate text;
* channel name;
* channel score;
* provenance;
* inferred or confirmed status.

Graph and summary candidates must resolve to source evidence before final synthesis.

## Query profiles

### Comprehensive

Uses every available retrieval channel.

### Truth

Places additional weight on:

* original text;
* lexical matches;
* direct provenance;
* user-confirmed knowledge.

### Associative

Places additional weight on:

* graph traversal;
* entities;
* cross-document relations;
* summaries;
* multi-hop context.

Profiles alter query-time weighting and budgets only. They do not use different stores.

## Storage ownership

* Source registration owns `raw/`.
* MinerU parsing owns `parsed/`.
* Canonical processing owns `canonical/`.
* Cognee adapter owns semantic knowledge writes.
* Lexical-index manager owns SQLite FTS.
* Model gateway owns only model process state.
* Feedback service owns confirmations, corrections, and rejection records.
* Exporters own read-only exported views.

## Schema evolution

The system is in pre-stable development.

The following may change destructively:

* canonical schema;
* chunking rules;
* DataPoint definitions;
* graph relation types;
* embedding dimensions;
* index schemas;
* ID-generation versions.

Schema changes must update explicit version fields.

When necessary, the system may rebuild canonical snapshots, Cognee stores, lexical indexes, and vector indexes.

The following remain immutable:

* original PDFs;
* source checksums;
* raw MinerU responses;
* parse manifests.

## Failure boundaries

Missing dependencies must cause explicit errors.

Examples include:

* missing PDF;
* invalid PDF header;
* missing MinerU API key;
* unavailable MinerU endpoint;
* failed MinerU task;
* missing GGUF file;
* unavailable local model gateway;
* invalid Cognee configuration;
* unavailable DeepSeek endpoint;
* schema-version mismatch;
* failed provenance validation.

The system must not silently switch providers or download missing resources.

## Process topology

Default processes:

```
Process 1: local model gateway
Process 2: PaperOS Worker
Process 3: PaperOS API
```

External dependencies:

```
MinerU Cloud or configured MinerU HTTP service
DeepSeek API
```

Local persistence:

```
SQLite
LanceDB
Kuzu
filesystem artifacts
```

## Architectural constraints

* One logical knowledge system.
* One canonical object model.
* One set of stable IDs per schema version.
* One writable semantic knowledge layer.
* No independent query-side document schema.
* No direct raw MinerU dependency outside the adapter and mapper.
* No automatic model or dataset downloads.
* No authentication or multi-user infrastructure.
* No requirement to preserve pre-stable derived databases across schema changes.
