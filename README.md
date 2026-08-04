# PaperOS Knowledge Core

PaperOS Knowledge Core is a single-user research-paper ingestion, knowledge storage, and comprehensive retrieval backend.

It accepts genuine academic PDF files, parses them through MinerU OCR, converts parser output into a canonical document model, writes structured knowledge into Cognee, builds lexical and vector indexes, and answers research questions through hybrid retrieval.

## Core capabilities

### Ingestion

* Accept genuine academic PDF files.
* Register source files using stable IDs and SHA-256 checksums.
* Call MinerU OCR through a configurable external API.
* Preserve raw MinerU output as immutable parser artifacts.
* Normalize and classify document structure.
* Construct sections, chunks, formulas, figures, tables, captions, footnotes, lists, and references.
* Convert canonical objects into Cognee DataPoints.
* Build graph, vector, metadata, and lexical indexes.
* Preserve provenance from derived knowledge back to the original PDF.

### Query

* Generate lexical, semantic, entity, relation, and HyDE queries.
* Run lexical, vector, graph, global-context, and confirmed-knowledge retrieval.
* Merge candidates by stable canonical object IDs.
* Backtrack graph and semantic results to source chunks.
* Rerank candidates with a local Qwen3 reranker.
* Generate grounded answers through DeepSeek.
* Return source evidence, document identifiers, section information, page information, and provenance.

### Knowledge improvement

* Record confirmations, corrections, and rejections.
* Store improvements as versioned derived knowledge.
* Preserve the original PDF, raw parser output, and source chunks.
* Rebuild Cognee stores and derived indexes when schemas change.

## System overview

```
User / AstrBot / Agent / HTTP API
                |
                v
      PaperOS application layer
      ├── ingestion service
      ├── query service
      ├── feedback service
      └── document management
                |
      ┌─────────┴─────────┐
      v                   v
MinerU OCR adapter    Cognee adapter
      |                   |
      v                   v
raw parser output    unified knowledge layer
      |                   |
      └─────────┬─────────┘
                v
     canonical document model
                |
      ┌─────────┼─────────┐
      v         v         v
   Cognee     SQLite     private local
   stores      FTS       inference
```

## Source-of-truth rules

* Original PDFs are immutable source artifacts.
* Raw MinerU responses are immutable parser artifacts.
* Canonical snapshots are versioned transformation artifacts.
* Cognee DataPoints and provenance form the unified writable knowledge layer.
* Lexical indexes, vector indexes, summaries, caches, and exports are rebuildable views.
* Graph relations and inferred knowledge must link back to source chunks.
* Query and ingestion modules must use the same canonical IDs and models.
* Markdown and Wiki exports are read-only views and must not become another writable knowledge source.

## Repository configuration

`config/paperos.toml` is the only structured configuration file. Secrets are
provided through `MINERU_API_KEY` and `DEEPSEEK_API_KEY` environment variables.
`.env` is not a second Cognee configuration source.

`config/paperos.toml` contains PaperOS configuration.

It defines:

* data directory;
* MinerU provider and endpoint;
* ingestion options;
* canonical processing options;
* chunking options;
* local model paths;
* lexical index path;
* retrieval profiles;
* API settings.

## Default data directory

The default data directory is:

```
~/paperos-knowledge-core/data
```

The final directory is resolved in this order:

```
PAPEROS_DATA_DIR
→ config/paperos.toml
→ ~/paperos-knowledge-core/data
```

All runtime data must remain under the resolved data directory.

## Data-directory layout

```
DATA_DIR/
├── raw/
│   └── <document_id>/
│       └── source.pdf
│
├── parsed/
│   └── <document_id>/
│       └── <parse_run_id>/
│           ├── manifest.json
│           ├── document.md
│           ├── content_list.json
│           ├── model_output.json
│           └── assets/
│
├── canonical/
│   └── <document_id>/
│       └── <schema_version>/
│           ├── manifest.json
│           ├── document.json
│           ├── sections.jsonl
│           ├── chunks.jsonl
│           ├── elements.jsonl
│           └── references.jsonl
│
├── cognee/
│   ├── system/
│   └── data/
│
├── indexes/
│   ├── lexical.sqlite
│   └── manifest.json
│
├── models/
│   ├── embedding/
│   ├── reranker/
│   └── query-expansion/
│
├── jobs/
├── cache/
├── exports/
├── logs/
├── test-corpus/
├── test-runs/
└── tmp/
```

## External services

### MinerU OCR

MinerU OCR is treated as an external parsing service.

The default configuration uses MinerU Cloud.

Supply the API key to the process environment:

```
export MINERU_API_KEY="your-token"
```

Secret values must not be emitted by status or configuration serialization.

A self-hosted MinerU-compatible endpoint may be configured in `paperos.toml`.

### DeepSeek

DeepSeek is the external LLM provider used by Cognee for:

* semantic extraction;
* entity and relation extraction;
* summaries;
* claims;
* query planning when required;
* final answer synthesis.

Non-secret DeepSeek configuration belongs in `config/paperos.toml`; its secret
is supplied as `DEEPSEEK_API_KEY`.

PaperOS does not start a local generative LLM service.

### Private local inference runtime

The PaperOS server owns a private Node child process that loads manually
provided GGUF files.

It provides:

* EmbeddingGemma embeddings;
* Qwen3 reranking;
* QMD query expansion.

Required files:

```
DATA_DIR/models/embedding/
└── embeddinggemma-300M-Q8_0.gguf

DATA_DIR/models/reranker/
└── qwen3-reranker-0.6b-q8_0.gguf

DATA_DIR/models/query-expansion/
└── qmd-query-expansion-1.7B-q4_k_m.gguf
```

The runtime must never download a missing model.

A missing file must produce an actionable startup error containing the expected path.

### Cognee local stores

The default local Cognee deployment uses:

* SQLite for relational and system metadata;
* LanceDB for vectors;
* Kuzu for graph storage.

All three stores reside under `DATA_DIR/cognee/`.

The configured dataset (or the dataset supplied to the HTTP ingest API) is materialized as an actual
Cognee Dataset. Each immutable source PDF is registered as a Cognee Data item,
and every structured write carries Cognee's default single-user principal,
Dataset, Data item, and PipelineRun in a `PipelineContext`. The official
Dataset and visualization APIs can therefore inspect the same nodes written by
PaperOS; manifests only record the binding for audit and rebuild.

Cognee/LanceDB is the sole semantic vector layer. Chunk text is embedded once
through `ChunkDataPoint.metadata.index_fields`; PaperOS does not create an
additional `indexes/vectors.sqlite3`. Semantic, Entity/Claim, Summary, and
Triplet lookup uses Cognee vectors, and graph retrieval performs typed multi-hop
traversal in Cognee before backtracking to canonical chunks. SQLite FTS5 remains
the exact lexical supplement.

No separate database server is required for the default configuration.

## Runtime environment

The project runs in one existing Conda environment:

```
paperos
```

Expected runtime versions:

```
Python 3.11
Node.js 22
```

PaperOS runs directly from source and is not installed as a wheel or editable
distribution.

## Initial configuration

From the repository root:

```
cp config/paperos.example.toml config/paperos.toml
```

Edit `config/paperos.toml` with:

* absolute data-directory path;
* MinerU and DeepSeek endpoints and models;
* local model paths;
* API port;
* retrieval settings.

## Start PaperOS

```bash
conda env create -f environment.yml
conda activate paperos
python scripts/setup_runtime.py
python server.py
```

MinerU and DeepSeek must already be available. Users place the three local
model files manually. The server initializes local storage and automatically
owns both the private Node inference child process and the background Worker.

Default endpoint:

```
http://127.0.0.1:8000
```

No authentication is required because the deployment is single-user. All
business operations use `/api/v1/*` HTTP endpoints; agents must not import
repositories or databases directly.

## Ingestion flow

```
PDF
→ validation and checksum
→ source registration
→ immutable raw PDF storage
→ MinerU OCR request
→ immutable raw parser artifacts
→ canonical normalization
→ cleaning and classification
→ section and element construction
→ deterministic chunking
→ reference processing
→ Cognee DataPoint mapping
→ graph and vector writes
→ lexical indexing
→ semantic enrichment
→ consistency validation
```

Every generated object must retain provenance to its source PDF, parse run, page, section, chunk, or element.

## Query flow

```
user question
→ query plan
→ lexical and semantic expansion
→ parallel retrieval
   ├── SQLite FTS
   ├── Cognee vectors
   ├── Cognee graph
   ├── global summaries
   └── confirmed knowledge
→ stable-ID candidate merge
→ reciprocal-rank fusion
→ evidence backtracking
→ local Qwen3 reranking
→ source diversification
→ DeepSeek answer synthesis
→ answer with evidence and provenance
```

The default query profile is `comprehensive`.

## Query profiles

### `comprehensive`

Runs all available retrieval channels and balances source evidence with graph and semantic context.

### `truth`

Prioritizes:

* original chunks;
* exact terminology;
* provenance;
* user-confirmed knowledge;
* direct source evidence.

### `associative`

Prioritizes:

* entities;
* graph relations;
* cross-document concepts;
* summaries;
* multi-hop associations.

All profiles use the same data and indexes. They differ only in retrieval budgets, graph depth, weighting, and output requirements.

## Testing

All functional tests use genuine academic PDF files supplied by the user.

Default corpus location:

```
DATA_DIR/test-corpus/
```

Expected layout:

```
test-corpus/
├── pdfs/
├── expected/
├── queries/
│   ├── truth.jsonl
│   ├── associative.jsonl
│   └── comprehensive.jsonl
├── manifest.json
└── checksums.sha256
```

Tests must not use:

* fake PDFs;
* synthetic PDFs;
* prerecorded MinerU responses;
* manually constructed canonical documents;
* manually seeded Cognee objects;
* precomputed embeddings;
* fixed reranker responses;
* fixed DeepSeek responses.

Development acceptance is cumulative:

```
PDF intake
PDF intake → MinerU
PDF intake → MinerU → canonical transformation
PDF intake → MinerU → canonical transformation → Cognee and indexes
complete ingestion → comprehensive query
```

A later acceptance test must execute every earlier part of the pipeline.

The detailed development order and gate requirements are defined in `IMPLEMENTATION_ORDER.md`.

## Schema evolution

Before the first stable release:

* backward compatibility is not required;
* canonical schemas may change;
* Cognee DataPoint definitions may change;
* stable-ID algorithms may be versioned;
* canonical snapshots may be rebuilt;
* local Cognee databases may be deleted and rebuilt;
* lexical and vector indexes may be deleted and rebuilt.

The following artifacts must always be preserved:

* original PDFs;
* source checksums;
* raw MinerU responses;
* parse-run manifests.

Provider-specific MinerU fields are isolated inside the MinerU adapter and canonical mapper.

Downstream modules consume versioned canonical models and must not depend directly on MinerU response fields.

## API surface

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
POST   /api/v1/improve
GET    /api/v1/health
GET    /api/v1/datasets
GET    /api/v1/datasets/{dataset_id}/data
GET    /api/v1/datasets/{dataset_id}/graph
GET    /api/v1/visualize?dataset_id={dataset_id}
```

## Completion criteria

The project is functionally complete only when a genuine academic PDF corpus successfully executes:

```
PDF ingestion
→ MinerU OCR
→ raw artifact persistence
→ canonical transformation
→ Cognee storage
→ lexical and vector indexing
→ comprehensive retrieval
→ local reranking
→ DeepSeek synthesis
→ source evidence and provenance
```

An isolated adapter, module, command, or unit test is not sufficient evidence of completion.
