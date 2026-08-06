# Architecture

## Binding decisions for 0.1.0

### A1: Product form

PaperOS is a source-deployed application, not a PyPI distribution.
`paperos_core` is an internal importable package. Wheels, editable installs,
console scripts, and compatibility import facades are not product interfaces.

### A2: Single normal entry point

The only normal startup command is:

```bash
python server.py
```

There are no `paperos serve`, `paperos model-gateway`, `paperos worker`, or
`paperos init` commands, including hidden or deprecated aliases.

### A3: External services

PaperOS configures, checks, and calls MinerU and the LLM provider selected by
Cognee configuration. It never starts or manages them. PaperOS business code
does not know the LLM vendor: all model calls go through Cognee's LLMGateway.

### A4: Internal libraries and stores

Cognee's Python library, SQLite, LanceDB, Kuzu, and FTS are initialized and
closed as application-owned dependencies. They are not independently deployed
PaperOS services.

### A5: Private local inference runtime

The Node embedding and optional-reranking runtime is a private child process.
It starts when the Cognee embedding configuration selects the PaperOS local
runtime or when the local reranker is enabled; with a fully remote
configuration, PaperOS never checks or loads local GGUF files. Embedding
provider/model/dimensions/token limits live in ``[cognee.embedding]``, while
``[local_inference]`` owns file paths, Node parameters, and the loopback port.
The Application lifecycle starts it, waits for readiness, and stops it. Its
loopback HTTP port is an implementation detail.

### A6: Worker ownership

One background `asyncio` Worker task runs inside the server process. It has one
consumer and no independent command.

### A7: Model files

Users provide GGUF files manually. PaperOS never downloads models at runtime or
as an installation side effect.

## Product boundary

```text
Agent / AstrBot / HTTP client
             |
             v
       FastAPI routers
             |
             v
     ApplicationServices
  ingestion | retrieval | documents | feedback | health
             |
        canonical IDs
             |
   +---------+----------+
   |                    |
Cognee graph/vector   SQLite FTS
   |
canonical provenance
```

MinerU and the configured LLM provider remain outside the process boundary. The
private Node child process and Worker are owned by `Application`.

## Application lifecycle

`create_application(settings)` performs object assembly only. FastAPI lifespan
constructs it once and invokes:

```text
Application.start
  1. initialize and validate local schema
  2. start private local inference
  3. wait for model readiness
  4. start the background Worker

Application.aclose
  1. stop the Worker
  2. dispose Cognee relational, graph, vector, and cache engines
  3. stop local inference
  4. close inference and MinerU clients
```

Health checks are read-only. They never start or restart resources. A local model
file, Node entry, occupied implementation port, early child exit, or readiness
timeout fails startup with an actionable error. External MinerU or LLM provider
failure is reported as degraded health and is never treated as authority to
launch those providers.

## Dependency assembly

`ApplicationServices` contains the business services. `ManagedRuntime`
contains `LocalInferenceRuntime` and `BackgroundWorker`. Services receive
`LocalInferenceClient`, not the runtime.

The Worker receives only its queue and the services required to execute ingest,
rebuild, reprocess, and improve jobs. It never holds the Application.

## Configuration ownership

`config/paperos.toml` is the only structured configuration. It owns data,
MinerU, the provider-neutral `[llm]` section, local inference, Cognee,
ingestion, retrieval, and API settings. `MINERU_API_KEY` and `LLM_API_KEY` are
environment-only secrets. Relative GGUF paths are resolved from the TOML
directory rather than the mutable runtime data directory.

Only `configure_cognee(RuntimeSettings)` translates PaperOS settings into the
environment variables required by Cognee. Cognee does not read a separate
project configuration source. `LLMClient` receives `LLMSettings` directly and
reaches the provider only through `cognee.infrastructure.llm.LLMGateway`;
switching LLM, embedding, vector, or graph providers is configuration-only.

## Storage and schema

`StorageInitializer` is the sole owner of PaperOS SQLite and FTS schema
creation. Application startup and `scripts/setup_runtime.py` use the same
initializer. Repository constructors only retain paths or connections.

All runtime data stays under `data.directory`:

- `raw/`: immutable source PDFs and SourceFile identity;
- `parsed/`: immutable ParseRun artifacts;
- `canonical/`: versioned canonical snapshots;
- `cognee/`: graph, vector, metadata, enrichment, and manifests;
- `indexes/`: SQLite FTS and index manifests;
- `jobs/`: registries, queue, and managed-process records;
- `cache/`, `logs/`, and `tmp/`: rebuildable or managed runtime data.

Derived stores can be destructively rebuilt from retained real artifacts.
Original PDFs, raw MinerU results, and source chunks are never overwritten by
feedback or rebuild.

## Knowledge model and retrieval

MinerU-specific fields exist only in the MinerU adapter and canonical mapper.
Downstream code consumes versioned canonical models. Stable IDs include an
explicit ID version and are shared by Cognee, graph edges, vectors, FTS rows,
evidence, and API responses.

The ingestion chain is a Cognee custom pipeline:

```text
MinerU -> Canonical Document / Section / Element / Reference
       -> AcademicChunkTask -> SemanticEnrichmentTask
       -> DataPointMappingTask -> add_data_points
```

PaperOS decides the academic chunking rules; Cognee executes the pipeline and
provides the tokenizer, token limits, and the final DataPoint write. Cognee's
public pipeline auto-creates the dataset and logs pipeline runs. Stable
data-item provenance (the relational Data row and DatasetData association
that ``add_data_points`` attributes nodes and edges to) is not auto-established
for custom canonical input: the provenance spike contract proves it, and the
minimal private registration stays centralized in
``paperos_core/adapters/cognee/compat.py``.

Retrieval calls Cognee's public search/recall surface:

```text
truth        -> FTS5 + GRAPH_COMPLETION restricted to ChunkDataPoint
associative  -> GRAPH_COMPLETION_DECOMPOSITION + Entity / Claim + typed graph
comprehensive-> FTS5 + GRAPH_COMPLETION + Cognee recall context + fusion
```

Each profile maps to a real Cognee SearchType, hits are constrained by the
returned node type, and candidates backtrack through canonical IDs / node IDs
/ source references (never by text-prefix matching). Absent vector distances
score 0.0 explicitly. PaperOS does not generate query embeddings, open vector
collections, or retain a duplicate embedding BLOB store. Private Cognee API
calls are centralized in `paperos_core/adapters/cognee/compat.py` and pinned
to cognee 1.4.0 by contract tests.

## Prompt ownership

Markdown files under `prompts/` are the only complete prompt source.
`PromptRepository` validates prompt names and records version and SHA-256.
Semantic enrichment manifests contain prompt name, prompt version, prompt
SHA-256, model, and model version.

## API organization

Each business module owns a real `APIRouter`. `api/app.py` only creates
FastAPI, owns lifespan and exception handling, and includes routers. Business
operations are HTTP-only. Long-running ingest, reprocess, rebuild, and improve
operations enter the internal queue and expose status through
`GET /api/v1/jobs/{job_id}`.

PaperOS is intentionally single-user. Authentication, multi-user authorization,
and a second authoritative knowledge store are outside scope.
