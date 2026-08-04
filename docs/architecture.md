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

PaperOS configures, checks, and calls MinerU, DeepSeek, and future remote
providers. It never starts or manages them.

### A4: Internal libraries and stores

Cognee's Python library, SQLite, LanceDB, Kuzu, and FTS are initialized and
closed as application-owned dependencies. They are not independently deployed
PaperOS services.

### A5: Private local inference runtime

The Node embedding, reranking, and query-expansion runtime is a private child
process. The Application lifecycle starts it, waits for readiness, and stops it.
Its loopback HTTP port is an implementation detail.

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

MinerU and DeepSeek remain outside the process boundary. The private Node child
process and Worker are owned by `Application`.

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
  2. stop local inference
  3. close inference, DeepSeek, and MinerU clients
```

Health checks are read-only. They never start or restart resources. A local model
file, Node entry, occupied implementation port, early child exit, or readiness
timeout fails startup with an actionable error. External MinerU or DeepSeek
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
MinerU, DeepSeek, local inference, Cognee, ingestion, retrieval, and API
settings. `MINERU_API_KEY` and `DEEPSEEK_API_KEY` are environment-only
secrets.

Only `configure_cognee(CogneeSettings)` translates PaperOS settings into the
environment variables required by Cognee. Cognee does not read a separate
project configuration source. `DeepSeekClient` receives
`DeepSeekSettings` directly.

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

Cognee is the structural and semantic retrieval layer:

```text
Cognee vectors -> Chunk / Entity / Claim / Summary / Triplet lookup
Cognee graph   -> typed traversal -> edge provenance -> source chunks
SQLite FTS     -> exact lexical supplement
```

PaperOS does not retain a duplicate embedding BLOB store. Every inferred object
and relation carries source chunk IDs. Dataset, Data item, User, and PipelineRun
context is propagated through Cognee writes.

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
