# Architecture

## Binding decisions for 1.0.0

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

PaperOS configures and calls MinerU, and checks/calls the LLM provider selected
by Cognee configuration. It never starts or manages external providers. PaperOS business code
does not know the LLM vendor: all model calls go through Cognee's LLMGateway.

### A4: Internal libraries and stores

Cognee's Python library, SQLite, LanceDB, Kuzu, and FTS are initialized and
closed as application-owned dependencies. They are not independently deployed
PaperOS services.

### A5: Private local inference runtime

The Node embedding and optional-reranking runtime is a private child process.
It starts only when `[local_inference].enabled=true` and the Cognee embedding
endpoint selects its loopback port or the local reranker is enabled; with a fully remote
configuration, PaperOS never checks or loads local GGUF files. Embedding
provider/model/dimensions/token limits live in ``[cognee.embedding]``, while
``[local_inference]`` owns enablement, GGUF paths, timeouts, the loopback port,
and the CUDA device allowlist. PaperOS passes that allowlist to its Node child
through ``CUDA_VISIBLE_DEVICES``; the child cannot allocate on other GPUs.
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

`config/paperos.toml` is the only persistent configuration. It owns PaperOS and
Cognee runtime settings; `.env` and `.env.example` do not exist. Provider keys
are Pydantic `SecretStr` values excluded from serialization. Three optional
`PAPEROS_*_API_KEY` variables override only secret fields, not general settings.

`CogneeConfigurator` applies LLM, embedding, relational, vector, and graph
settings through Cognee's public runtime API before any Cognee engine, gateway,
or pipeline is created. It never persists Cognee configuration. The
credential-free `CogneeRuntimeConfigReader` reads the applied state for health,
doctor, local-runtime activation, and provider/model metadata.

## Storage and schema

`StorageInitializer` is the sole owner of PaperOS SQLite and FTS schema
creation. Application startup and `scripts/setup_runtime.py` use the same
initializer. Repository constructors only retain paths or connections.
The registry and lexical databases use SQLite `PRAGMA user_version = 1`.
Existing PaperOS tables without that version and versions newer than this
release fail closed; there is no pre-1.0 migration.

`storage.registry_filename`, `storage.lexical_filename`, and
`cognee.storage.database_name` are store identities, restricted to safe names
under their assigned data directories. Changing an identity after first use
selects the newly named store; PaperOS does not locate, migrate, copy, rename,
merge, or delete the previous store.

All runtime data stays under `data.directory`:

- `raw/`: immutable source PDFs and SourceFile identity;
- `parsed/`: immutable ParseRun artifacts;
- `canonical/`: versioned canonical snapshots;
- `cognee/system`, `cognee/data`, `cognee/vector`, and `cognee/graph`: Cognee
  runtime stores derived from the data root rather than user-supplied paths;
- `cognee/enrichment` and `cognee/manifests`: PaperOS projection metadata;
- `indexes/`: SQLite FTS and index manifests;
- `jobs/`: registries, queue, and managed-process records;
- `cache/`, `logs/`, and `tmp/`: rebuildable or managed runtime data.

The default data root is `<repository_root>/data`. Relative TOML paths resolve
from the TOML directory. `DataPathCodec` is the sole persistent-path boundary:
SQLite and long-lived JSON store data-root-relative POSIX references, while
repositories decode those references to runtime absolute `Path` objects. Domain
models may therefore use absolute runtime paths without making artifacts
machine-specific. Public API responses and health omit filesystem paths.

PID/process records are machine-local operational state, not portable data.
Application startup removes stale local-inference and Worker records and the
runtime recreates them as needed; records never establish retained identity.

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
       -> AcademicChunkTask -> ScholarlyIdentityTask
       -> SemanticEnrichmentTask -> DataPointMappingTask
       -> add_data_points
```

Inside `AcademicChunkTask`, source structure is resolved before citation
resolution:

```text
MinerU source item/domain/offset
-> Canonical Element + SourceSpan
-> ordered DocumentRegion state machine
-> CitationNamespace assignment
-> inline-domain segmentation and CitationCandidate detection
-> atomic citation resolution in the preassigned namespace
-> ChunkProjection and citation-to-chunk span attachment
```

One actual `REFERENCES` region instance creates exactly one
`CitationNamespace`; gaps between Reference elements never create namespace
boundaries. After the complete region stream exists, each body region binds to
the nearest following bibliography, or to the nearest preceding bibliography
when none follows. This gives main and supplement independent namespaces when
both have bibliographies, while an appendix after the main bibliography
inherits that bibliography. The resolver does not retry another namespace: a
missing binding is retained as `NAMESPACE_NOT_ASSIGNED`.

Reference regions feed `ReferenceEntry` and citation resolution only. They are
excluded from ordinary knowledge chunks and citation scanning. Appendix and
supplement bodies, tables, figure/table captions, and footnotes remain eligible
source evidence. Inline/display math is segmented before bracket scanning, so
brackets inside math are not prose citation candidates.

`Chunk.text` contains only authoritative source-derived evidence. Rebuildable
context such as paper title, section breadcrumb, and resolved referenced-work
identity belongs in `retrieval_text`, which is the lexical/vector projection.

PaperOS decides the academic chunking and scholarly identity rules. Its
authoritative hard-max estimate is deterministic UTF-8 byte length, a
conservative upper bound that does not depend on Cognee's model resolver or
optional tokenizer packages. Cognee executes the pipeline and owns the final
DataPoint write. A permanent random Work ID is allocated once in
`registry.db`; DOI, arXiv, and normalized bibliographic fields are reconciliation
attributes rather than ID material. Document and Reference links plus merge
redirects remain authoritative across reprocess and Cognee rebuild. Cognee's
`ScholarlyWorkDataPoint` is only a projection and external Works never receive
fabricated source-file, parse-run, snapshot, or chunk provenance.

The citation backbone is `Document --REPRESENTS_WORK--> ScholarlyWork`,
`ReferenceEntry --RESOLVES_TO--> ScholarlyWork`, and
`ScholarlyWork --CITES--> ScholarlyWork`. Each CITES edge derives from its
ReferenceEntry and carries the body Chunk IDs whose resolved citation mentions
actually target that ReferenceEntry. Reference-list paragraphs are not used as
CITES evidence. Chunks are
formally split from the canonical snapshot: canonical artifacts carry
document/sections/elements/references only, and the derived
``ChunkProjection`` is produced by the pipeline and rebuilt on demand. Cognee's
public pipeline auto-creates the dataset and logs pipeline runs. Stable
data-item provenance (the relational Data row and DatasetData association
that ``add_data_points`` attributes nodes and edges to) is not auto-established
for custom canonical input: the provenance spike contract proves it, and the
minimal private registration stays centralized in
``paperos_core/adapters/cognee/compat.py``.

Production retrieval has one architecture for every natural-language query:

```text
Query
-> FTS5 Chunk retrieval + PAPEROS_CHUNKS vector retrieval
-> reciprocal-rank fusion
-> chunk_id deduplication
-> rerank
-> top source Chunk seeds
-> optional local context expansion
-> optional direct semantic relation expansion
-> chunk_id deduplication and a second rerank when new Chunks were added
-> whole-Chunk synthesis budget selection
-> final canonical source-grounded Evidence
-> FinalSynthesisContext
-> one rendered Markdown synthesis prompt
   |-> LLM synthesis -> answer
   `-> QueryReplay.replay_text
```

Only caller-provided document/work IDs are hard filters. There is no QueryScope
planner, title routing, task classification, comparison/limitation classifier,
or profile-selected retrieval world. Local expansion is restricted to ±1 Chunk
inside the same document region and major section. Direct semantic expansion
follows only ``Seed Chunk -> semantic objects grounded in that Chunk -> one
direct semantic relation -> relation.source_chunk_ids -> canonical Chunk
candidates``; the query itself never searches graph nodes. CITES is
scholarly/provenance infrastructure and is not an ordinary semantic Search
expansion path. Derived text can aid discovery/provenance but never becomes paper
evidence. Evidence is always rehydrated from the current ChunkProjection.
The final synthesis renderer preserves the caller's original query, Evidence
ordering, canonical `Chunk.text`, and available paper provenance. Its rendered
prompt is both the exact LLM user input and the production Query Replay; Replay
does not rerender, persist queries, or trigger another search or model call.
Before rendering, `retrieval.synthesis_max_input_tokens` selects the longest
ranked Evidence prefix whose complete Markdown prompt fits the deterministic
token estimate. Selected Chunks remain complete; source text is never substring
truncated.

Claim enrichment is optional and disabled by default. The disabled path uses a
prompt and response schema with no Claim output field, creates no ClaimDataPoint
or ABOUT edge, and therefore avoids Claim-generation LLM work. Enabling Claims
does not add a Query-to-Claim search channel.

## Prompt ownership

Versioned provider-level and enrichment prompts live under `prompts/`;
`PromptRepository` validates their names and records version and SHA-256.
`render_synthesis_prompt()` is the single owner of the query-dependent final
synthesis user prompt shared with Query Replay. Semantic enrichment manifests
contain coverage IDs/ratio, prompt name/version/SHA-256, and Cognee's actual
provider/model.

## API organization

Each business module owns a real `APIRouter`. `api/app.py` only creates
FastAPI, owns lifespan and exception handling, and includes routers. Business
operations are HTTP-only. Long-running ingest, reprocess, rebuild, and improve
operations enter the internal queue and expose status through
`GET /api/v1/jobs/{job_id}`.

PaperOS is intentionally single-user. Authentication, multi-user authorization,
and a second authoritative knowledge store are outside scope.
