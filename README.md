# PaperOS Knowledge Core

PaperOS Knowledge Core is a source-deployed, single-user research-paper knowledge
backend. It ingests genuine PDFs through MinerU, preserves immutable parser
artifacts, builds versioned canonical documents, writes structured knowledge and
vectors through Cognee, maintains SQLite FTS, and returns evidence-bound answers.

## Runtime shape

One command owns the complete application lifecycle:

```bash
python server.py
```

The FastAPI lifespan initializes local schemas, starts the private Node
inference child process when the Cognee embedding configuration selects the
PaperOS local runtime (embedding model, optional reranker), starts one
background Worker, and then accepts HTTP requests. With a remote embedding
provider, local GGUF files are never checked or loaded. Shutdown stops the
Worker, asks the child to exit through its authenticated private HTTP protocol,
and closes Cognee and HTTP clients. OS process signals are used only as a
timeout/error fallback.

MinerU and the LLM provider selected by Cognee configuration are external
dependencies. PaperOS checks and calls them but never starts them, and its code
never knows the LLM vendor. Cognee, SQLite, LanceDB, Kuzu, and FTS are internal
libraries or stores rather than separately deployed PaperOS services.

## Setup

```bash
conda env create -f environment.yml
conda activate paperos

python scripts/init_config.py
# Set mineru.api_key in the git-ignored config/paperos.toml.
# Set Cognee LLM/embedding/database values in .env.

cd services/local_models
npm ci
npm run build
cd ../..

python scripts/setup_runtime.py
python server.py
```

MinerU and the configured LLM provider must already be reachable. Users
manually place the local embedding (and optional reranker) GGUF files at the
configured paths; relative model paths are resolved from
`config/paperos.toml`, independently of the shell working directory and runtime
data directory. All relative filesystem paths in the TOML use that same base.
PaperOS never installs dependencies or downloads models at runtime.

`config/paperos.toml` contains only PaperOS settings. Cognee exclusively owns
and loads `.env`; PaperOS never parses or overwrites it. MinerU's key is a
redacted `SecretStr` in the git-ignored real TOML.

## HTTP API

Normal business operations use HTTP only:

```text
POST   /api/v1/ingest
GET    /api/v1/jobs/{job_id}
POST   /api/v1/query
GET    /api/v1/documents
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST   /api/v1/documents/{document_id}/reprocess
POST   /api/v1/feedback
POST   /api/v1/improve
POST   /api/v1/rebuild
GET    /api/v1/health
GET    /api/v1/visualize
```

Ingest, reprocess, improve, and rebuild enqueue work and return HTTP 202 with a
job ID. Agents should use [scripts/agent_client.py](scripts/agent_client.py)
rather than importing repositories or database files.

## Data ownership

All runtime state is beneath `data.directory`:

```text
raw/        immutable source PDFs
parsed/     immutable MinerU responses and artifacts
canonical/  versioned canonical snapshots
cognee/     Cognee system, graph, vector, metadata, and enrichment data
indexes/    rebuildable FTS and index manifests
jobs/       registry, operational queue, and process records
cache/      rebuildable query cache
logs/       application and local-inference logs
tmp/        managed upload staging
```

Original PDFs and raw MinerU responses are immutable. Canonical snapshots are
versioned. Cognee and FTS projections, enrichment, summaries, caches, and exports
are rebuildable. Every derived object and graph relation retains canonical chunk
provenance.

Cognee/LanceDB is the only semantic vector layer. SQLite FTS5 is the lexical
supplement. Both use the same canonical IDs.

## Local inference

The repository-owned Node runtime is private to PaperOS and listens on a loopback
implementation port. It starts only when the Cognee embedding configuration
selects the PaperOS local runtime, and/or local reranking is enabled. It loads
only the GGUF models actually used. A remote embedding plus local reranker
therefore starts the child with only the reranker; fully remote retrieval skips
Node and all GGUF checks. Application services receive only a `LocalInferenceClient`; only the
Application lifecycle can start or stop the child process.

## Operational scripts

- `scripts/setup_runtime.py`: initializes schema and verifies Node/build/models.
- `scripts/doctor.py`: read-only configuration and dependency diagnostics.
- `scripts/acceptance_real_pipeline.py`: cumulative validation using the genuine
  four-paper corpus, live MinerU/Cognee providers, all retrieval profiles, and
  lifecycle cleanup. This is the only project test entry; the project does not
  use pytest, mocks, fabricated documents, or prerecorded downstream results.
- `scripts/debug_pipeline.py`: real retained-stage pipeline debugging.
- `scripts/agent_client.py`: HTTP client example for agents and integrations.

Run the complete acceptance path with:

```bash
python scripts/acceptance_real_pipeline.py
```

See [docs/architecture.md](docs/architecture.md),
[docs/data_model.md](docs/data_model.md), and
[docs/interfaces.md](docs/interfaces.md) for the binding internal contracts.
