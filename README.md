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

The FastAPI lifespan initializes local schemas, starts the private Node inference
child process, waits for its three local models, starts one background Worker,
and then accepts HTTP requests. Shutdown stops the Worker, terminates the child
process, and closes Cognee and HTTP clients.

MinerU and DeepSeek are external dependencies. PaperOS checks and calls them but
never starts them. Cognee, SQLite, LanceDB, Kuzu, and FTS are internal libraries
or stores rather than separately deployed PaperOS services.

## Setup

```bash
conda env create -f environment.yml
conda activate paperos

cp config/paperos.example.toml config/paperos.toml
export MINERU_API_KEY="..."
export DEEPSEEK_API_KEY="..."

python scripts/setup_runtime.py
python server.py
```

MinerU and DeepSeek must already be reachable. Users manually place all three
GGUF files at the configured paths; relative model paths are resolved from
`config/paperos.toml`, independently of the runtime data directory. PaperOS
never installs dependencies or downloads models at runtime.

`config/paperos.toml` is the only structured configuration source. The two API
keys are environment-only secrets. PaperOS does not load `.env` as a Cognee
configuration file.

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
implementation port. It loads manually supplied EmbeddingGemma, Qwen3 Reranker,
and QMD Query Expansion GGUF files. Application services receive only a
`LocalInferenceClient`; only the Application lifecycle can start or stop the
child process.

## Operational scripts

- `scripts/setup_runtime.py`: initializes schema and verifies Node/build/models.
- `scripts/doctor.py`: read-only configuration and dependency diagnostics.
- `scripts/debug_pipeline.py`: real retained-stage pipeline debugging.
- `scripts/agent_client.py`: HTTP client example for agents and integrations.

See [docs/architecture.md](docs/architecture.md),
[docs/data_model.md](docs/data_model.md), and
[docs/interfaces.md](docs/interfaces.md) for the binding internal contracts.
