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
# Set MinerU, Cognee LLM, embedding, and storage values in the
# git-ignored config/paperos.toml.

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

`config/paperos.toml` is the only persistent configuration file. PaperOS applies
its Cognee tables through Cognee's public runtime configuration API before any
engine or gateway is created; no `.env` is loaded or written. Provider keys are
redacted `SecretStr` values and are excluded from status, health, and logs.

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

The default root is the repository's `data/` directory. A relative
`data.directory` is resolved from the TOML directory, not the current working
directory. SQLite and long-lived JSON store only data-root-relative POSIX
references; repositories decode them to absolute `Path` objects at runtime.
Copying the project or `data/` therefore does not rewrite retained artifacts.

Original PDFs and raw MinerU responses are immutable. Canonical snapshots are
versioned. Cognee and FTS projections, enrichment, summaries, caches, and exports
are rebuildable. Every derived object and graph relation retains canonical chunk
provenance.

`registry.db` also owns stable `ScholarlyWork` identities independently of
parsed Documents. External cited works may exist without a PDF. Cognee projects
these identities without fake canonical provenance, and rebuild reproduces
Document-to-Work, Reference-to-Work, and Work-to-Work CITES edges with the same
permanent Work IDs.

Cognee/LanceDB is the only semantic vector layer and SQLite FTS5 is the lexical
supplement. Production query discovery searches only canonical
``ChunkDataPoint`` objects in ``PAPEROS_CHUNKS``. Query text never selects an
Entity, Claim, Summary, or Graph search channel. Explicit context expansion
starts from first-stage Chunk hits and returns only canonical source Chunks.
Version-locked graph/vector access remains isolated in
``paperos_core/adapters/cognee/compat.py``; PaperOS does not create a second
vector index, embedding client, or graph store.

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
- `scripts/migrate_portable_paths.py`: dry-run/transactional migration of legacy
  absolute SQLite and JSON references.
- `tests/validation/retrieval.py`: real-PDF Chunk-first validation using live
  MinerU/Cognee/LLM providers, with JSON and Markdown review artifacts.
- `tests/contract/test_portable_data_paths.py`: permanent portable-path and real
  retained-data relocation contract, run directly without pytest.
- `tests/contract/test_cognee_retrieval_boundary.py`: permanent static/live
  Cognee search, dataset-scope, and provenance boundary contract.
- `tests/contract/test_scholarly_identity.py`: fast retained-canonical contract
  for repeated Work resolution, reconciliation, redirects, external provenance,
  and citation-backbone reprojection; it is not a PDF reprocess test.
- `tests/validation/scholarly_work_reference_acceptance.py`: resumable four-PDF
  acceptance for provisional-to-ingested identity reuse, real reprocess, formal
  rebuild, and live Cognee citation-backbone readback.
- `scripts/backfill_scholarly_works.py`: deterministic registry backfill from
  retained canonical snapshots without invoking MinerU, LLM, or Cognee rebuild.
- `scripts/debug_pipeline.py`: real retained-stage pipeline debugging.
- `scripts/agent_client.py`: HTTP client example for agents and integrations.

PaperOS does not use pytest, mocks, fabricated papers, or prerecorded downstream
results. Acceptance exercises behavior; permanent contracts protect boundaries.

Run the complete acceptance path with the authoritative validation corpus:

```bash
python tests/validation/retrieval.py --rebuild
```
See [docs/architecture.md](docs/architecture.md),
[docs/data_model.md](docs/data_model.md), and
[docs/interfaces.md](docs/interfaces.md) for the binding internal contracts.
