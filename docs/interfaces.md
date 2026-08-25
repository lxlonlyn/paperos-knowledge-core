# Interfaces

## Public process interface

The only supported normal process entry is:

```bash
python server.py
```

The server binds `api.host` and `api.port`. All business operations use
`/api/v1/*`.

## HTTP routes

### Ingest

`POST /api/v1/ingest` accepts multipart field `file` and optional `dataset`.
The upload is staged under the configured data directory, an operational job is
created, and the route returns HTTP 202:

```json
{"job_id": "opjob_...", "status": "pending"}
```

The Worker validates the real PDF, creates SourceFile and IngestionJob records,
calls MinerU, persists parser artifacts and canonical data, and writes Cognee and
FTS projections. Staged bytes are removed after the job finishes.

### Job status

`GET /api/v1/jobs/{job_id}` returns the operational job, payload, status,
error, and result. Status is one of `pending`, `running`, `completed`, or
`failed`.

### Query

`POST /api/v1/query` accepts the shared `QueryRequest` domain model and returns
`QueryResponse` with answer, candidates, canonical evidence, channel usage,
provenance, the retrieval trace, and `replay`. `replay.original_query` preserves
the caller's exact question, while `replay.replay_text` is the same standalone
Markdown user prompt sent to final synthesis. A caller can paste that text into
a new web LLM conversation without rerunning retrieval.

### Documents and maintenance

```text
GET    /api/v1/documents
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
POST   /api/v1/documents/{document_id}/reprocess  -> 202
POST   /api/v1/rebuild                            -> 202
```

Logical delete removes active derived projections while retaining immutable
evidence. Reprocess and rebuild execute through the internal queue.

### Feedback and improvement

`POST /api/v1/feedback` records confirmation, rejection, or correction against
canonical evidence. `POST /api/v1/improve` enqueues versioned derived
improvement and returns HTTP 202.

Feedback never modifies source PDFs, raw MinerU artifacts, or original canonical
chunks.

### Health

`GET /api/v1/health` reports application and dependency health. MinerU or
LLM provider failure degrades health. Health is read-only and cannot start a
process, download a model, or initialize an external service. Public health,
status, document, parse, and indexing payloads expose neither credentials nor
resolved filesystem paths.

## External provider interfaces

The MinerU adapter owns submission, task polling, finite timeout/retry handling,
result retrieval, and provider-specific response parsing. No MinerU field passes
the canonical boundary.

`CogneeConfigurator` applies the single TOML's Cognee settings before any
engine or gateway is created. `LLMClient` then reads credential-free
provider/model metadata through `CogneeRuntimeConfigReader` and provides
semantic enrichment and evidence-bound answer synthesis exclusively through
Cognee's `LLMGateway`. It never starts the provider and never knows the vendor.

## Private local inference protocol

The Application-owned Node child listens only at the configured loopback
implementation address. `LocalInferenceClient` uses:

```text
GET  /health
GET  /v1/models
POST /v1/embeddings
POST /v1/rerank
```

No public command or remote-provider mode exposes this protocol. Missing models,
checksum mismatch, missing Node build output, port conflicts, early child exit,
and readiness timeout are startup errors.

## Repository contracts

- repositories accept paths or connections and never initialize global schema;
- `DataPathCodec` alone maps retained relative POSIX references to runtime paths;
- `StorageInitializer` owns SQLite/FTS creation and validation;
- canonical models are the only downstream document contract;
- Cognee DataPoint and relation types are centralized;
- Cognee, vector, graph, and FTS projections use canonical stable IDs;
- every graph or semantic result can backtrack to source chunks;
- prompt metadata accompanies semantic enrichment manifests.

Agents and external programs must not import repositories. They use HTTP or
`scripts/agent_client.py`.
