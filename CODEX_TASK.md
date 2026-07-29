# Codex Execution Contract

## Scope

Codex must:

1. Implement the Python and TypeScript source files in this repository.
2. Implement and run cumulative integration and evaluation tests using genuine academic PDF files.
3. Use the existing activated environment and already-installed dependencies.
4. Call MinerU OCR, Cognee, DeepSeek, and local model services only through repository adapters.
5. Implement the project strictly in the stage order defined by `IMPLEMENTATION_ORDER.md`.
6. Complete and verify the current stage before writing code for the next stage.
7. Test every stage cumulatively from the original PDF input.
8. Preserve the same IDs, schemas, provenance, and artifacts across stage boundaries.
9. Continue automatically to the next stage only after the current cumulative gate passes.
10. Stop and report an actionable blocker when a required real service, model, configuration, or PDF is unavailable.

## Stage-gating rules

Codex must follow these rules:

1. Stage 2 code must not be written before Stage 1 passes.
2. Stage 3 code must not be written before the complete Stage 1 → Stage 2 path passes.
3. Stage 4 code must not be written before the complete Stage 1 → Stage 2 → Stage 3 path passes.
4. Query code must not be written before the complete Stage 1 → Stage 2 → Stage 3 → Stage 4 ingestion path passes.
5. A later stage must consume artifacts produced by the immediately preceding real stage.
6. A later stage must never construct its own substitute input.
7. Tests for Stage N must begin from the genuine PDF and execute every stage from Stage 1 through Stage N.
8. Passing an isolated module test does not satisfy a stage gate.
9. Codex must not implement future stages in parallel while an earlier stage is incomplete.
10. When a cumulative test fails, Codex must repair the earliest incorrect stage or interface rather than bypassing it downstream.

## Real-input test rules

Codex must:

- use genuine academic PDFs from the configured test corpus;
- submit those PDFs to the configured live MinerU OCR service;
- consume the actual MinerU response produced during the test run;
- use the configured local Cognee stores;
- load the configured local embedding, reranking, and query-expansion models;
- call the configured DeepSeek provider when the tested stage requires LLM processing;
- keep all test-run artifacts under the configured PaperOS data directory.

Codex must not:

- use fake, mock, stub, synthetic, generated, simplified, or reconstructed PDF inputs;
- use prerecorded, manually written, or precomputed MinerU responses;
- seed a later-stage database with manually constructed objects;
- replace unavailable services with test doubles;
- hard-code expected parser output, chunks, embeddings, graph objects, or answers;
- skip a cumulative stage because its isolated downstream code already passes.

## Prohibited operations

Codex must not:

- create, remove, clone, or modify Conda environments;
- run `pip install`, `uv pip install`, `conda install`, `npm install`, or equivalent installers;
- download models, datasets, databases, OCR packages, PDFs, or external repositories;
- modify the user's real `.env`;
- modify user-provided test PDFs;
- add automatic model downloads or implicit Hugging Face downloads;
- start unmanaged long-running external services;
- introduce authentication or multi-user access control;
- introduce a second authoritative knowledge store;
- preserve backward compatibility at the cost of blocking necessary development changes before the first stable release.

## Runtime assumptions

- Conda environment name: `paperos`.
- Python 3.11 and Node.js 22 are already installed.
- Required Python and Node.js dependencies are already installed.
- The project is installed in editable mode.
- `.env` and `config/paperos.toml` are created by the user.
- Genuine academic PDFs are located under the configured test corpus directory.
- MinerU OCR is reachable through the configured adapter.
- DeepSeek is configured as Cognee's external LLM provider.
- Local GGUF model files exist under the configured data directory.
- Cognee uses local SQLite, LanceDB, and Kuzu under the configured data directory.
- Codex may start repository-owned processes only as bounded test subprocesses.
- Codex must not access public download services.

## Implementation standard

- Python 3.11.
- Node.js 22.
- Async I/O for external APIs.
- Pydantic v2 models.
- Stable deterministic IDs within a declared schema version.
- Single-writer ingestion and index updates.
- Explicit provenance on every inferred object or relation.
- No hidden fallback to paid APIs.
- Missing local files or endpoints must fail with actionable errors.
- Raw PDFs and raw MinerU outputs are immutable.
- Canonical data, Cognee data, lexical indexes, vector indexes, and derived knowledge may be destructively rebuilt during development.
- Provider-specific MinerU fields must be isolated inside the MinerU adapter and mapper.
- Downstream modules must consume canonical models rather than raw MinerU payloads.
- Shared models and interfaces must be defined centrally and reused by every stage.
- Premature compatibility layers and duplicated schemas are prohibited.