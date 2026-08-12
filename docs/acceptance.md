# Real-case acceptance

PaperOS validation uses only genuine academic PDFs supplied under
`DATA_DIR/validation/corpus/pdfs`. The project does not use pytest, mocks, stubs,
synthetic documents, reconstructed parser output, precomputed embeddings,
fixed reranking output, or fixed LLM responses.

`scripts/acceptance_real_pipeline.py` is the cumulative real-chain acceptance
entry. Permanent boundary contracts are separate direct-run scripts. Acceptance
verifies each PDF checksum and executes:

```text
genuine PDF corpus
→ source registration
→ live MinerU parsing and immutable raw artifacts
→ canonical Document / Section / Element / Reference
→ academic ChunkProjection
→ live Cognee LLM semantic enrichment
→ DataPoint graph and provenance
→ Cognee graph/vector persistence
→ SQLite FTS5 projection
→ truth, associative, and comprehensive retrieval
→ evidence-bound answers with canonical citations
→ health and child-process cleanup
```

Expected structural properties live in `DATA_DIR/validation/corpus/expected`.
Research questions and evidence requirements live in
`DATA_DIR/validation/corpus/queries`. Run artifacts are written beneath
`DATA_DIR/validation/runs/<timestamp>` unless `--run-root` is supplied. The
corpus and run outputs therefore share one validation root; obsolete run
directories can be removed without touching production data or the corpus.

Run:

```bash
python scripts/doctor.py
python scripts/acceptance_real_pipeline.py
```

An interrupted run can continue without repeating completed paper ingestion:

```bash
python scripts/acceptance_real_pipeline.py \
  --run-root DATA_DIR/validation/runs/<run> \
  --dataset <dataset-from-the-original-run> \
  --resume
```

Retrieval is not allowed to mutate raw, parsed, or canonical evidence. Every
accepted query must return provenance-complete evidence, canonical chunk IDs,
and citations in the generated answer.

## Hard integrity and soft quality

`pipeline_status` is the acceptance gate. Missing or corrupt PDFs, MinerU or
Cognee write failures, invalid Canonical/ChunkProjection structure, token or
Section boundary violations, projection ID inconsistency, empty retrieval,
missing evidence/page provenance, uncited answers, unexecuted profile stages,
and leaked child processes are hard failures.

Model-dependent expectations are measurements, not gates: expected-paper,
concept, evidence-group, and distinct-document hit rates become
`quality_metrics`; misses become `quality_warnings`. The report records
`quality_status` as `reasonable`, `weak`, or `unevaluated`. Natural-language
wording and optional extracted entities/claims are never exact-match assertions.
The current completion criterion is `pipeline_status == passed`.

Each of `truth`, `associative`, and `comprehensive` must still execute at least
one genuine query and its expected channels/stages. This verifies code-path
coverage rather than a model-specific answer.

## Permanent direct-run contracts

These scripts are retained across Cognee upgrades and do not use pytest:

```bash
python tests/contract/test_cognee_retrieval_boundary.py
python tests/contract/test_cognee_retrieval_boundary.py \
  --live-data-dir data --dataset papers
python tests/contract/test_portable_data_paths.py \
  --data-dir data/validation/runs/<latest> --relocate
```

The first contract keeps private Cognee retrieval details outside business
modules. Live mode derives queries from real retained Chunk, Entity, Claim, and
Summary DataPoints, compares public search, public recall, compatibility vector
search, and typed graph context, then atomically writes
`logs/contracts/cognee-retrieval-boundary.json`. Public limitations are not a
pipeline failure when the compatibility path preserves identity and provenance.
The second scans retained SQLite/JSON references and performs a real
raw/parsed/canonical checksum and FTS relocation check.

## Retrieval quality benchmark

Capability and quality are separate. The live boundary contract decides whether
identity/provenance compatibility is required; the direct-run benchmark compares
relevance and runtime on the retained 22 real queries without pytest, mocks, or
production tuning:

```bash
python tests/validation/retrieval_quality_benchmark.py \
  --run-root data/validation/runs/<latest> \
  --dataset <dataset-from-the-original-run> --resume
```

The report is written to
`logs/contracts/cognee-retrieval-quality-benchmark.json`. It stores every raw
case, separates formatted context from structured provenance, and treats
semantic relevance as a soft metric. Configuration F runs only the ten
associative/comprehensive cases because context extension invokes the real LLM.

## Retrieval graph validation

Graph rendering is optional and remains a test artifact:

```bash
python scripts/acceptance_real_pipeline.py   --run-root DATA_DIR/validation/runs/<run>   --dataset <dataset-from-the-original-run>   --resume   --visualize-graphs
```

After retrieval and evidence validation, associative and comprehensive cases
write `logs/graphs/<case_id>.graph.json` and `*.graph.svg`. Nodes and edges
come only from the real QueryResponse provenance and retained Cognee graph
snapshots; no relation is inferred and no second graph index is created.
Rendering is capped and records truncation counts. A rendering error is reported
through `graph_visualization_status` and warnings, never through
`pipeline_status`.

The final report records profile case counts, actual Cognee LLM/embedding
provider and model, Cognee version, fallbacks actually used by that process,
live-contract status/path, and graph visualization status/outputs.
