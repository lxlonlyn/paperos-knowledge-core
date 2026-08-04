# Architecture convergence checkpoint

- Time: 2026-08-04T14:59:13+00:00
- Branch: `master`
- Baseline: `acc51d7d6e29c41848b8f559d44971f7af7a5dff`
- Status: source-deployed single-process architecture implemented and live chain passed.

## Commits

1. `be700dd docs: define PaperOS as a source-deployed single-process application`
2. `acb6b94 refactor: remove Python distribution and Typer CLI`
3. `28cb5ba refactor: introduce single server entry and application lifecycle`
4. `e326e78 refactor: internalize local inference runtime and background worker`
5. `a70b1ce refactor: consolidate API routes, configuration, and storage initialization`
6. `2ce1c3e chore: remove facade modules, placeholder scripts, dead prompts, and unused settings`
7. `test: validate single-entry lifecycle and real-PDF cumulative pipeline`

## Commands executed

```bash
git status
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
python -m compileall src
pytest tests/unit tests/contract -q

conda run -n paperos python scripts/setup_runtime.py
npm run build
conda run -n paperos ruff check .
conda run -n paperos python -m compileall -q paperos_core scripts server.py tests
conda run -n paperos pytest tests/unit tests/contract -q
conda run -n paperos pytest tests/integration/test_local_inference_failures.py -q

PAPEROS_RUN_SERVER_LIFECYCLE=true \
PAPEROS_TEST_RUN_ID=architecture-final-20260804 \
conda run -n paperos pytest \
  tests/integration/test_server_lifecycle.py::test_python_server_owns_one_runtime_and_worker \
  -q --tb=short

set -a
source .env
set +a
PAPEROS_RUN_LIVE_CUMULATIVE=true \
PAPEROS_TEST_RUN_ID=architecture-final-20260804 \
conda run -n paperos pytest \
  tests/integration/test_server_lifecycle.py::test_real_pdf_cumulative_pipeline_uses_only_http \
  -q --tb=short

PAPEROS_MAINTENANCE_RUN_ROOT="$PWD/data/test-runs/architecture-final-20260804/server-cumulative" \
conda run -n paperos pytest \
  tests/integration/test_server_lifecycle.py::test_real_http_maintenance_routes_preserve_source_evidence \
  -q --tb=short
```

## Test results

- Baseline before changes: 12 passed.
- Unit and contract: 16 passed, 0 failed, 10 Cognee/Pydantic deprecation warnings.
- Local inference startup failures: 5 passed, 0 failed.
- Real `python server.py` lifecycle: 1 passed in 25.58 seconds.
- Real cumulative HTTP chain: 1 passed in 116.90 seconds.
- Real HTTP maintenance chain: 1 passed in 218.40 seconds.

## Genuine input

- File: `3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf`
- SHA-256: `f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11`
- Bytes: 35,829,709

## Live cumulative result

```text
HTTP upload -> queued job -> SourceFile -> live MinerU -> 46 parser files
-> CanonicalSnapshot (24 sections, 258 elements, 30 chunks, 61 references)
-> DeepSeek enrichment -> Cognee graph/LanceDB (401 objects, 82 vectors)
-> SQLite FTS (92 objects) -> truth query -> 6 evidence records -> synthesis
```

- MinerU provider: `mineru_cloud`
- MinerU task: `0b54feac-49a5-4eee-a64d-0502c666b5d3`
- SourceFile: `src_618639b789dd64b1c80a67ebafe0586b`
- ParseRun: `parse_fdbe4387a13641afaa87dd3bfe61b9cc`
- CanonicalSnapshot: `snapshot_193c95f1ac56fa8856f51f6e38cea685`
- Document: `doc_abc7c1488a03de43e683d8c5989f916c`
- Cognee dataset: `781b54d7-89e3-5277-bbce-71ee8dfe5a20` (`papers`)
- Query: truth profile, lexical + semantic, reranked and synthesized, complete provenance.
- Final health: every reported component healthy.

## HTTP maintenance result

- Feedback confirmation was stored through `POST /api/v1/feedback`.
- Improve and rebuild jobs both completed through the single Worker.
- Reprocess completed through live MinerU and created one additional immutable snapshot.
- Two identical post-rebuild queries returned the same answer ID and evidence IDs.
- Logical delete removed 92 lexical and 82 vector objects while retaining source evidence.
- Hashes of every pre-existing PDF, MinerU artifact, and canonical file remained unchanged.

## Data and logs

- Test run: `data/test-runs/architecture-final-20260804/server-cumulative/`
- Immutable PDF: `raw/src_618639b789dd64b1c80a67ebafe0586b/source.pdf`
- Parser artifacts: `parsed/src_618639b789dd64b1c80a67ebafe0586b/`
- Canonical snapshot: `canonical/src_618639b789dd64b1c80a67ebafe0586b/`
- Cognee stores: `cognee/system/`, `cognee/graph/`, `cognee/vector/`
- Cognee manifest/enrichment: `cognee/manifests/`, `cognee/enrichment/`
- Indexes: `indexes/lexical.sqlite3`, `indexes/manifests/`
- Job registry: `jobs/registry.sqlite3`
- Acceptance logs: `logs/server.log`, `logs/local-inference.log`, `logs/ingestion-job.json`, `logs/query.json`, `logs/health.json`
- Full test output: `data/test-runs/architecture-final-20260804/logs/quality.log`, `unit-contract.log`, `local-inference-failures.log`, `server-lifecycle-test.log`, `cumulative-live-test.log`, and `maintenance-api-test.log`.

## External and owned runtime results

- MinerU: reachable, authenticated, real task completed.
- DeepSeek: reachable, authenticated, enrichment and answer synthesis completed.
- Local inference: real EmbeddingGemma, Qwen3 reranker, and QMD query-expansion models loaded.
- SIGTERM: FastAPI lifespan completed; Worker and Node process records became `stopped`; ports 8000 and 8081 closed; no child process remained.
- Cognee: relational engine disposed; graph/vector/cache engines closed; Kuzu/LanceDB subprocesses reaped.

## Known limitations

- Acceptance used the prepared Node 24 runtime; `environment.yml` pins Node 22 as the supported deployment version.
- Cognee reports that no matching Hugging Face tokenizer package is installed and uses its TikToken fallback for approximate internal token counts. No dependency was installed or downloaded to alter the prepared environment.
- Cognee 1.4 emits upstream Pydantic deprecation warnings during tests.

## Next entry

Run `python scripts/setup_runtime.py`, then start the application only with
`python server.py`. Agents use the HTTP API or `scripts/agent_client.py`.
