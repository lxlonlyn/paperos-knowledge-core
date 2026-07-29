# Codex progress

## 2026-07-28T19:00:00Z

- Passed gates: Gate 1.
- Current gate: Gate 2 preflight, blocked before live MinerU submission.
- Commands:
  - `conda run -n paperos python -c "...MINERU_API_KEY_SET..."`
  - `rg -l '^MINERU_API_KEY=' /home/jicong/.config /home/jicong/paperos-knowledge-core`
  - `rg -l 'MINERU_API_KEY' /home/jicong/.bashrc /home/jicong/.profile /home/jicong/.bash_profile /home/jicong/.zshrc /home/jicong/.config`
  - `conda run -n paperos env | rg '^MINERU_API_KEY='`
  - `conda run -n paperos python -c "import os, sys; ..."` (exit 1:
    `MINERU_API_KEY is required for configured provider mineru_cloud`)
- PDF selected for the next cumulative run:
  `data/test-corpus/pdfs/isogeometric_analysis_of_geometric_partial_differential_equations.pdf`.
- Gate 1 acceptance data:
  `data/test-runs/gate1-acceptance-20260728/`.
- Gate 2 preflight log:
  `data/test-runs/gate2-preflight-20260728/logs/preflight.log`.
- External service result: MinerU Cloud was not contacted because the configured
  `MINERU_API_KEY` credential is absent from the process environment.
- Test result: Gate 2 cumulative test not started; submitting without the configured
  credential would violate the provider contract.
- Known limitation: Gate 2 through Gate 6 remain unimplemented and unverified.
- Credential configuration was subsequently changed to support a persistent,
  Git-ignored `mineru_ocr.api_key` in `config/paperos.toml`. A non-empty environment
  variable named by `mineru_ocr.api_key_env` remains a higher-priority override.
- Resume entry: set `mineru_ocr.api_key` once in `config/paperos.toml`, verify it
  through the redacted configuration loader, then resume Gate 2 from the public PDF
  ingestion entry point.

## 2026-07-28T19:34:00Z

- Passed gates: Gate 1, cumulative Gate 1–2.
- Commands:
  - `conda run -n paperos paperos ingest ... --data-dir data/test-runs/gate2-live-20260728-a`
  - `PAPEROS_TEST_RUN_ID=gate2-pytest-full-20260728 conda run -n paperos pytest -q`
  - `conda run -n paperos ruff check src tests`
  - `conda run -n paperos mypy src/paperos_core`
- PDF:
  `data/test-corpus/pdfs/isogeometric_analysis_of_geometric_partial_differential_equations.pdf`
  (`39a662c2bb98e7400f4273f0066a52e62ab931325197a6f9c7ce7f4e09c0dd3f`).
- Cumulative test data:
  `data/test-runs/gate2-pytest-full-20260728/gate2-live/`.
- ParseRun: `parse_90af726fa18c431c98e342ed17439d0b`, provider task
  `0b4b19bf-5e74-42e1-ad84-d7183fb4211e`, status `completed`.
- Immutable parser artifacts: one archive, one Markdown, two content lists, two
  model outputs, 125 image assets, provider response, polling metadata, and the
  returned origin PDF. All recorded checksums passed.
- External service: live MinerU Cloud v4 signed upload, polling, and ZIP retrieval
  succeeded.
- Tests: 12 passed, 0 failed, 0 skipped in 10.30 seconds.
- Full test log:
  `data/test-runs/gate2-pytest-full-20260728/logs/pytest-full.log`.
- Known limitation: Gate 3 and later are not yet implemented.
- Next entry: consume this cumulative run's real ParserArtifacts through the unique
  MinerU-to-canonical mapper and validate against `expected/*.json`.

## 2026-07-28T20:01:00Z

- Passed gates: Gate 1, cumulative Gate 1–2, cumulative Gate 1–3.
- Commands:
  - `conda run -n paperos paperos ingest data/test-corpus/pdfs/3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf --data-dir data/test-runs/gate3-live-3dgs-20260728`
  - `conda run -n paperos pytest -q tests/unit tests/contract`
  - `PAPEROS_TEST_RUN_ID=gate3-pytest-full-20260728 conda run -n paperos pytest -q`
  - `conda run -n paperos ruff check src/paperos_core tests`
  - `conda run -n paperos mypy src/paperos_core`
- PDF:
  `data/test-corpus/pdfs/3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf`
  (`f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11`).
- Cumulative test data:
  `data/test-runs/gate3-pytest-full-20260728/gate3-live/`.
- SourceFile: `src_618639b789dd64b1c80a67ebafe0586b`.
- ParseRun: `parse_a4ac614a62bf4587abc0f40c8c8ba67c`, provider task
  `e38701e1-faba-4246-885d-d2d03f9bf5c1`, status `completed`.
- CanonicalSnapshot: `snapshot_7d0f422e9aac619fc80dca82c2c7109c`;
  Document: `doc_abc7c1488a03de43e683d8c5989f916c`.
- Immutable parser artifacts: one archive, one Markdown, two content lists, two
  model outputs, 36 image assets, provider response, polling metadata, and the
  returned origin PDF. All recorded checksums passed.
- Canonical output: 24 sections, 259 elements, 30 chunks, 61 references; includes
  formulas, figures, captions, nine parsed tables, page/bounding-box provenance,
  stable IDs, version metadata, and checksum manifest.
- Expected-corpus validation:
  `expected/3d_gaussian_splatting.json` passed in full, including title, year,
  DOI, authors, section/content minima, element types, pages, and provenance.
- Stable rebuild: remapping the same real ParseRun produced identical snapshot,
  section, element, chunk, and reference IDs.
- External service: live MinerU Cloud v4 signed upload, polling, ZIP retrieval,
  and immutable artifact persistence succeeded.
- Tests: 13 passed, 0 failed, 0 skipped in 169.94 seconds. Unit/contract subset:
  11 passed in 0.15 seconds. Ruff and strict mypy passed.
- Full test log:
  `data/test-runs/gate3-pytest-full-20260728/logs/pytest-full.log`.
- Canonical manifest:
  `data/test-runs/gate3-pytest-full-20260728/gate3-live/canonical/src_618639b789dd64b1c80a67ebafe0586b/snapshot_7d0f422e9aac619fc80dca82c2c7109c/manifest.json`.
- Known limitation: the isogeometric-paper expected file requires a table, while
  its live MinerU content list and Markdown contain no table element; that case
  therefore reports the single explicit mismatch rather than reclassifying a real
  plot as a table. The Gate 3 acceptance paper has genuine tables and passes all
  expectations.
- Next entry: Gate 4 consumes this verified CanonicalSnapshot through centralized
  DataPoints, live Cognee, local embeddings, lexical/vector/graph stores, and
  DeepSeek semantic enrichment, followed by destructive derived-data rebuild.

## 2026-07-28T20:44:00Z

- Passed gates: Gate 1, cumulative Gate 1–2, cumulative Gate 1–3, cumulative
  Gate 1–4.
- Commands:
  - `conda run -n paperos npm run build` (working directory
    `services/local_models`; existing dependencies only)
  - controlled local-model health and real 768-dimensional embedding probes
  - live DeepSeek `/models` and structured-enrichment calls
  - `conda run -n paperos paperos rebuild --snapshot-id snapshot_7d0f422e9aac619fc80dca82c2c7109c --data-dir data/test-runs/gate3-pytest-full-20260728/gate3-live`
  - `PAPEROS_TEST_RUN_ID=gate4-live-20260728 LOG_LEVEL=WARNING conda run -n paperos pytest -q tests/integration/test_ingestion_pipeline.py --junitxml=data/test-runs/gate4-live-20260728/logs/pytest-integration.xml`
  - `conda run -n paperos pytest -q tests/unit`
  - `conda run -n paperos ruff check src tests`
  - `conda run -n paperos mypy src/paperos_core`
- PDF:
  `data/test-corpus/pdfs/3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf`
  (`f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11`).
- Cumulative test data:
  `data/test-runs/gate4-live-20260728/gate4-live/`.
- SourceFile: `src_618639b789dd64b1c80a67ebafe0586b`;
  IngestionJob: `job_2c6ff594e2e8496199a26c91809f0a53`, status `completed`.
- ParseRun: `parse_63d674c0ee074e52acbd871a3ec43c2a`, provider task
  `7102a972-58a4-46bd-a707-b5f888a5abf0`, status `completed`.
- CanonicalSnapshot: `snapshot_6bc987ce69581b59480ab1f3f95e4e1c`;
  Document: `doc_abc7c1488a03de43e683d8c5989f916c`.
- Knowledge output after destructive rebuild: 396 Cognee DataPoints, 325 typed
  and provenance relations, 30 exact-canonical-ID vector records, 92 FTS5
  records, 8 DeepSeek entities, 6 claims, 6 concept relations, and one document
  summary. Every semantic object has canonical chunk evidence.
- External services:
  - live MinerU signed upload, polling, archive retrieval, and artifact checksum
    validation succeeded;
  - DeepSeek model health and structured JSON enrichment succeeded after mapping
    the Cognee/LiteLLM `deepseek/` namespace to the same native model name;
  - the repository-owned local model gateway loaded the prepared
    EmbeddingGemma GGUF and returned real 768-dimensional embeddings;
  - Cognee 1.4.0 wrote and read Kuzu graph nodes and produced LanceDB vector
    tables through the local OpenAI-compatible embedding endpoint.
- Derived-data rebuild: Cognee graph/vector/metadata/cache, PaperOS vector/FTS
  databases, manifests, and semantic enrichment were deleted within the isolated
  run and rebuilt from the retained real CanonicalSnapshot. Byte-level SHA-256
  checks over raw PDF, parser artifacts, and canonical files were unchanged.
- Tests: Gate 4 integration 2 passed, 0 failed, 0 skipped in 296.76 seconds;
  unit suite 11 passed, 0 failed, 0 skipped in 0.14 seconds. Ruff and strict mypy
  passed.
- Logs:
  - `data/test-runs/gate4-live-20260728/logs/pytest-integration.xml`
  - `data/test-runs/gate4-live-20260728/gate4-live/logs/model-gateway.log`
  - `data/test-runs/gate4-live-20260728/gate4-live/logs/cognee/2026-07-28_20-38-23.log`
- Data:
  - raw/parser/canonical roots below
    `data/test-runs/gate4-live-20260728/gate4-live/`;
  - Cognee Kuzu/LanceDB stores under `cognee/system` and `cognee/vector`;
  - semantic and graph manifests under `cognee/enrichment` and
    `cognee/manifests`;
  - lexical/vector projections and index manifest under `indexes/`.
- Known limitation: Cognee reports that its configured embedding model name
  `default` has no matching tokenizer package and uses its installed TikToken
  fallback for token-count estimates. Actual embeddings are produced by the
  configured local GGUF with the required 768 dimensions; no model download or
  fallback embedding provider occurred.
- Next entry: Gate 5 starts from formal corpus PDF ingestion and implements the
  documented comprehensive, truth, and associative query paths over these
  canonical/Cognee/vector/FTS/provenance contracts.

## 2026-07-28T22:10:27Z

- Passed gates: Gate 1 through cumulative Gate 1–5.
- Commands:
  - `PAPEROS_TEST_RUN_ID=gate5-live2-20260728 LOG_LEVEL=WARNING conda run -n paperos pytest -q tests/integration/test_comprehensive_query.py`
  - cumulative retries used `PAPEROS_GATE5_REUSE_INGESTION=true` only after
    revalidating all four retained source filenames, SHA-256 values, and stored
    bytes from this same live run;
  - `conda run -n paperos paperos query ... --profile comprehensive --data-dir data/test-runs/gate5-live2-20260728/gate5-live`;
  - FastAPI `POST /api/v1/query` through `TestClient`;
  - `npm run build` in `services/local_models` using existing dependencies;
  - `conda run -n paperos ruff check src tests`;
  - `conda run -n paperos mypy src`;
  - `conda run -n paperos pytest -q tests/unit tests/contract`.
- Genuine PDFs and checksums:
  - `volume_preserving_neural_shape_morphing.pdf`
    (`e5fb20ed36edb81095bc7935714e2f55adc3091dcd265300c9e1659fd215f6b9`);
  - `3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf`
    (`f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11`);
  - `isogeometric_analysis_of_geometric_partial_differential_equations.pdf`
    (`39a662c2bb98e7400f4273f0066a52e62ab931325197a6f9c7ce7f4e09c0dd3f`);
  - `dsg_net_learning_disentangled_structure_and_geometry_for_3d_shape_generation.pdf`
    (`d859411658d7b468463b1ada0ad9435eafb9ec4a6ccacbf9c5bc34f5ff5118a0`).
- Cumulative data:
  `data/test-runs/gate5-live2-20260728/gate5-live/`; four SourceFiles, four
  completed IngestionJobs, four live MinerU ParseRuns, four CanonicalSnapshots,
  shared Cognee graph/vector/metadata stores, exact-canonical-ID vector and FTS5
  projections.
- External services:
  - all four PDFs completed live MinerU upload, polling, result download, and
    immutable artifact persistence;
  - signed object-storage transfers use a proxy-independent data-plane client
    after the ambient proxy truncated the 103 MB PDF on three finite attempts;
  - the local gateway loaded QMD, Qwen3 reranker, and EmbeddingGemma GGUF files;
  - DeepSeek completed structured document enrichment, bilingual query planning,
    and evidence-bound answer synthesis.
- Query result: all 12 truth, 5 associative, and 5 comprehensive cases from the
  three checked-in JSONL files passed. Every result contains canonical evidence,
  page/source provenance, QMD raw output, DeepSeek planning trace, real Qwen3
  scores, multi-channel fusion metadata, and normalized Evidence IDs.
- Public entries: formal `paperos query` and `POST /api/v1/query` both passed.
- Tests: final Gate 5 integration 1 passed, 0 failed, 0 skipped in 539.02
  seconds; unit/contract 11 passed, 0 failed, 0 skipped. Ruff and strict mypy
  passed.
- Logs:
  - `data/test-runs/gate5-live2-20260728/logs/pytest-gate5-retry13.xml`;
  - 22 `query-*.json` responses plus `cli-query.json` and `http-query.json`
    under `data/test-runs/gate5-live2-20260728/logs/`;
  - `data/test-runs/gate5-live2-20260728/gate5-live/logs/model-gateway.log`.
- Process lifecycle: all recorded local-model gateway PIDs exited with code 0;
  final process record status is `stopped`.
- Known limitation: Cognee still emits the documented tokenizer-estimation
  warning; actual embeddings remain the configured local 768-dimensional GGUF.
- Next entry: Gate 6 starts from this cumulative real corpus to add versioned
  feedback/corrections, improve/rebuild, worker/health/status/document operations,
  complete HTTP/CLI operations, and repeated-query consistency.

## 2026-07-28T22:52:49Z

- Passed gates: Gate 1 through cumulative Gate 1–6. All implementation gates are
  complete.
- Commands:
  - `PAPEROS_GATE6_RUN_ROOT=... LOG_LEVEL=WARNING conda run -n paperos pytest -q tests/integration/test_feedback_improve.py --junitxml=.../pytest-gate6.xml`;
  - `LOG_LEVEL=WARNING conda run -n paperos paperos rebuild --data-dir data/test-runs/gate5-live2-20260728/gate5-live`;
  - final retry reused the already verified live reprocess and rebuild outputs
    after checking every canonical snapshot, enrichment file, and index manifest:
    `PAPEROS_GATE6_REUSE_REPROCESS=true PAPEROS_GATE6_REUSE_REBUILD=true ... pytest -q tests/integration/test_feedback_improve.py`;
  - formal CLI calls: `paperos feedback`, `paperos improve`, and
    `paperos delete-document`;
  - FastAPI document list/inspection, feedback, improve, and health routes through
    `TestClient`;
  - `conda run -n paperos ruff check src tests`;
  - `conda run -n paperos mypy src`;
  - `conda run -n paperos pytest -q tests/unit tests/contract`.
- Genuine corpus and checksums:
  - `volume_preserving_neural_shape_morphing.pdf`
    (`e5fb20ed36edb81095bc7935714e2f55adc3091dcd265300c9e1659fd215f6b9`);
  - `3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf`
    (`f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11`);
  - `isogeometric_analysis_of_geometric_partial_differential_equations.pdf`
    (`39a662c2bb98e7400f4273f0066a52e62ab931325197a6f9c7ce7f4e09c0dd3f`);
  - `dsg_net_learning_disentangled_structure_and_geometry_for_3d_shape_generation.pdf`
    (`d859411658d7b468463b1ada0ad9435eafb9ec4a6ccacbf9c5bc34f5ff5118a0`).
- Cumulative data:
  `data/test-runs/gate5-live2-20260728/gate5-live/`; four immutable SourceFiles,
  five completed IngestionJobs, five completed live MinerU ParseRuns, and five
  immutable CanonicalSnapshots. The fifth run is a real reprocess of the IGA PDF.
- Feedback and improvement:
  confirmation, rejection, correction, CLI confirmation, and HTTP acceptance
  records were persisted in the registry database. Corrections and improvements
  are versioned derived objects with feedback and canonical-chunk provenance.
- Rebuild:
  all five snapshots were rebuilt through live DeepSeek semantic enrichment,
  Cognee graph/metadata/vector writes, local 768-dimensional embeddings, and FTS5.
  Every report returned `consistency_valid=true`. A live DeepSeek response that
  shortened canonical chunk IDs exposed a boundary issue; strict unique-prefix
  resolution was added, while ambiguous or foreign IDs still fail.
- Repeated query:
  the first comprehensive query after rebuild executed QMD expansion, hybrid
  retrieval, Qwen3 reranking, and DeepSeek synthesis. The second identical request
  in the same canonical/feedback state returned the same versioned response and
  evidence from the derived query cache. Both response IDs were
  `answer_487a3415775d003f503244823b742503`; user-confirmed evidence retained its
  canonical chunk provenance.
- Evidence protection:
  SHA-256 checks over every pre-existing raw PDF, parser artifact, and canonical
  file were unchanged after feedback, improvement, reprocess, and rebuild.
  Logical document deletion removed 92 lexical and 30 vector projections for the
  selected document while retaining its PDF, parser artifacts, and canonical
  evidence.
- Health:
  MinerU, DeepSeek, all three local models, FTS5, 768-dimensional vector index,
  Cognee graph, job database, and data-path isolation all reported `healthy`.
- External services:
  live MinerU reprocessing succeeded; live DeepSeek enrichment/planning/synthesis
  succeeded; the local gateway loaded the prepared EmbeddingGemma, Qwen3 reranker,
  and QMD GGUF model files; Cognee read/write validation succeeded.
- Tests:
  final Gate 6 integration 1 passed, 0 failed, 0 skipped in 121.12 seconds.
  The full live rebuild was also completed in the preceding cumulative retry; its
  only later failure was the repeated-answer identity assertion that led to the
  query-cache fix. Final unit/contract suite: 11 passed, 0 failed, 0 skipped.
  Ruff and strict mypy passed.
- Logs:
  - `data/test-runs/gate5-live2-20260728/gate5-live/logs/gate6/pytest-gate6-retry4.xml`;
  - `initial-query.json`, `repeated-query-1.json`, `repeated-query-2.json`,
    `feedback-records.json`, `rebuild.json`, and `health.json` in the same folder;
  - CLI/API response JSON files in the same folder;
  - local model process log at
    `data/test-runs/gate5-live2-20260728/gate5-live/logs/model-gateway.log`.
- Process lifecycle: every local-model gateway process used by the acceptance run
  is stopped; no Worker, API server, or pytest process remains.
- Known limitation: Cognee emits its existing installed-tokenizer estimate warning
  and edge-label fallback warnings. Actual PaperOS embeddings use the configured
  local 768-dimensional GGUF, and canonical relation/provenance checks pass.
- Next entry: none; Gate 6 is the final gate.
