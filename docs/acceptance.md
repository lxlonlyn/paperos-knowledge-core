# Real-case acceptance

PaperOS validation uses only genuine academic PDFs supplied under
`DATA_DIR/validation/corpus/pdfs`. The project does not use pytest, mocks, stubs,
synthetic documents, reconstructed parser output, precomputed embeddings,
fixed reranking output, or fixed LLM responses.

`scripts/acceptance_real_pipeline.py` is the single cumulative validation
entry. It verifies each PDF checksum and executes:

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
