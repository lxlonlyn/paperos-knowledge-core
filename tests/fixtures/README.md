# Test Corpus

No fake, mock, stub, synthetic, generated, simplified, reconstructed, or prerecorded document inputs are permitted.

The repository does not contain substitute OCR results or manually constructed downstream objects.

Genuine academic PDF files are supplied by the user under:

```text
DATA_DIR/test-corpus/pdfs/
````

Expected structural checks are stored under:

```text
DATA_DIR/test-corpus/expected/
```

Research-query requirements are stored under:

```text
DATA_DIR/test-corpus/queries/
```

Test-run artifacts are stored under:

```text
DATA_DIR/test-runs/<run_id>/
```

## Cumulative test rules

### Gate 1

```text
PDF
→ source registration
```

### Gate 2

```text
PDF
→ source registration
→ live MinerU OCR
→ raw parser artifacts
```

### Gate 3

```text
PDF
→ source registration
→ live MinerU OCR
→ raw parser artifacts
→ canonical transformation
```

### Gate 4

```text
PDF
→ source registration
→ live MinerU OCR
→ canonical transformation
→ Cognee and derived indexes
```

### Query gate

```text
PDF corpus
→ complete ingestion
→ comprehensive query
→ source evidence and provenance
```

A later test must execute every earlier stage.

Tests must not:

* inject Stage 2 parser artifacts directly;
* construct Stage 3 canonical objects manually;
* seed Stage 4 Cognee stores manually;
* seed query indexes manually;
* use precomputed embeddings;
* use fixed reranking output;
* use fixed DeepSeek output;
* skip an earlier stage because downstream code is under test.

Pure transformation assertions may inspect real artifacts produced during the same cumulative test run. They must not use independently fabricated input.
