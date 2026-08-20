# Claim / ABOUT ablation fixture

Place this directory at:

`tests/fixtures/scholarly_work_reference/claim_about_ablation/`

It supplements, and does not replace, the existing:

- `reference_corpus_manifest.json`
- `reference_ground_truth.json`
- `reference_queries.json`

Files:

- `ablation_fact_metadata.json`: difficulty / mention-mode labels for existing real-paper facts.
- `ablation_queries.json`: retrieval-ablation queries. Primary cases carry explicit `QueryRequest.scope`.
- `ablation_experiment_spec.json`: required configurations, budgets, metrics, and fairness constraints.

Do not copy PDF files into this directory. The benchmark must use the same four real papers already used by the existing scholarly-work reference acceptance corpus.
