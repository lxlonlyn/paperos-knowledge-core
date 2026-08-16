# Reference Test Papers - Acceptance Tasks

## Corpus identity

Use the four supplied real PDFs. Recommended ingestion order is reverse chronological:

`EFIS 2026 -> Volume/ADADIV 2025 -> NISE 2023 -> LipMLP 2022`

This intentionally creates provisional cited Works before their PDFs are ingested.

## Stage 01 acceptance

Hard requirements:

1. The four supplied PDFs must resolve to exactly four distinct stable `ingested` ScholarlyWork identities. Additional provisional/identified Works for references outside this corpus are allowed.
2. Before older PDFs are ingested, EFIS references must be able to create/reuse provisional Works for the 2025/2023/2022 papers.
3. Ingesting those PDFs later must upgrade/reuse those existing Works rather than creating duplicates.
4. The six within-corpus `Work --CITES--> Work` edges in `reference_ground_truth.json` must exist. Other citation edges to works outside this corpus are allowed.
5. Forbidden reverse citation edges must not be created.
6. `ReferenceEntry --RESOLVES_TO--> Work` must provide provenance for each CITES edge.
7. Reprocess each of the four PDFs once: Work IDs must remain stable.
8. Rebuild derived Cognee storage: Work IDs and the six citation edges must remain identical.

Primary hard contract:
`reference_ground_truth.json -> citation_edges_within_corpus`

## Stage 02 acceptance

Hard requirements:

1. Cross-paper claims must preserve the source paper and target Work separately.
2. Facts in `cross_paper_facts` must be recoverable as Claim/ABOUT/evidence structures.
3. A Claim discussing a cited Work must have evidence chunks from the source paper, not from the target paper.
4. One Claim may ABOUT multiple Works when the text directly compares multiple methods.
5. `ADADIV` in EFIS must resolve to the same Work as `Volume Preserving Neural Shape Morphing`.
6. Self-limitations must use `role=self`; external evaluations must not be mislabeled as self-claims.
7. Page/canonical provenance must resolve for every accepted fact.
8. ABOUT/CITES/identity edges must not be duplicated into uncontrolled new TripletDataPoints.

Primary hard examples:
- EFIS -> NISE: handcrafted vector field drawback.
- Volume 2025 -> NISE: volume disappear/reappear / lack of volume control.
- Volume 2025 -> LipMLP: no volume control.
- EFIS -> Volume 2025: ADADIV identity and adaptive-divergence description.

## Stage 03 acceptance

Hard requirements:

1. `source_scope` and `subject_scope` must produce different behavior.
2. Querying NISE as a subject must be able to return evidence from Volume 2025 and EFIS 2026.
3. Restricting source to NISE must exclude evidence from later papers.
4. Multi-work queries must merge evidence from multiple Works without requiring deep provenance traversal.
5. The normal path for hidden/external limitations should be shallow:
   `Work A <- ABOUT - Claim B -> source chunk B`.
6. Final evidence must retain canonical/page provenance.
7. Existing semantic traversal budget must remain bounded.

Run all cases in `reference_queries.json`; hard cases are acceptance gates, soft cases are quality observations.
