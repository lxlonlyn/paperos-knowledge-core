"""Deterministic Task 02 Claim→ABOUT→ScholarlyWork schema and mapper contracts.

This project intentionally does not use pytest. Run:

    python tests/contract/test_claim_about_semantic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.datapoints import ClaimDataPoint, TripletDataPoint
from paperos_core.adapters.cognee.models import (
    _consolidate_relations,
    _semantic_relations,
    _triplet_datapoints,
    canonical_to_datapoints,
)
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Document,
    ReferenceEntry,
)
from paperos_core.domain.knowledge import (
    Claim,
    ClaimAboutTarget,
    KnowledgeStatus,
    SemanticEnrichment,
)
from paperos_core.domain.provenance import RelationRecord, RelationType
from paperos_core.domain.scholarly import (
    ReferenceWorkResolution,
    ScholarlyContext,
    ScholarlyWork,
    WorkIdentityStatus,
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _snapshot() -> CanonicalSnapshot:
    return CanonicalSnapshot(
        id="snapshot_test",
        document_id="doc_test",
        source_file_id="src_test",
        parse_run_id="parse_test",
        dataset_id="dataset_test",
        manifest_path=Path("/tmp/manifest_test.json"),
        schema_version="1.0",
        id_version="1",
    )


def _document(snapshot: CanonicalSnapshot) -> Document:
    return Document(
        id="doc_test",
        source_file_id=snapshot.source_file_id,
        parse_run_id=snapshot.parse_run_id,
        canonical_snapshot_id=snapshot.id,
        language="en",
        title="Source Paper Title",
    )


def _work(work_id: str, title: str) -> ScholarlyWork:
    return ScholarlyWork(
        id=work_id,
        title=title,
        normalized_title=title.casefold(),
        identity_status=WorkIdentityStatus.INGESTED,
        identity_confidence=1.0,
        year=2024,
        authors=["Author A"],
    )


def retained_enrichment_contract() -> dict[str, object]:
    """Old enrichment JSON without ABOUT / source_work_id must still load."""
    legacy = {
        "entities": [],
        "claims": [
            {
                "id": "claim_legacy",
                "canonical_snapshot_id": "snapshot_test",
                "text": "Legacy claim without ABOUT.",
                "status": "extracted",
                "derived_from_ids": ["chunk_a"],
                "source_chunk_ids": ["chunk_a"],
                "schema_version": "1.0",
                "id_version": "1",
            }
        ],
        "relations": [],
        "summaries": [],
        "model": "contract",
        "provider": "contract",
        "model_version": "contract",
        "prompt_name": "semantic_enrichment",
        "prompt_version": "1",
        "prompt_sha256": "0" * 64,
    }
    enrichment = SemanticEnrichment.model_validate(legacy)
    _require(len(enrichment.claims) == 1, "Legacy claim failed to load.")
    claim = enrichment.claims[0]
    _require(claim.about == [], "Legacy claim about default must be [].")
    _require(claim.source_work_id is None, "Legacy source_work_id default must be None.")
    _require(
        claim.source_document_id is None,
        "Legacy source_document_id default must be None.",
    )
    return {"status": "passed", "legacy_about_default": []}


def about_mapper_contract() -> dict[str, object]:
    snapshot = _snapshot()
    document = _document(snapshot)
    source_work = _work("work_source", "Source Paper Title")
    target_work = _work("work_target", "Cited Paper Title")
    bundle = CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[],
        elements=[],
        references=[
            ReferenceEntry(
                id="ref_1",
                document_id=document.id,
                canonical_snapshot_id=snapshot.id,
                raw_text="Cited Paper Title. Author A. 2024.",
                order=0,
            )
        ],
    )
    scholarly = ScholarlyContext(
        document_work=source_work,
        works=[source_work, target_work],
        reference_resolutions=[
            ReferenceWorkResolution(
                reference_id="ref_1",
                source_document_id=document.id,
                work_id=target_work.id,
                resolution_status="resolved",
                confidence=1.0,
                source_chunk_ids=["chunk_a"],
            )
        ],
    )
    enrichment = SemanticEnrichment(
        entities=[],
        claims=[
            Claim(
                id="claim_external",
                canonical_snapshot_id=snapshot.id,
                text="The cited method needs a handcrafted vector field.",
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=["chunk_a"],
                derived_from_ids=["chunk_a"],
                source_document_id=document.id,
                source_work_id=source_work.id,
                about=[
                    ClaimAboutTarget(
                        work_id=target_work.id,
                        roles=["subject"],
                        source_chunk_ids=["chunk_a"],
                        derived_from_ids=["chunk_a"],
                    )
                ],
            ),
            Claim(
                id="claim_self",
                canonical_snapshot_id=snapshot.id,
                text="Our method requires shared topology.",
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=["chunk_b"],
                derived_from_ids=["chunk_b"],
                source_document_id=document.id,
                source_work_id=source_work.id,
                about=[
                    ClaimAboutTarget(
                        work_id=source_work.id,
                        roles=["self"],
                        source_chunk_ids=["chunk_b"],
                        derived_from_ids=["chunk_b"],
                    )
                ],
            ),
        ],
        relations=[],
        summaries=[],
        model="contract",
        provider="contract",
        model_version="contract",
        prompt_name="semantic_enrichment",
        prompt_version="2",
        prompt_sha256="1" * 64,
    )
    from paperos_core.domain.canonical import Chunk

    chunks = [
        Chunk(
            id="chunk_a",
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            text="handcrafted vector field",
            order=0,
            element_ids=["el_a"],
        ),
        Chunk(
            id="chunk_b",
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            text="shared topology",
            order=1,
            element_ids=["el_b"],
        ),
    ]
    graph = canonical_to_datapoints(bundle, chunks, enrichment, scholarly)
    about_edges = [
        relation
        for relation in graph.relations
        if relation.relation_type is RelationType.ABOUT
    ]
    _require(len(about_edges) == 2, f"Expected 2 ABOUT edges, got {len(about_edges)}")
    by_claim = {edge.source_id: edge for edge in about_edges}
    external = by_claim["claim_external"]
    self_edge = by_claim["claim_self"]
    _require(external.target_id == target_work.id, "External ABOUT target mismatch.")
    _require(external.roles == ["subject"], "External ABOUT role mismatch.")
    _require(external.source_chunk_ids == ["chunk_a"], "External ABOUT provenance mismatch.")
    _require(self_edge.target_id == source_work.id, "Self ABOUT target mismatch.")
    _require(self_edge.roles == ["self"], "Self ABOUT role mismatch.")
    claim_nodes = [node for node in graph.nodes if isinstance(node, ClaimDataPoint)]
    _require(len(claim_nodes) == 2, "ClaimDataPoint count mismatch.")
    for node in claim_nodes:
        _require(node.source_document_id == document.id, "Claim source_document_id missing.")
        _require(node.source_work_id == source_work.id, "Claim source_work_id missing.")
    about_triplets = [
        node
        for node in graph.nodes
        if isinstance(node, TripletDataPoint) and node.relation_type == "ABOUT"
    ]
    _require(about_triplets == [], "ABOUT must not produce TripletDataPoint nodes.")
    return {
        "status": "passed",
        "about_edge_count": len(about_edges),
        "about_triplet_count": 0,
        "source_work_id": source_work.id,
        "external_target_work_id": target_work.id,
        "external_source_work_ne_target": source_work.id != target_work.id,
    }


def about_dedup_contract() -> dict[str, object]:
    duplicates = [
        RelationRecord(
            source_id="claim_1",
            target_id="work_1",
            relation_type=RelationType.ABOUT,
            source_chunk_ids=["chunk_a"],
            derived_from_ids=["chunk_a"],
            roles=["subject"],
        ),
        RelationRecord(
            source_id="claim_1",
            target_id="work_1",
            relation_type=RelationType.ABOUT,
            source_chunk_ids=["chunk_b"],
            derived_from_ids=["chunk_b"],
            roles=["comparison_target"],
        ),
    ]
    merged = _consolidate_relations(duplicates)
    about = [item for item in merged if item.relation_type is RelationType.ABOUT]
    _require(len(about) == 1, "ABOUT edges were not deduplicated.")
    edge = about[0]
    _require(
        edge.source_chunk_ids == ["chunk_a", "chunk_b"],
        "ABOUT provenance merge failed.",
    )
    _require(
        set(edge.roles) == {"subject", "comparison_target"},
        "ABOUT roles merge failed.",
    )
    return {"status": "passed", "deduped_about_count": 1, "roles": edge.roles}


def about_no_triplet_contract() -> dict[str, object]:
    snapshot = _snapshot()
    relations = [
        RelationRecord(
            source_id="claim_1",
            target_id="work_1",
            relation_type=RelationType.ABOUT,
            source_chunk_ids=["chunk_a"],
            derived_from_ids=["chunk_a"],
            roles=["subject"],
        )
    ]
    enrichment = SemanticEnrichment(
        entities=[],
        claims=[],
        relations=[],
        summaries=[],
        model="contract",
        provider="contract",
        model_version="contract",
        prompt_name="semantic_enrichment",
        prompt_version="2",
        prompt_sha256="1" * 64,
    )
    # Ensure ABOUT is excluded from searchable triplet generation.
    nodes: list = []
    common = {
        "canonical_snapshot_id": snapshot.id,
        "source_file_id": "src",
        "parse_run_id": "parse",
    }
    triplets, _links = _triplet_datapoints(
        CanonicalBundle(
            snapshot=snapshot,
            document=_document(snapshot),
            sections=[],
            elements=[],
            references=[],
        ),
        nodes,
        relations,
        common,
    )
    about_triplets = [item for item in triplets if item.relation_type == "ABOUT"]
    _require(about_triplets == [], "ABOUT leaked into TripletDataPoint generation.")
    semantic = _semantic_relations(
        CanonicalBundle(
            snapshot=snapshot,
            document=_document(snapshot),
            sections=[],
            elements=[],
            references=[],
        ),
        enrichment,
    )
    _require(
        all(item.relation_type is not RelationType.ABOUT for item in semantic),
        "Empty enrichment unexpectedly produced ABOUT.",
    )
    return {"status": "passed", "about_triplet_count": 0}


def work_catalog_contract() -> dict[str, object]:
    from paperos_core.adapters.cognee.llm import _SELF_WORK_KEY, _build_work_catalog

    snapshot = _snapshot()
    document = _document(snapshot)
    source_work = _work("work_source", "Source Paper Title")
    target_work = _work("work_target", "Cited Paper Title")
    other_work = _work("work_other", "Other Cited Title")
    bundle = CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[],
        elements=[],
        references=[
            ReferenceEntry(
                id="ref_1",
                document_id=document.id,
                canonical_snapshot_id=snapshot.id,
                raw_text="Cited Paper Title. Author A. 2024.",
                order=0,
            ),
            ReferenceEntry(
                id="ref_2",
                document_id=document.id,
                canonical_snapshot_id=snapshot.id,
                raw_text="Other Cited Title. Author B. 2023.",
                order=1,
            ),
        ],
    )
    scholarly = ScholarlyContext(
        document_work=source_work,
        works=[source_work, target_work, other_work],
        reference_resolutions=[
            ReferenceWorkResolution(
                reference_id="ref_1",
                source_document_id=document.id,
                work_id=target_work.id,
                resolution_status="resolved",
                confidence=1.0,
            ),
            ReferenceWorkResolution(
                reference_id="ref_2",
                source_document_id=document.id,
                work_id=other_work.id,
                resolution_status="resolved",
                confidence=1.0,
            ),
        ],
    )
    catalog = _build_work_catalog(bundle, scholarly)
    _require(_SELF_WORK_KEY in catalog.key_to_work_id, "SELF key missing.")
    _require(
        catalog.key_to_work_id[_SELF_WORK_KEY] == source_work.id,
        "SELF must map to document Work.",
    )
    _require(
        all(key == _SELF_WORK_KEY or key.startswith("CITED_") for key in catalog.key_to_work_id),
        "Catalog keys must be SELF/CITED_*.",
    )
    _require(
        set(catalog.key_to_work_id.values())
        == {source_work.id, target_work.id, other_work.id},
        "Catalog must cover current and cited Works.",
    )
    for entry in catalog.entries:
        _require("work_key" in entry and "title" in entry, "Catalog entry incomplete.")
        _require("work_" not in entry.get("work_key", ""), "Catalog leaked raw Work IDs as keys.")
    return {
        "status": "passed",
        "catalog_size": len(catalog.entries),
        "keys": sorted(catalog.key_to_work_id),
    }


def main() -> None:
    report = {
        "retained_enrichment": retained_enrichment_contract(),
        "work_catalog": work_catalog_contract(),
        "about_mapper": about_mapper_contract(),
        "about_dedup": about_dedup_contract(),
        "about_no_triplet": about_no_triplet_contract(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if any(section.get("status") != "passed" for section in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
