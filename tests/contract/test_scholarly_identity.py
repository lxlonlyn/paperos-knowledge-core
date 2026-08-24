"""Direct ScholarlyWork identity and citation-backbone contract using real papers.

This project intentionally does not use pytest. Run:

    python tests/contract/test_scholarly_identity.py \
        --live-data-dir data/validation/scholarly_work_reference/output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.datapoints import ScholarlyWorkDataPoint
from paperos_core.adapters.cognee.models import canonical_to_datapoints
from paperos_core.domain.canonical import Person
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.domain.provenance import RelationType
from paperos_core.domain.scholarly import WorkIdentityStatus
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import build_data_paths
from paperos_core.storage.initializer import StorageInitializer


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _prepare_real_copy(source: Path, target: Path) -> None:
    for name in ("canonical",):
        source_dir = source / name
        _require(source_dir.is_dir(), f"Real acceptance data is missing {name}/.")
        shutil.copytree(source_dir, target / name)
    for child in ("chunks", "enrichment"):
        source_dir = source / "cognee" / child
        if source_dir.is_dir():
            shutil.copytree(source_dir, target / "cognee" / child)
    registry_source = source / "jobs" / "registry.sqlite3"
    _require(registry_source.is_file(), "Real acceptance registry.sqlite3 is missing.")
    registry_target = target / "jobs" / "registry.sqlite3"
    registry_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_source, registry_target)


def _identity_signature(registry: ScholarlyRegistry) -> str:
    snapshot = registry.identity_snapshot()
    stable = {
        "work_ids": sorted(work["id"] for work in snapshot["works"]),
        "document_links": snapshot["document_links"],
        "reference_links": snapshot["reference_links"],
        "redirects": snapshot["redirects"],
    }
    return json.dumps(stable, ensure_ascii=False, sort_keys=True)


def _graph_signature(graph: Any) -> str:
    payload = {
        "works": sorted(
            json.dumps(
                {
                    field: getattr(node, field)
                    for field in (
                        "id",
                        "canonical_id",
                        "derived_from_ids",
                        "title",
                        "normalized_title",
                        "doi",
                        "arxiv_id",
                        "year",
                        "authors",
                        "identity_status",
                        "identity_confidence",
                        "metadata",
                    )
                },
                default=str,
                sort_keys=True,
            )
            for node in graph.nodes
            if isinstance(node, ScholarlyWorkDataPoint)
        ),
        "backbone": sorted(
            (
                relation.source_id,
                relation.relation_type.value,
                relation.target_id,
                tuple(relation.derived_from_ids),
                tuple(relation.source_chunk_ids),
            )
            for relation in graph.relations
            if relation.relation_type
            in {
                RelationType.REPRESENTS_WORK,
                RelationType.RESOLVES_TO,
                RelationType.CITES,
            }
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _load_enrichment(source_root: Path, snapshot_id: str) -> SemanticEnrichment:
    path = source_root / "cognee" / "enrichment" / f"{snapshot_id}.json"
    _require(path.is_file(), f"Real enrichment is missing for {snapshot_id}.")
    return SemanticEnrichment.model_validate_json(path.read_text(encoding="utf-8"))


def _real_graph_contract(
    source_root: Path,
    repository: CanonicalRepository,
    registry: ScholarlyRegistry,
) -> tuple[dict[str, object], dict[str, str]]:
    graph_signatures: dict[str, str] = {}
    citation_count = 0
    external_work_count = 0
    work_node_count = 0
    for snapshot_id in repository.list_snapshot_ids():
        bundle = repository.get_bundle(snapshot_id)
        projection = repository.get_chunk_projection(snapshot_id)
        context = registry.resolve_bundle(bundle, projection.chunks)
        enrichment = _load_enrichment(source_root, snapshot_id)
        graph = canonical_to_datapoints(
            bundle, projection.chunks, enrichment, context
        )
        graph_signatures[snapshot_id] = _graph_signature(graph)
        work_ids = {
            node.canonical_id
            for node in graph.nodes
            if isinstance(node, ScholarlyWorkDataPoint)
        }
        reference_ids = {reference.id for reference in bundle.references}
        document_work_ids = {
            link["work_id"]
            for link in registry.identity_snapshot()["document_links"]
        }
        for node in graph.nodes:
            if not isinstance(node, ScholarlyWorkDataPoint):
                continue
            work_node_count += 1
            payload = node.model_dump(mode="json")
            for forbidden in (
                "source_file_id",
                "parse_run_id",
                "canonical_snapshot_id",
                "source_chunk_ids",
            ):
                _require(
                    forbidden not in payload,
                    f"External Work projection contains fake provenance: {forbidden}",
                )
            if node.canonical_id not in document_work_ids:
                external_work_count += 1

        for relation in graph.relations:
            if relation.relation_type is not RelationType.CITES:
                continue
            citation_count += 1
            _require(
                relation.source_id in work_ids and relation.target_id in work_ids,
                "Citation backbone contains a non-Work endpoint.",
            )
            _require(
                bool(relation.derived_from_ids)
                and set(relation.derived_from_ids) <= reference_ids,
                "Work CITES Work is missing its ReferenceEntry provenance.",
            )

    _require(work_node_count > 0, "No ScholarlyWork DataPoints were mapped.")
    _require(external_work_count > 0, "No external cited Work was projected.")
    _require(citation_count > 0, "No Work-to-Work citation edge was mapped.")
    return (
        {
            "status": "passed",
            "snapshot_count": len(graph_signatures),
            "work_node_count": work_node_count,
            "external_work_count": external_work_count,
            "work_citation_count": citation_count,
            "fake_external_provenance": False,
        },
        graph_signatures,
    )


def _identity_cases(
    repository: CanonicalRepository,
    registry: ScholarlyRegistry,
) -> dict[str, object]:
    _require(
        registry._authors_compatible("liu h t d", "hsueh ti derek liu"),
        "Short surname boundary matching regressed.",
    )
    bundles = repository.list_bundles()
    _require(bundles, "Real corpus contains no canonical bundles.")

    document = bundles[0].document
    first = registry.resolve_document(document)
    second = registry.resolve_document(document)
    _require(first.id == second.id, "Repeated resolution changed the Work ID.")

    distinct_doi_documents = {
        registry.normalize_doi(bundle.document.doi): bundle.document
        for bundle in bundles
        if bundle.document.doi
    }
    doi_documents = list(distinct_doi_documents.values())
    _require(len(doi_documents) >= 2, "Real corpus needs two distinct DOI papers.")
    doi_document = doi_documents[0]
    doi_work = registry.resolve_document(doi_document)
    doi_resolution_probe = doi_document.model_copy(
        update={"id": f"{doi_document.id}_doi_contract"}
    )
    _require(
        registry.resolve_document(doi_resolution_probe).id == doi_work.id,
        "Exact DOI did not reconcile to the existing Work.",
    )

    contract_arxiv_id = "2608.00001"
    arxiv_work = registry.resolve_document(
        doi_document.model_copy(update={"arxiv_id": contract_arxiv_id})
    )
    reference_template = next(
        reference for bundle in bundles for reference in bundle.references
    )
    arxiv_reference = reference_template.model_copy(
        update={
            "id": f"{reference_template.id}_arxiv_contract",
            "document_id": document.id,
            "title": doi_document.title,
            "authors": [
                person.display_name for person in doi_document.authors
            ],
            "year": doi_document.year,
            "doi": None,
            "arxiv_id": f"https://arxiv.org/abs/{contract_arxiv_id}v3",
        }
    )
    arxiv_resolution = registry.resolve_reference(arxiv_reference)
    _require(
        arxiv_resolution.work_id == arxiv_work.id,
        "Normalized arXiv exact match did not reconcile.",
    )

    provisional_work = next(
        (
            work
            for work in registry.list_works()
            if work.identity_status is WorkIdentityStatus.PROVISIONAL
        ),
        None,
    )
    _require(
        provisional_work is not None,
        "Real references did not yield a provisional external Work.",
    )
    provisional_reference = next(
        (
            reference
            for bundle in bundles
            for reference in bundle.references
            if (linked := registry.work_for_reference(reference.id)) is not None
            and linked.id == provisional_work.id
        ),
        None,
    )
    _require(
        provisional_reference is not None
        and provisional_work.title
        and provisional_work.year
        and provisional_work.authors,
        "Provisional Work lacks real title/year/author identity metadata.",
    )
    ingested_people = [
        Person(
            id=f"contract_person_{index}",
            display_name=name,
            raw_name=name,
        )
        for index, name in enumerate(provisional_work.authors)
    ]
    ingested_document = document.model_copy(
        update={
            "id": f"{document.id}_provisional_ingest_contract",
            "title": provisional_work.title,
            "year": provisional_work.year,
            "authors": ingested_people,
            "doi": provisional_reference.doi,
            "arxiv_id": provisional_reference.arxiv_id,
        }
    )
    promoted = registry.resolve_document(ingested_document)
    _require(
        promoted.id == provisional_work.id,
        "Later ingest did not reuse the provisional Work ID.",
    )
    _require(
        promoted.identity_status is WorkIdentityStatus.INGESTED,
        "Provisional Work was not promoted to ingested.",
    )

    left_document = next(item for item in doi_documents if item.authors)
    right_document = next(item for item in doi_documents if item is not left_document)
    left_work = registry.resolve_document(left_document)
    right_clone = right_document.model_copy(
        update={
            "id": f"{right_document.id}_title_ambiguity_contract",
            "title": left_document.title,
            "year": left_document.year,
            "authors": left_document.authors,
        }
    )
    right_work = registry.resolve_document(right_clone)
    _require(left_work.id != right_work.id, "Different exact DOIs were merged.")
    ambiguous_reference = reference_template.model_copy(
        update={
            "id": f"{reference_template.id}_title_ambiguity_contract",
            "document_id": document.id,
            "title": left_document.title,
            "authors": [
                person.display_name for person in left_document.authors
            ],
            "year": left_document.year,
            "doi": None,
            "arxiv_id": None,
        }
    )
    ambiguous = registry.resolve_reference(ambiguous_reference)
    _require(
        ambiguous.work_id is None
        and ambiguous.resolution_status == "ambiguous",
        "Ambiguous title was automatically mis-merged.",
    )

    merge_candidate = next(
        (
            work
            for work in registry.list_works()
            if work.identity_status is WorkIdentityStatus.PROVISIONAL
            and work.id != promoted.id
        ),
        None,
    )
    _require(merge_candidate is not None, "No second provisional merge case exists.")
    survivor = registry.merge(left_work.id, merge_candidate.id)
    _require(
        survivor.id == left_work.id,
        "Merge priority did not retain the ingested Work.",
    )
    _require(
        registry.canonicalize_work_id(merge_candidate.id) == survivor.id
        and registry.list_redirects().get(merge_candidate.id) == survivor.id,
        "Merged Work ID was not retained as a canonical redirect.",
    )

    return {
        "status": "passed",
        "stable_repeated_resolution_work_id": first.id,
        "doi_exact_work_id": doi_work.id,
        "arxiv_exact_work_id": arxiv_work.id,
        "promoted_provisional_work_id": promoted.id,
        "ambiguous_title_preserved": True,
        "merge_redirect_survivor": survivor.id,
    }


def run_contract(source_root: Path) -> dict[str, object]:
    source_root = source_root.expanduser().resolve(strict=False)
    with tempfile.TemporaryDirectory(prefix="paperos-scholarly-") as directory:
        target = Path(directory) / "data"
        _prepare_real_copy(source_root, target)
        paths = build_data_paths(target)
        StorageInitializer(paths).initialize()
        repository = CanonicalRepository(paths)
        registry = ScholarlyRegistry(paths)

        initial_contexts = registry.backfill(repository)
        _require(initial_contexts, "Backfill processed no real documents.")
        first_signature = _identity_signature(registry)
        graph_report, first_graphs = _real_graph_contract(
            source_root, repository, registry
        )

        registry.backfill(repository)
        second_signature = _identity_signature(registry)
        _require(
            first_signature == second_signature,
            "Repeated deterministic backfill changed Work/link identity.",
        )
        _, second_graphs = _real_graph_contract(
            source_root, repository, registry
        )
        _require(
            first_graphs == second_graphs,
            "Reprojection changed the Work citation backbone.",
        )

        return {
            "backfill": {
                "status": "passed",
                "document_count": len(initial_contexts),
                "active_work_count": len(registry.list_works()),
                "identity_and_backbone_stable": True,
            },
            "graph": graph_report,
            "identity_cases": _identity_cases(repository, registry),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_contract(args.live_data_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
