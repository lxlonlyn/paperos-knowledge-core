"""Canonical-to-Cognee DataPoint mapping using shared canonical IDs."""

from __future__ import annotations

from dataclasses import dataclass

from paperos_core.adapters.cognee.compat import cognee_uuid
from paperos_core.adapters.cognee.datapoints import (
    ChunkDataPoint,
    ClaimDataPoint,
    ConceptRelationDataPoint,
    DocumentDataPoint,
    ElementDataPoint,
    EntityDataPoint,
    PaperOSGraphDataPoint,
    ReferenceDataPoint,
    ScholarlyWorkDataPoint,
    SectionDataPoint,
    SummaryDataPoint,
    TripletDataPoint,
)
from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.domain.ids import knowledge_triplet_id
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.domain.provenance import RelationRecord, RelationType
from paperos_core.domain.scholarly import ScholarlyContext


@dataclass(slots=True)
class DataPointGraph:
    nodes: list[PaperOSGraphDataPoint]
    relations: list[RelationRecord]

    @property
    def id_mapping(self) -> dict[str, str]:
        return {node.canonical_id: str(node.id) for node in self.nodes}

    def to_json(self) -> dict[str, object]:
        return {
            "nodes": [
                {"__type__": type(node).__name__, **node.model_dump(mode="json")}
                for node in self.nodes
            ],
            "relations": [
                relation.model_dump(mode="json") for relation in self.relations
            ],
        }

def canonical_to_datapoints(
    bundle: CanonicalBundle,
    chunks: list[Chunk],
    enrichment: SemanticEnrichment,
    scholarly: ScholarlyContext,
) -> DataPointGraph:
    snapshot = bundle.snapshot
    common = {
        "canonical_snapshot_id": snapshot.id,
        "source_file_id": snapshot.source_file_id,
        "parse_run_id": snapshot.parse_run_id,
    }
    document = bundle.document
    resolutions = scholarly.resolution_by_reference()
    nodes: list[PaperOSGraphDataPoint] = [
        ScholarlyWorkDataPoint(
            id=cognee_uuid(work.id),
            canonical_id=work.id,
            title=work.title,
            normalized_title=work.normalized_title,
            doi=work.doi,
            arxiv_id=work.arxiv_id,
            year=work.year,
            authors=work.authors,
            identity_status=work.identity_status.value,
            identity_confidence=work.identity_confidence,
            derived_from_ids=[
                item.reference_id
                for item in scholarly.reference_resolutions
                if item.work_id == work.id
            ] + ([document.id] if work.id == scholarly.document_work.id else []),
        )
        for work in scholarly.works
    ]
    nodes.append(
        DocumentDataPoint(
            id=cognee_uuid(document.id),
            canonical_id=document.id,
            work_id=scholarly.document_work.id,
            title=document.title,
            document_type=document.document_type,
            language=document.language,
            doi=document.doi,
            year=document.year,
            **common,
        )
    )
    nodes.extend(
        SectionDataPoint(
            id=cognee_uuid(section.id),
            canonical_id=section.id,
            document_id=document.id,
            title=section.title,
            path=section.path,
            level=section.level,
            **common,
        )
        for section in bundle.sections
    )
    nodes.extend(
        ChunkDataPoint(
            id=cognee_uuid(chunk.id),
            canonical_id=chunk.id,
            document_id=document.id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_chunk_ids=[chunk.id],
            derived_from_ids=chunk.element_ids,
            **common,
        )
        for chunk in chunks
    )
    chunks_by_element: dict[str, list[str]] = {}
    for chunk in chunks:
        for element_id in chunk.element_ids:
            chunks_by_element.setdefault(element_id, []).append(chunk.id)
    nodes.extend(
        ElementDataPoint(
            id=cognee_uuid(element.id),
            canonical_id=element.id,
            document_id=document.id,
            section_id=element.section_id,
            element_type=element.element_type.value,
            text=element.text or element.latex,
            page=element.page,
            source_chunk_ids=chunks_by_element.get(element.id, []),
            **common,
        )
        for element in bundle.elements
    )
    nodes.extend(
        ReferenceDataPoint(
            id=cognee_uuid(reference.id),
            canonical_id=reference.id,
            document_id=document.id,
            raw_text=reference.raw_text,
            doi=reference.doi,
            year=reference.year,
            resolved_work_id=(
                resolutions[reference.id].work_id
                if reference.id in resolutions
                else None
            ),
            resolution_status=(
                resolutions[reference.id].resolution_status
                if reference.id in resolutions
                else "unresolved"
            ),
            source_chunk_ids=chunks_by_element.get(reference.source_element_id or "", []),
            derived_from_ids=([reference.source_element_id] if reference.source_element_id else []),
            **common,
        )
        for reference in bundle.references
    )
    nodes.extend(
        EntityDataPoint(
            id=cognee_uuid(entity.id),
            canonical_id=entity.id,
            entity_type=entity.entity_type,
            name=entity.name,
            description=entity.description,
            status=entity.status.value,
            confidence=entity.confidence,
            source_chunk_ids=entity.source_chunk_ids,
            derived_from_ids=entity.derived_from_ids,
            **common,
        )
        for entity in enrichment.entities
    )
    nodes.extend(
        ClaimDataPoint(
            id=cognee_uuid(claim.id),
            canonical_id=claim.id,
            text=claim.text,
            claim_type=claim.claim_type,
            status=claim.status.value,
            confidence=claim.confidence,
            source_chunk_ids=claim.source_chunk_ids,
            derived_from_ids=claim.derived_from_ids,
            **common,
        )
        for claim in enrichment.claims
    )
    nodes.extend(
        ConceptRelationDataPoint(
            id=cognee_uuid(relation.id),
            canonical_id=relation.id,
            relation_type=relation.relation_type,
            source_object_id=relation.source_object_id,
            target_object_id=relation.target_object_id,
            description=relation.description,
            status=relation.status.value,
            confidence=relation.confidence,
            source_chunk_ids=relation.source_chunk_ids,
            derived_from_ids=relation.derived_from_ids,
            **common,
        )
        for relation in enrichment.relations
    )
    nodes.extend(
        SummaryDataPoint(
            id=cognee_uuid(summary.id),
            canonical_id=summary.id,
            summary_type=summary.summary_type,
            text=summary.text,
            status=summary.status.value,
            source_chunk_ids=summary.source_chunk_ids,
            derived_from_ids=summary.derived_from_ids,
            **common,
        )
        for summary in enrichment.summaries
    )
    relations = _consolidate_relations(
        _canonical_relations(bundle, chunks, scholarly)
        + _semantic_relations(bundle, enrichment)
    )
    triplet_nodes, triplet_links = _triplet_datapoints(
        bundle,
        nodes,
        relations,
        common,
    )
    nodes.extend(triplet_nodes)
    relations.extend(triplet_links)
    return DataPointGraph(nodes=nodes, relations=relations)


def _consolidate_relations(
    relations: list[RelationRecord],
) -> list[RelationRecord]:
    """Merge duplicate typed edges while retaining every provenance ID."""
    consolidated: dict[tuple[str, RelationType, str], RelationRecord] = {}
    for relation in relations:
        key = (
            relation.source_id,
            relation.relation_type,
            relation.target_id,
        )
        existing = consolidated.get(key)
        if existing is None:
            consolidated[key] = relation.model_copy(deep=True)
            continue
        existing.source_chunk_ids = list(
            dict.fromkeys(
                [*existing.source_chunk_ids, *relation.source_chunk_ids]
            )
        )
        existing.derived_from_ids = list(
            dict.fromkeys(
                [*existing.derived_from_ids, *relation.derived_from_ids]
            )
        )
    return list(consolidated.values())


def _triplet_datapoints(
    bundle: CanonicalBundle,
    nodes: list[PaperOSGraphDataPoint],
    relations: list[RelationRecord],
    common: dict[str, str],
) -> tuple[list[TripletDataPoint], list[RelationRecord]]:
    """Represent every typed edge as a vector-searchable graph node."""
    nodes_by_id = {node.canonical_id: node for node in nodes}
    triplets: list[TripletDataPoint] = []
    links: list[RelationRecord] = []
    searchable_relations = {
        RelationType.MENTIONS,
        RelationType.SUPPORTS,
        RelationType.CONTRADICTS,
        RelationType.USES,
        RelationType.EXTENDS,
        RelationType.COMPARES_WITH,
        RelationType.EVALUATES_ON,
        RelationType.PROPOSES,
        RelationType.RELATED_TO,
    }
    for relation in relations:
        if relation.relation_type not in searchable_relations:
            continue
        source = nodes_by_id.get(relation.source_id)
        target = nodes_by_id.get(relation.target_id)
        if source is None or target is None:
            continue
        source_chunks = list(
            dict.fromkeys(
                [
                    *relation.source_chunk_ids,
                    *source.source_chunk_ids,
                    *target.source_chunk_ids,
                ]
            )
        )
        triplet_id = knowledge_triplet_id(
            bundle.snapshot.id,
            relation.source_id,
            relation.relation_type.value,
            relation.target_id,
            source_chunks,
        )
        triplets.append(
            TripletDataPoint(
                id=cognee_uuid(triplet_id),
                canonical_id=triplet_id,
                relation_type=relation.relation_type.value,
                source_object_id=relation.source_id,
                target_object_id=relation.target_id,
                text=(
                    f"{_datapoint_label(source)} "
                    f"{relation.relation_type.value.replace('_', ' ').casefold()} "
                    f"{_datapoint_label(target)}"
                ),
                source_chunk_ids=source_chunks,
                derived_from_ids=list(
                    dict.fromkeys(
                        [
                            relation.source_id,
                            relation.target_id,
                            *relation.derived_from_ids,
                        ]
                    )
                ),
                **common,
            )
        )
        for target_id, link_type in (
            (relation.source_id, RelationType.TRIPLET_SOURCE),
            (relation.target_id, RelationType.TRIPLET_TARGET),
        ):
            links.append(
                RelationRecord(
                    source_id=triplet_id,
                    target_id=target_id,
                    relation_type=link_type,
                    source_chunk_ids=source_chunks,
                    derived_from_ids=relation.derived_from_ids,
                )
            )
    return triplets, links


def _datapoint_label(node: PaperOSGraphDataPoint) -> str:
    for field in ("name", "title", "text", "raw_text", "description"):
        value = getattr(node, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return node.canonical_id


def _canonical_relations(
    bundle: CanonicalBundle, chunks: list[Chunk], scholarly: ScholarlyContext
) -> list[RelationRecord]:
    document_id = bundle.document.id
    relations: list[RelationRecord] = []
    for section in bundle.sections:
        relations.append(
            RelationRecord(
                source_id=document_id,
                target_id=section.id,
                relation_type=RelationType.HAS_SECTION,
            )
        )
    for chunk in chunks:
        relations.append(
            RelationRecord(
                source_id=chunk.section_id or document_id,
                target_id=chunk.id,
                relation_type=RelationType.HAS_CHUNK,
                source_chunk_ids=[chunk.id],
            )
        )
        relations.extend(
            RelationRecord(
                source_id=chunk.id,
                target_id=element_id,
                relation_type=RelationType.HAS_ELEMENT,
                source_chunk_ids=[chunk.id],
            )
            for element_id in chunk.element_ids
        )
    relations.extend(
        RelationRecord(
            source_id=document_id,
            target_id=reference.id,
            relation_type=RelationType.HAS_REFERENCE,
        )
        for reference in bundle.references
    )
    relations.append(
        RelationRecord(
            source_id=document_id,
            target_id=scholarly.document_work.id,
            relation_type=RelationType.REPRESENTS_WORK,
            derived_from_ids=[document_id],
        )
    )
    for resolution in scholarly.reference_resolutions:
        if resolution.work_id is None:
            continue
        relations.append(
            RelationRecord(
                source_id=resolution.reference_id,
                target_id=resolution.work_id,
                relation_type=RelationType.RESOLVES_TO,
                source_chunk_ids=resolution.source_chunk_ids,
                derived_from_ids=[resolution.reference_id],
            )
        )
        relations.append(
            RelationRecord(
                source_id=scholarly.document_work.id,
                target_id=resolution.work_id,
                relation_type=RelationType.CITES,
                source_chunk_ids=resolution.source_chunk_ids,
                derived_from_ids=[resolution.reference_id],
            )
        )
    return relations


def _semantic_relations(
    bundle: CanonicalBundle, enrichment: SemanticEnrichment
) -> list[RelationRecord]:
    relations: list[RelationRecord] = []
    for entity in enrichment.entities:
        _append_provenance_relations(
            relations,
            entity.id,
            entity.source_chunk_ids,
            entity.derived_from_ids,
        )
    for claim in enrichment.claims:
        _append_provenance_relations(
            relations,
            claim.id,
            claim.source_chunk_ids,
            claim.derived_from_ids,
        )
    for summary in enrichment.summaries:
        _append_provenance_relations(
            relations,
            summary.id,
            summary.source_chunk_ids,
            summary.derived_from_ids,
        )
    for relation in enrichment.relations:
        try:
            kind = RelationType(relation.relation_type)
        except ValueError:
            kind = RelationType.RELATED_TO
        relations.extend(
            [
                RelationRecord(
                    source_id=relation.source_object_id,
                    target_id=relation.target_object_id,
                    relation_type=kind,
                    source_chunk_ids=relation.source_chunk_ids,
                    derived_from_ids=relation.derived_from_ids,
                )
            ]
        )
        _append_provenance_relations(
            relations,
            relation.id,
            relation.source_chunk_ids,
            relation.derived_from_ids,
        )
    relations.extend(
        RelationRecord(
            source_id=summary.id,
            target_id=bundle.document.id,
            relation_type=RelationType.SUMMARIZES,
            source_chunk_ids=summary.source_chunk_ids,
            derived_from_ids=summary.derived_from_ids,
        )
        for summary in enrichment.summaries
    )
    return relations


def _append_provenance_relations(
    relations: list[RelationRecord],
    object_id: str,
    source_chunk_ids: list[str],
    derived_from_ids: list[str],
) -> None:
    relations.extend(
        RelationRecord(
            source_id=object_id,
            target_id=chunk_id,
            relation_type=RelationType.DERIVED_FROM,
            source_chunk_ids=[chunk_id],
            derived_from_ids=derived_from_ids,
        )
        for chunk_id in source_chunk_ids
    )
