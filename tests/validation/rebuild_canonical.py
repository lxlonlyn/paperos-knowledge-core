"""Rebuild canonical snapshots from cached MinerU parsed artifacts (no re-OCR)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.enums import ParserArtifactType, ParseRunStatus
from paperos_core.domain.ids import canonical_snapshot_id
from paperos_core.domain.parsing import ParserArtifact, ParseRun
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import build_data_paths


def rebuild_all_canonical_snapshots(
    *,
    run_dir: Path,
    dataset_id: str = "paperos-chunk-corpus-review",
) -> list[dict[str, Any]]:
    """Re-map every cached parse run under ``run_dir/parsed`` into canonical."""
    paths = build_data_paths(run_dir)
    mapper = MinerUCanonicalMapper()
    repository = CanonicalRepository(paths)
    results: list[dict[str, Any]] = []
    parsed_root = run_dir / "parsed"
    for src_dir in sorted(parsed_root.glob("src_*")):
        parse_dirs = sorted(src_dir.glob("parse_*"))
        if not parse_dirs:
            continue
        parse_dir = parse_dirs[-1]
        source_file_id = src_dir.name.removeprefix("src_")
        result = rebuild_canonical_from_parse_dir(
            parse_dir=parse_dir,
            source_file_id=source_file_id,
            canonical_source_dir=src_dir.name,
            mapper=mapper,
            repository=repository,
            dataset_id=dataset_id,
            raw_root=run_dir / "raw",
        )
        results.append(result)
    return results


def rebuild_canonical_from_parse_dir(
    *,
    parse_dir: Path,
    source_file_id: str,
    canonical_source_dir: str,
    mapper: MinerUCanonicalMapper,
    repository: CanonicalRepository,
    dataset_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    manifest = json.loads((parse_dir / "manifest.json").read_text(encoding="utf-8"))
    parse_run_id = manifest["parse_run_id"]
    artifacts = _artifacts_from_manifest(parse_dir, manifest)
    parse_run = ParseRun(
        id=parse_run_id,
        source_file_id=source_file_id,
        provider=manifest.get("provider", "mineru"),
        backend=manifest.get("backend", "mineru"),
        status=ParseRunStatus.COMPLETED,
        request_options={},
        created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
        completed_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
        artifact_manifest_path=parse_dir / "manifest.json",
    )
    raw_path = raw_root / f"src_{source_file_id}"
    pdf_candidates = sorted(raw_path.glob("*.pdf"))
    source = SourceFile(
        id=source_file_id,
        original_filename=pdf_candidates[0].name if pdf_candidates else f"{source_file_id}.pdf",
        storage_path=pdf_candidates[0] if pdf_candidates else raw_path,
        sha256="0" * 64,
        size_bytes=max(pdf_candidates[0].stat().st_size, 1) if pdf_candidates else 1,
        media_type="application/pdf",
        created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
    )
    manifest_path = (
        repository.paths.canonical / canonical_source_dir / canonical_snapshot_id(parse_run_id) / "manifest.json"
    )
    bundle = mapper.build_canonical_snapshot(
        source=source,
        parse_run=parse_run,
        artifacts=artifacts,
        manifest_path=manifest_path,
        dataset_id=dataset_id,
    )
    _overwrite_snapshot(repository, bundle)
    return {
        "source_file_id": source_file_id,
        "canonical_source_dir": canonical_source_dir,
        "parse_run_id": parse_run_id,
        "snapshot_id": bundle.snapshot.id,
        "element_count": len(bundle.elements),
        "reference_count": len(bundle.references),
        "status": "rebuilt",
    }


def _artifacts_from_manifest(parse_dir: Path, manifest: dict[str, Any]) -> list[ParserArtifact]:
    artifacts: list[ParserArtifact] = []
    for item in manifest.get("artifacts", []):
        artifact_type = ParserArtifactType(item["type"])
        storage_path = (parse_dir / item["path"]).resolve()
        artifacts.append(
            ParserArtifact(
                id=item["id"],
                parse_run_id=manifest["parse_run_id"],
                artifact_type=artifact_type,
                storage_path=storage_path,
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
                created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
            )
        )
    return artifacts


def _overwrite_snapshot(repository: CanonicalRepository, bundle) -> None:
    """Write canonical files, replacing prior snapshot content when normalization changes."""
    snapshot = bundle.snapshot
    root = snapshot.manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "snapshot.json": repository._json_bytes(snapshot),
        "document.json": repository._json_bytes(bundle.document),
        "sections.jsonl": repository._jsonl_bytes(bundle.sections),
        "elements.jsonl": repository._jsonl_bytes(bundle.elements),
        "references.jsonl": repository._jsonl_bytes(bundle.references),
        "warnings.json": json.dumps(
            bundle.warnings,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode(),
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    manifest = {
        "schema_version": snapshot.schema_version,
        "id_version": snapshot.id_version,
        "pipeline_version": snapshot.pipeline_version,
        "canonical_snapshot_id": snapshot.id,
        "source_file_id": snapshot.source_file_id,
        "parse_run_id": snapshot.parse_run_id,
        "document_id": snapshot.document_id,
        "dataset_id": snapshot.dataset_id,
        "counts": {
            "sections": len(bundle.sections),
            "elements": len(bundle.elements),
            "references": len(bundle.references),
        },
        "files": [
            {
                "path": name,
                "sha256": __import__("hashlib").sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        ],
    }
    snapshot.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def isogeometric_regression_text(bundle_elements: list) -> dict[str, bool]:
    """Check Isogeometric paragraph markers survived canonical mapping."""
    combined = "\n".join(
        (element.text or element.markdown or "")
        for element in bundle_elements
        if (element.text or element.markdown)
    )
    markers = ["[29]", "[32]", "[30]", "[31]", "[34]", "T <", "J_W", "16\\pi", "16π"]
    return {marker: marker in combined for marker in markers}
