"""Typer command surface for the cumulative PaperOS application."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer

from paperos_core.bootstrap import Application, build_application
from paperos_core.errors import PaperOSError
from paperos_core.feedback.models import FeedbackRequest, FeedbackType

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.pipeline import KnowledgeIngestionResult
    from paperos_core.retrieval.candidates import QueryResponse

from paperos_core.retrieval.candidates import QueryRequest, RetrievalProfile

app = typer.Typer(no_args_is_help=True)
DataDirOption = Annotated[
    Path | None, typer.Option("--data-dir", help="Override runtime data root.")
]
ConfigOption = Annotated[Path | None, typer.Option("--config", help="PaperOS TOML path.")]


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _fail(error: PaperOSError) -> NoReturn:
    typer.echo(json.dumps(error.as_dict(), ensure_ascii=False, indent=2), err=True)
    raise typer.Exit(code=2)


def _build(config: Path | None, data_dir: Path | None) -> Application:
    try:
        return build_application(config_path=config, data_dir=data_dir)
    except PaperOSError as exc:
        _fail(exc)


async def _ingest_and_close(
    application: Application,
    path: Path,
    *,
    dataset: str | None,
    metadata: dict[str, Any] | None,
) -> KnowledgeIngestionResult:
    try:
        return await application.ingestion.ingest_pdf_to_knowledge(
            path, dataset=dataset, user_metadata=metadata
        )
    finally:
        await application.aclose()


async def _rebuild_and_close(application: Application, snapshot_id: str | None) -> dict[str, Any]:
    try:
        report = await application.rebuilder.rebuild(snapshot_id)
        return report.model_dump(mode="json")
    finally:
        await application.aclose()


async def _query_and_close(
    application: Application, request: QueryRequest
) -> QueryResponse:
    try:
        return await application.retrieval.query(request)
    finally:
        await application.aclose()


@app.command()
def init(data_dir: DataDirOption = None, config: ConfigOption = None) -> None:
    """Initialize Gate 1 runtime directories and source registry."""
    application = _build(config, data_dir)
    _emit(
        {
            "status": "initialized",
            "data_dir": str(application.paths.root),
            "registry_db": str(application.paths.registry_db),
        }
    )


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Path to a genuine academic PDF.")],
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
    metadata_json: Annotated[
        str | None,
        typer.Option("--metadata-json", help="Optional JSON object stored with a new SourceFile."),
    ] = None,
) -> None:
    """Ingest a genuine PDF through cumulative parsing, canonical, and indexing."""
    application = _build(config, data_dir)
    metadata: dict[str, Any] | None = None
    if metadata_json is not None:
        try:
            decoded = json.loads(metadata_json)
            if not isinstance(decoded, dict):
                raise TypeError("metadata JSON must be an object")
            metadata = decoded
        except (json.JSONDecodeError, TypeError) as exc:
            typer.echo(
                json.dumps(
                    {
                        "error": {
                            "code": "invalid_cli_input",
                            "message": f"Invalid --metadata-json: {exc}",
                            "retryable": False,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                err=True,
            )
            raise typer.Exit(code=2) from exc
    try:
        result = asyncio.run(
            _ingest_and_close(application, path, dataset=dataset, metadata=metadata)
        )
    except PaperOSError as exc:
        _fail(exc)
    _emit(result.public_dict())


@app.command()
def status(
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
    source_id: Annotated[str | None, typer.Option("--source-id")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 20,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Query Gate 1 registry status, a job, or a source record."""
    application = _build(config, data_dir)
    try:
        if job_id and source_id:
            typer.echo("Use only one of --job-id or --source-id.", err=True)
            raise typer.Exit(code=2)
        if job_id:
            _emit({"job": application.ingestion.get_job(job_id).model_dump(mode="json")})
        elif source_id:
            _emit(
                {"source_file": application.ingestion.get_source(source_id).model_dump(mode="json")}
            )
        else:
            payload = application.ingestion.status(limit=limit)
            payload["document_count"] = len(application.documents.list_documents())
            payload["operational_jobs"] = [
                item.model_dump(mode="json")
                for item in application.queue.list_jobs(limit=limit)
            ]
            payload["indexes"] = {
                "lexical": application.retrieval.index_manager.lexical.status(),
                "vector": asyncio.run(
                    application.knowledge_pipeline.cognee_repository.vector_status()
                ),
            }
            _emit(payload)
    except PaperOSError as exc:
        _fail(exc)
    finally:
        asyncio.run(application.aclose())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Serve the complete local HTTP API."""
    import uvicorn

    from paperos_core.api.app import create_app

    configuration = _build(config, data_dir)
    selected_host = host or configuration.config.api.host
    selected_port = port or configuration.config.api.port
    asyncio.run(configuration.aclose())
    uvicorn.run(
        create_app(config_path=config, data_dir=data_dir),
        host=selected_host,
        port=selected_port,
    )


@app.command()
def worker(
    once: Annotated[bool, typer.Option("--once")] = False,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Run the managed operational worker."""
    application = _build(config, data_dir)

    async def run() -> None:
        try:
            if application.worker is None:
                raise RuntimeError("Worker is not configured")
            with application.worker.lifecycle_lock():
                if once:
                    await application.worker.run_once()
                    return
                while True:
                    job = await application.worker.run_once()
                    if job is None:
                        await asyncio.sleep(
                            application.config.worker.poll_interval_seconds
                        )
        finally:
            await application.aclose()

    try:
        asyncio.run(run())
    except PaperOSError as exc:
        _fail(exc)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Research question.")],
    profile: Annotated[
        RetrievalProfile, typer.Option("--profile")
    ] = RetrievalProfile.COMPREHENSIVE,
    top_k: Annotated[int | None, typer.Option("--top-k", min=1, max=100)] = None,
    document_id: Annotated[
        list[str] | None, typer.Option("--document-id")
    ] = None,
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Query the cumulative canonical, graph, vector, and lexical corpus."""
    application = _build(config, data_dir)
    request = QueryRequest(
        query=question,
        profile=profile,
        dataset=dataset,
        top_k=top_k,
        document_ids=document_id,
    )
    try:
        result = asyncio.run(_query_and_close(application, request))
    except PaperOSError as exc:
        _fail(exc)
    _emit(result.model_dump(mode="json"))


@app.command("rebuild")
def rebuild(
    snapshot_id: Annotated[str | None, typer.Option("--snapshot-id")] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Destructively rebuild Cognee and indexes from retained canonical snapshots."""
    application = _build(config, data_dir)
    try:
        payload = asyncio.run(_rebuild_and_close(application, snapshot_id))
    except PaperOSError as exc:
        _fail(exc)
    _emit(payload)


@app.command()
def feedback(
    target_id: Annotated[str, typer.Argument()],
    feedback_type: Annotated[FeedbackType, typer.Option("--type")],
    evidence_id: Annotated[list[str] | None, typer.Option("--evidence-id")] = None,
    replacement_text: Annotated[
        str | None, typer.Option("--replacement-text")
    ] = None,
    comment: Annotated[str | None, typer.Option("--comment")] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Record immutable user feedback."""
    application = _build(config, data_dir)
    try:
        record = application.feedback.record(
            FeedbackRequest(
                target_id=target_id,
                feedback_type=feedback_type,
                evidence_ids=evidence_id or [],
                replacement_text=replacement_text,
                comment=comment,
            )
        )
        _emit(record.model_dump(mode="json"))
    except PaperOSError as exc:
        _fail(exc)
    finally:
        asyncio.run(application.aclose())


@app.command()
def improve(
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Materialize pending feedback as versioned derived knowledge."""
    application = _build(config, data_dir)
    try:
        _emit(application.feedback.improve().model_dump(mode="json"))
    except PaperOSError as exc:
        _fail(exc)
    finally:
        asyncio.run(application.aclose())


@app.command()
def documents(
    include_deleted: Annotated[
        bool, typer.Option("--include-deleted")
    ] = False,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """List canonical documents."""
    application = _build(config, data_dir)
    _emit(
        {
            "documents": [
                item.model_dump(mode="json")
                for item in application.documents.list_documents(
                    include_deleted=include_deleted
                )
            ]
        }
    )
    asyncio.run(application.aclose())


@app.command()
def reprocess(
    document_id: Annotated[str, typer.Argument()],
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Create a new live ParseRun and derived snapshot for one document."""
    application = _build(config, data_dir)

    async def run() -> dict[str, object]:
        try:
            return await application.documents.reprocess(document_id)
        finally:
            await application.aclose()

    try:
        _emit(asyncio.run(run()))
    except PaperOSError as exc:
        _fail(exc)


@app.command("delete-document")
def delete_document(
    document_id: Annotated[str, typer.Argument()],
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Logically delete one document while retaining immutable source evidence."""
    application = _build(config, data_dir)

    async def run() -> dict[str, object]:
        try:
            report = await application.documents.delete(document_id)
            return report.model_dump(mode="json")
        finally:
            await application.aclose()

    try:
        _emit(asyncio.run(run()))
    except PaperOSError as exc:
        _fail(exc)


@app.command("model-gateway")
def model_gateway(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Override the configured gateway host."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help="Override the configured gateway port.",
        ),
    ] = None,
    data_dir: DataDirOption = None,
    config: ConfigOption = None,
) -> None:
    """Run the local model HTTP gateway until SIGINT or SIGTERM."""
    application = _build(config, data_dir)

    async def run() -> None:
        try:
            await application.model_process.serve_forever(host=host, port=port)
        finally:
            await application.aclose()

    try:
        asyncio.run(run())
    except PaperOSError as exc:
        _fail(exc)
