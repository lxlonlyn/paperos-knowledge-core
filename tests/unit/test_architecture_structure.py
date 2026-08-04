from pathlib import Path

from paperos_core.api.app import create_app
from paperos_core.config import load_settings
from paperos_core.prompt_repository import PromptRepository

ROOT = Path(__file__).resolve().parents[2]


def test_single_source_entry_and_removed_facades() -> None:
    assert (ROOT / "server.py").is_file()
    assert not (ROOT / "src").exists()
    assert not (ROOT / "paperos_core" / "cli.py").exists()
    for relative in (
        "paperos_core/ingestion/pipeline.py",
        "paperos_core/ingestion/indexing.py",
        "paperos_core/ingestion/postprocess.py",
        "paperos_core/adapters/cognee/tasks.py",
        "paperos_core/adapters/cognee/retriever.py",
        "paperos_core/adapters/cognee/improve.py",
    ):
        assert not (ROOT / relative).exists()
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "paperos_core").rglob("*.py")
    )
    assert "serve_forever" not in source
    assert "LocalModelGateway" not in source
    assert "import typer" not in source


def test_all_business_routes_are_http_routers() -> None:
    routes = create_app(load_settings()).openapi()["paths"]
    assert {
        "/api/v1/ingest",
        "/api/v1/jobs/{job_id}",
        "/api/v1/query",
        "/api/v1/documents",
        "/api/v1/documents/{document_id}",
        "/api/v1/documents/{document_id}/reprocess",
        "/api/v1/feedback",
        "/api/v1/improve",
        "/api/v1/rebuild",
        "/api/v1/health",
    } <= set(routes)


def test_prompt_repository_is_versioned_and_hashed() -> None:
    repository = PromptRepository()
    for name in (
        "semantic_enrichment",
        "query_planning",
        "query_expansion",
        "answer_synthesis",
    ):
        descriptor = repository.describe(name)
        assert descriptor.text == repository.load(name)
        assert descriptor.version == "1"
        assert len(descriptor.sha256) == 64
