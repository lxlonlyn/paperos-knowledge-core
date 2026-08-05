"""Contract tests binding PaperOS to the current Cognee version and boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUSINESS_ROOTS = (
    ROOT / "paperos_core" / "adapters",
    ROOT / "paperos_core" / "api",
    ROOT / "paperos_core" / "domain",
    ROOT / "paperos_core" / "feedback",
    ROOT / "paperos_core" / "indexes",
    ROOT / "paperos_core" / "ingestion",
    ROOT / "paperos_core" / "jobs",
    ROOT / "paperos_core" / "retrieval",
    ROOT / "paperos_core" / "runtime",
    ROOT / "paperos_core" / "storage",
)


def _business_python_files() -> list[Path]:
    files: list[Path] = []
    for root in BUSINESS_ROOTS:
        files.extend(path for path in root.rglob("*.py") if path.name != "compat.py")
    files.append(ROOT / "paperos_core" / "application.py")
    files.append(ROOT / "paperos_core" / "config.py")
    files.append(ROOT / "paperos_core" / "health.py")
    files.append(ROOT / "paperos_core" / "documents.py")
    return files


def test_pinned_cognee_version_contract() -> None:
    import cognee

    assert cognee.__version__ == "1.4.0"


def test_business_code_uses_only_the_sanctioned_cognee_surface() -> None:
    sanctioned = {
        "cognee",  # top-level public API (search, recall, run_custom_pipeline)
        "cognee.infrastructure.llm",  # LLMGateway (explicit task requirement)
        "cognee.infrastructure.llm.utils",  # test_llm_connection health check
        "cognee.infrastructure.llm.tokenizer.resolver",  # Cognee-provided tokenizer
    }
    import_pattern = re.compile(r"^\s*(?:from|import)\s+(cognee(?:\.[\w.]+)?)")
    violations: list[str] = []
    for path in _business_python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = import_pattern.match(line)
            if match is None:
                continue
            imported = match.group(1)
            if not any(
                imported == item or imported.startswith(f"{item}.")
                for item in sanctioned
            ):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not violations, "\n".join(violations)


def test_vendor_and_vector_boundaries_are_absent_from_business_code() -> None:
    forbidden = (
        "DeepSeek",
        "deepseek",
        "/chat/completions",
        "embed_data(",
        "DataPoint_text",  # hardcoded vector collection names
        "PipelineRun(",
        "DatasetData(",
    )
    violations: list[str] = []
    for path in _business_python_files():
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not violations, "\n".join(violations)


def test_private_cognee_api_is_centralized_in_compat() -> None:
    compat = ROOT / "paperos_core" / "adapters" / "cognee" / "compat.py"
    assert compat.is_file()
    assert not (ROOT / "paperos_core" / "adapters" / "cognee" / "repository.py").exists()
    source = compat.read_text(encoding="utf-8")
    for private_import in (
        "cognee.modules.pipelines.tasks.task",
        "cognee.tasks.storage.add_data_points",
        "cognee.infrastructure.databases",
        "cognee.modules.data",
        "cognee.modules.graph",
    ):
        assert private_import in source


def test_compat_adapter_exposes_the_required_narrow_surface() -> None:
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter

    for method in (
        "aclose",
        "ensure_dataset",
        "register_data_item",
        "add_data_points",
        "verify_graph",
        "verify_vector_indexes",
        "vector_status",
        "delete_document_vectors",
        "provenance_counts",
        "resolve_graph_nodes",
        "typed_traverse",
        "read_manifest",
    ):
        assert callable(getattr(CogneeCompatibilityAdapter, method))


def test_custom_pipeline_tasks_are_declared() -> None:
    from paperos_core.adapters.cognee.pipeline_tasks import (
        academic_chunk_task,
        datapoint_mapping_task,
        semantic_enrichment_task,
        store_datapoints_task,
    )

    for task_function in (
        academic_chunk_task,
        semantic_enrichment_task,
        datapoint_mapping_task,
        store_datapoints_task,
    ):
        assert callable(task_function)


def test_search_and_pipeline_adapter_entry_points() -> None:
    from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
    from paperos_core.adapters.cognee.search import CogneeSearchAdapter

    assert callable(CogneePipelineAdapter)
    assert callable(CogneeSearchAdapter)
