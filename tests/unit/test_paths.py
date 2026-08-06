from pathlib import Path

from paperos_core.paths import build_data_paths, initialize_data_paths


def test_data_paths_are_centralized_and_initialized(gate1_run_dir: Path) -> None:
    root = gate1_run_dir / "paths"
    paths = initialize_data_paths(build_data_paths(root))
    assert paths.root == root.resolve()
    assert paths.raw == paths.root / "raw"
    assert paths.test_corpus == paths.root / "test-corpus"
    assert paths.test_runs == paths.root / "test-runs"
    assert paths.registry_db == paths.root / "jobs" / "registry.sqlite3"
    for directory in paths.runtime_directories():
        assert directory.is_dir()
        assert directory.is_relative_to(paths.root)


def test_services_root_is_derived_from_module_location() -> None:
    from paperos_core.runtime.local_inference.runtime import (
        PROJECT_ROOT,
        SERVICES_ROOT,
    )

    assert PROJECT_ROOT.name == "paperos-knowledge-core"
    assert SERVICES_ROOT == PROJECT_ROOT / "services"
    assert (SERVICES_ROOT / "local_models" / "package.json").is_file()
