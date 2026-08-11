"""Initialize local storage and validate pre-provided runtime assets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.configurator import CogneeConfigurator
from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader
from paperos_core.config import load_settings, resolve_local_model_path
from paperos_core.locations import SERVICES_ROOT
from paperos_core.paths import build_data_paths
from paperos_core.runtime.local_inference.runtime import local_runtime_usage
from paperos_core.storage import StorageInitializer


def main() -> None:
    settings = load_settings()
    paths = build_data_paths(settings.data_dir)
    CogneeConfigurator().apply(settings, paths)
    storage = StorageInitializer(paths)
    storage.initialize()
    checks: list[dict[str, object]] = []
    usage = local_runtime_usage(settings, CogneeRuntimeConfigReader())

    try:
        if not usage.required:
            raise RuntimeError("local runtime is not required")
        completed = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True, timeout=10
        )
        version = completed.stdout.strip()
        major = int(version.lstrip("v").split(".", 1)[0])
        checks.append(
            {"name": "node", "ok": major >= 22, "version": version, "required": ">=22"}
        )
    except RuntimeError as exc:
        checks.append({"name": "node", "ok": True, "required": False, "skipped": str(exc)})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        checks.append({"name": "node", "ok": False, "error": str(exc)})

    service_root = SERVICES_ROOT / "local_models"
    node_modules = service_root / "node_modules"
    node_entry = service_root / "dist" / "server.js"
    remediation = ["cd services/local_models", "npm ci", "npm run build"]
    checks.append(
        {
            "name": "node_modules",
            "ok": node_modules.is_dir() or not usage.required,
            "present": node_modules.is_dir(),
            "required": usage.required,
            "path": str(node_modules),
            "remediation": remediation,
        }
    )
    checks.append(
        {
            "name": "node_entry",
            "ok": node_entry.is_file() or not usage.required,
            "present": node_entry.is_file(),
            "required": usage.required,
            "path": str(node_entry),
            "remediation": remediation,
        }
    )
    model_checks: list[tuple[str, Path]] = []
    if usage.embedding:
        model_checks.append(("embedding_model", settings.local_inference.embedding_model_path))
    if usage.reranker:
        model_checks.append(("reranker_model", settings.local_inference.reranker_model_path))
    for name, configured in model_checks:
        path = resolve_local_model_path(settings, configured)
        item: dict[str, object] = {"name": name, "ok": path.is_file(), "path": str(path)}
        checks.append(item)
    checks.append({"name": "local_runtime_activation", "ok": True, "required": usage.required, "embedding": usage.embedding, "reranker": usage.reranker, "cuda_devices": settings.local_inference.cuda_devices})
    storage_status = storage.validate()
    checks.append(
        {
            "name": "storage",
            "ok": storage_status.valid,
            "registry": str(storage_status.registry_database),
            "lexical": str(storage_status.lexical_database),
            "missing_tables": list(storage_status.missing_tables),
        }
    )
    payload = {"ok": all(bool(item["ok"]) for item in checks), "checks": checks}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
