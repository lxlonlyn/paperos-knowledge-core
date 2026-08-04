"""Initialize local storage and validate pre-provided runtime assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import load_settings, resolve_local_model_path
from paperos_core.paths import build_data_paths
from paperos_core.storage import StorageInitializer


def main() -> None:
    settings = load_settings()
    paths = build_data_paths(settings.data_dir)
    storage = StorageInitializer(paths)
    storage.initialize()
    checks: list[dict[str, object]] = []

    try:
        completed = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True, timeout=10
        )
        version = completed.stdout.strip()
        major = int(version.lstrip("v").split(".", 1)[0])
        checks.append(
            {"name": "node", "ok": major >= 22, "version": version, "required": ">=22"}
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        checks.append({"name": "node", "ok": False, "error": str(exc)})

    node_entry = REPOSITORY_ROOT / "services" / "local_models" / "dist" / "server.js"
    checks.append({"name": "node_entry", "ok": node_entry.is_file(), "path": str(node_entry)})
    for name, model in (
        ("embedding_model", settings.local_inference.embedding),
        ("reranker_model", settings.local_inference.reranker),
        ("query_expansion_model", settings.local_inference.query_expansion),
    ):
        path = resolve_local_model_path(settings, model.model_path)
        item: dict[str, object] = {"name": name, "ok": path.is_file(), "path": str(path)}
        if path.is_file() and model.sha256:
            actual = _sha256(path)
            item.update(
                {
                    "ok": actual == model.sha256.casefold(),
                    "expected_sha256": model.sha256,
                    "actual_sha256": actual,
                }
            )
        checks.append(item)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
