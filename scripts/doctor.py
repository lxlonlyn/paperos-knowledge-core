"""Read-only diagnostics for PaperOS and its configured external dependencies."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader
from paperos_core.adapters.mineru.providers import DEFAULT_MINERU_CLOUD_ENDPOINT
from paperos_core.config import load_settings, resolve_local_model_path
from paperos_core.runtime.local_inference.runtime import local_runtime_usage


class _RuntimeNotRequired(Exception):
    pass


async def diagnose() -> dict[str, object]:
    settings = load_settings()
    cognee_reader = CogneeRuntimeConfigReader()
    cognee = cognee_reader.read()
    usage = local_runtime_usage(settings, cognee_reader)
    checks: dict[str, object] = {
        "python": {"version": platform.python_version(), "compatible": sys.version_info[:2] in {(3, 11), (3, 12)}},
        "configuration": {"valid": True, "path": str(settings.config_path)},
    }
    try:
        if not usage.required:
            raise _RuntimeNotRequired
        process = await asyncio.create_subprocess_exec(
            "node",
            "--version",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        if process.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        node = stdout.decode().strip()
        checks["node"] = {
            "version": node,
            "compatible": int(node[1:].split(".")[0]) >= 22,
        }
    except _RuntimeNotRequired:
        checks["node"] = {"required": False, "skipped": True}
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        checks["node"] = {"compatible": False, "error": str(exc)}

    models: dict[str, object] = {}
    model_checks: list[tuple[str, Path]] = []
    if usage.embedding:
        model_checks.append(("embedding", settings.local_inference.embedding_model_path))
    if usage.reranker:
        model_checks.append(("reranker", settings.local_inference.reranker_model_path))
    for name, configured in model_checks:
        path = resolve_local_model_path(settings, configured)
        models[name] = {"exists": path.is_file(), "path": str(path.resolve(strict=False))}
    checks["models"] = models
    checks["local_runtime"] = {
        "enabled": settings.local_inference.enabled,
        "required": usage.required,
        "embedding_used": usage.embedding,
        "reranker_used": usage.reranker,
        "embedding_endpoint_matches": cognee.embedding_targets(
            settings.local_inference.host, settings.local_inference.port
        ),
        "embedding_provider": cognee.embedding_provider,
        "embedding_model": cognee.embedding_model,
        "cuda_devices": settings.local_inference.cuda_devices,
    }

    mineru_endpoint = (settings.mineru.endpoint or DEFAULT_MINERU_CLOUD_ENDPOINT).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
            response = await client.get(mineru_endpoint)
        checks["mineru"] = {
            "reachable": True,
            "status_code": response.status_code,
            "api_key_configured": bool(settings.mineru.api_key_value()),
        }
    except httpx.HTTPError as exc:
        checks["mineru"] = {"reachable": False, "error": str(exc)}
    checks["llm"] = {
        "provider": cognee.llm_provider,
        "endpoint": cognee.llm_endpoint,
        "model": cognee.llm_model,
    }
    checks["data_directory"] = {
        "path": str(settings.data_dir),
        "exists": settings.data_dir.is_dir(),
        "readable": os.access(settings.data_dir, os.R_OK),
        "writable": os.access(settings.data_dir, os.W_OK),
    }
    checks["cognee"] = {
        "db_provider": cognee.db_provider,
        "db_path": cognee.db_path,
        "vector_provider": cognee.vector_db_provider,
        "vector_url": cognee.vector_db_url,
        "graph_provider": cognee.graph_database_provider,
        "graph_path": cognee.graph_file_path,
    }
    checks["ports"] = {
        "api": {
            "host": settings.api.host,
            "port": settings.api.port,
            "occupied": _port_occupied(settings.api.host, settings.api.port),
        },
        "local_inference": {
            "host": settings.local_inference.host,
            "port": settings.local_inference.port,
            "occupied": _port_occupied(
                settings.local_inference.host, settings.local_inference.port
            ),
        },
    }
    return checks


def _port_occupied(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((probe_host, port)) == 0


def main() -> None:
    print(json.dumps(asyncio.run(diagnose()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
