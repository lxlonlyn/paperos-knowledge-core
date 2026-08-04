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

from paperos_core.adapters.mineru.providers import DEFAULT_MINERU_CLOUD_ENDPOINT
from paperos_core.config import load_settings, resolve_local_model_path


async def diagnose() -> dict[str, object]:
    settings = load_settings()
    checks: dict[str, object] = {
        "python": {"version": platform.python_version(), "compatible": sys.version_info[:2] in {(3, 11), (3, 12)}},
        "configuration": {"valid": True, "path": str(settings.config_path)},
    }
    try:
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
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        checks["node"] = {"compatible": False, "error": str(exc)}

    models: dict[str, object] = {}
    for name, configured in (
        ("embedding", settings.local_inference.embedding.model_path),
        ("reranker", settings.local_inference.reranker.model_path),
        ("query_expansion", settings.local_inference.query_expansion.model_path),
    ):
        path = resolve_local_model_path(settings, configured)
        models[name] = {"exists": path.is_file(), "path": str(path.resolve(strict=False))}
    checks["models"] = models

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
    checks["deepseek"] = {
        "endpoint": settings.deepseek.endpoint,
        "model": settings.deepseek.model,
        "api_key_configured": bool(settings.deepseek.api_key_value()),
    }
    checks["data_directory"] = {
        "path": str(settings.data_dir),
        "exists": settings.data_dir.is_dir(),
        "readable": os.access(settings.data_dir, os.R_OK),
        "writable": os.access(settings.data_dir, os.W_OK),
    }
    checks["cognee"] = {
        "system": settings.cognee.system_database.exists(),
        "vector": settings.cognee.vector_database.exists(),
        "graph": settings.cognee.graph_database.exists(),
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
