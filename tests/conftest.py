"""Shared cumulative-gate setup using only the genuine-paper corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import pytest

from paperos_core.config import load_config


@pytest.fixture(scope="session")
def configured_data_dir() -> Path:
    return load_config().data_dir


@pytest.fixture(scope="session")
def corpus_manifest(configured_data_dir: Path) -> dict:
    manifest_path = configured_data_dir / "test-corpus" / "manifest.json"
    if not manifest_path.is_file():
        pytest.fail(f"Required real-paper manifest is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def real_pdf_case(configured_data_dir: Path, corpus_manifest: dict) -> tuple[Path, dict]:
    case = next(
        item for item in corpus_manifest["papers"] if item["case_id"] == "3d_gaussian_splatting"
    )
    pdf_path = configured_data_dir / "test-corpus" / "pdfs" / case["pdf_file"]
    if not pdf_path.is_file():
        pytest.fail(f"Required genuine academic PDF is missing: {pdf_path}")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest != case["sha256"]:
        pytest.fail(f"Real-paper checksum mismatch: {pdf_path}")
    return pdf_path, case


@pytest.fixture(scope="session")
def gate1_run_dir(configured_data_dir: Path) -> Path:
    requested = os.getenv("PAPEROS_TEST_RUN_ID")
    if requested and not re.fullmatch(r"[A-Za-z0-9_.-]+", requested):
        pytest.fail("PAPEROS_TEST_RUN_ID contains unsafe path characters")
    run_id = requested or f"pytest-gate1-{uuid.uuid4().hex}"
    run_dir = configured_data_dir / "test-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=bool(requested))
    return run_dir
