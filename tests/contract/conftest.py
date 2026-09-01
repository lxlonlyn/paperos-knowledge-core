"""Pytest collection guard for the complete contract directory."""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT_ROOT = Path(__file__).resolve().parent


def _collecting_contract_directory(config: pytest.Config) -> bool:
    for raw_argument in config.args:
        target = str(raw_argument).split("::", maxsplit=1)[0]
        if Path(target).resolve(strict=False) == CONTRACT_ROOT:
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Fail a directory-level contract run if any test module collects nothing."""

    if not _collecting_contract_directory(config):
        return
    expected = {path.resolve() for path in CONTRACT_ROOT.glob("test_*.py")}
    collected = {Path(str(item.path)).resolve() for item in items}
    missing = sorted(path.name for path in expected - collected)
    if missing:
        raise pytest.UsageError(
            "Contract modules without collected tests: " + ", ".join(missing)
        )
