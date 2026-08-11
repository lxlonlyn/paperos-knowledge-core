"""Create the single machine-local PaperOS TOML without overwriting it."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.locations import CONFIG_ROOT


def _copy_if_missing(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
    except FileExistsError:
        return "kept existing"
    return "created from example"


def main() -> None:
    copies = ((CONFIG_ROOT / "paperos.example.toml", CONFIG_ROOT / "paperos.toml"),)
    for source, target in copies:
        if not source.is_file():
            raise SystemExit(f"Required example file is missing: {source}")
        status = _copy_if_missing(source, target)
        print(f"{target}: {status}")


if __name__ == "__main__":
    main()
