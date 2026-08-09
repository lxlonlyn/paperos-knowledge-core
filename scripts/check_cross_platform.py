"""Static cross-platform smoke checks that use only temporary state."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import paperos_core
import paperos_core.ingestion
import paperos_core.jobs
import paperos_core.runtime.local_inference
from paperos_core.config import DEFAULT_CONFIG_PATH, load_settings
from paperos_core.locations import CONFIG_ROOT


def main() -> None:
    if DEFAULT_CONFIG_PATH != CONFIG_ROOT / "paperos.toml":
        raise RuntimeError("Default configuration is not anchored to CONFIG_ROOT.")
    with tempfile.TemporaryDirectory(prefix="paperos-compat-") as temporary:
        root = Path(temporary)
        config_path = root / "machine" / "paperos.toml"
        config_path.parent.mkdir()
        config_path.write_text(
            '[data]\ndirectory = "runtime-data"\n',
            encoding="utf-8",
        )
        original_cwd = os.getcwd()
        try:
            os.chdir(root)
            default_settings = load_settings()
            settings = load_settings(config_path)
        finally:
            os.chdir(original_cwd)
        if default_settings.config_path != DEFAULT_CONFIG_PATH:
            raise RuntimeError("Default config changed with the shell working directory.")
        expected_data = (config_path.parent / "runtime-data").resolve(strict=False)
        if settings.data_dir != expected_data:
            raise RuntimeError("Relative data path was not resolved from paperos.toml.")

    _ = (
        paperos_core,
        paperos_core.jobs,
        paperos_core.ingestion,
        paperos_core.runtime.local_inference,
    )
    print("cross-platform static checks passed")


if __name__ == "__main__":
    main()
