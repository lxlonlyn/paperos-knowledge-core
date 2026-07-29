from pathlib import Path

import pytest

from paperos_core.config import load_config
from paperos_core.errors import ConfigurationError


def _write_config(path: Path, data_dir: Path, *, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[project]\ndata_dir = "{data_dir}"\ndataset = "papers"\n{extra}',
        encoding="utf-8",
    )


def test_data_dir_priority_cli_env_toml(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-tests" / "paperos.toml"
    toml_dir = gate1_run_dir / "from-toml"
    env_dir = gate1_run_dir / "from-env"
    cli_dir = gate1_run_dir / "from-cli"
    _write_config(config_path, toml_dir)

    assert load_config(config_path, environ={}).data_dir == toml_dir.resolve()
    assert (
        load_config(config_path, environ={"PAPEROS_DATA_DIR": str(env_dir)}).data_dir
        == env_dir.resolve()
    )
    assert (
        load_config(
            config_path,
            data_dir=cli_dir,
            environ={"PAPEROS_DATA_DIR": str(env_dir)},
        ).data_dir
        == cli_dir.resolve()
    )


def test_invalid_config_is_actionable(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-invalid" / "paperos.toml"
    _write_config(config_path, gate1_run_dir / "invalid", extra="unknown = true\n")
    with pytest.raises(ConfigurationError) as raised:
        load_config(config_path, environ={})
    assert raised.value.code == "configuration_error"
    assert str(config_path) in raised.value.affected


def test_mineru_api_key_can_persist_in_project_config(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-secret" / "paperos.toml"
    _write_config(
        config_path,
        gate1_run_dir / "secret-data",
        extra='[mineru_ocr]\napi_key = "persistent-token"\n',
    )

    config = load_config(config_path, environ={})

    assert config.mineru_ocr.api_key_value() == "persistent-token"
    assert "persistent-token" not in repr(config.mineru_ocr)
    assert "persistent-token" not in config.model_dump_json()


def test_mineru_environment_key_overrides_project_config(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-secret-env" / "paperos.toml"
    _write_config(
        config_path,
        gate1_run_dir / "secret-env-data",
        extra=('[mineru_ocr]\napi_key_env = "CUSTOM_MINERU_KEY"\napi_key = "persistent-token"\n'),
    )

    config = load_config(config_path, environ={"CUSTOM_MINERU_KEY": "environment-token"})

    assert config.mineru_ocr.api_key_value() == "environment-token"
