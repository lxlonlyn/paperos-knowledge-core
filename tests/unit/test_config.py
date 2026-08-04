from pathlib import Path

import pytest

from paperos_core.config import load_settings
from paperos_core.errors import ConfigurationError


def _write_config(path: Path, data_dir: Path, *, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[data]\ndirectory = "{data_dir}"\ndataset = "papers"\n{extra}',
        encoding="utf-8",
    )


def test_data_dir_priority_environment_over_toml(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-tests" / "paperos.toml"
    toml_dir = gate1_run_dir / "from-toml"
    env_dir = gate1_run_dir / "from-env"
    _write_config(config_path, toml_dir)

    assert load_settings(config_path, environ={}).data_dir == toml_dir.resolve()
    assert (
        load_settings(config_path, environ={"PAPEROS_DATA_DIR": str(env_dir)}).data_dir
        == env_dir.resolve()
    )


def test_invalid_config_is_actionable(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-invalid" / "paperos.toml"
    _write_config(config_path, gate1_run_dir / "invalid", extra="unknown = true\n")
    with pytest.raises(ConfigurationError) as raised:
        load_settings(config_path, environ={})
    assert raised.value.code == "configuration_error"
    assert str(config_path) in raised.value.affected


def test_blank_project_dataset_is_rejected(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-blank-dataset" / "paperos.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f'[data]\ndirectory = "{gate1_run_dir / "blank-dataset"}"\ndataset = "   "\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="data.dataset must not be blank"):
        load_settings(config_path, environ={})


def test_secrets_are_loaded_only_from_documented_environment(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-secret" / "paperos.toml"
    _write_config(
        config_path,
        gate1_run_dir / "secret-data",
        extra='[mineru]\nprovider = "cloud"\n[deepseek]\nendpoint = "https://example.test/v1"\nmodel = "deepseek/test"\n',
    )

    config = load_settings(
        config_path,
        environ={
            "MINERU_API_KEY": "mineru-environment-token",
            "DEEPSEEK_API_KEY": "deepseek-environment-token",
        },
    )

    assert config.mineru.api_key_value() == "mineru-environment-token"
    assert config.deepseek.api_key_value() == "deepseek-environment-token"
    serialized = config.model_dump_json()
    assert "mineru-environment-token" not in serialized
    assert "deepseek-environment-token" not in serialized


def test_secret_in_toml_is_rejected(gate1_run_dir: Path) -> None:
    config_path = gate1_run_dir / "config-secret-env" / "paperos.toml"
    _write_config(
        config_path,
        gate1_run_dir / "secret-env-data",
        extra='[mineru]\napi_key = "must-not-be-in-toml"\n',
    )

    with pytest.raises(ConfigurationError, match="API keys"):
        load_settings(config_path, environ={})
