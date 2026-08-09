"""Stable repository locations independent of the process working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
SERVICES_ROOT = PROJECT_ROOT / "services"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"

__all__ = ["CONFIG_ROOT", "PROJECT_ROOT", "PROMPTS_ROOT", "SERVICES_ROOT"]
