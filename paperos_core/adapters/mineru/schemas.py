"""Provider-neutral MinerU task and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MinerUTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    task_id: str
    state: str
    backend: str
    data_id: str
    result_archive_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    progress: dict[str, Any] | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class MinerUParseResult:
    provider: str
    provider_task_id: str
    backend: str
    archive_bytes: bytes
    final_metadata: dict[str, Any]
    poll_history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
