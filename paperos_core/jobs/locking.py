"""Advisory single-process lock for the local Worker lifecycle."""

from __future__ import annotations

import fcntl
from types import TracebackType
from typing import IO, Self

from paperos_core.errors import JobQueueError
from paperos_core.paths import DataPaths


class WorkerLifecycleLock:
    def __init__(self, paths: DataPaths) -> None:
        self.path = paths.jobs / "worker.lock"
        self._stream: IO[str] | None = None

    def __enter__(self) -> Self:
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise JobQueueError(
                "Another PaperOS Worker already owns the lifecycle lock.",
                affected=self.path,
            ) from exc
        self._stream = stream
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None
