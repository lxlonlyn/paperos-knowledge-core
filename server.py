"""The sole normal startup entry for PaperOS Knowledge Core."""

from __future__ import annotations

import uvicorn

from paperos_core.api.app import create_app
from paperos_core.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
    )


if __name__ == "__main__":
    main()
