"""Shared FastAPI dependency accessors."""

from typing import cast

from fastapi import Request

from paperos_core.application import Application


def get_application(request: Request) -> Application:
    return cast(Application, request.app.state.paperos)
