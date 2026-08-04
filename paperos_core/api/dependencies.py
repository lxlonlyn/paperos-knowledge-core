"""Shared FastAPI dependency accessors."""

from typing import Annotated, cast

from fastapi import Depends, Request

from paperos_core.application import Application


def get_application(request: Request) -> Application:
    return cast(Application, request.app.state.paperos)


ApplicationDep = Annotated[Application, Depends(get_application)]
