"""Explicit error contracts for the generic API servicer."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .observability import json_safe, log_event


LOGGER = logging.getLogger("servicer.errors")


class ServiceError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.error_code = str(error_code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.details = details or {}
        self.request_id = request_id


def _json_safe(value: Any) -> Any:
    return json_safe(value)


def error_payload(
    error_code: str,
    message: str,
    *,
    request_id: str | None,
    details: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "request_id": request_id,
        "error_code": error_code,
        "message": message,
        "details": _json_safe(details or {}),
    }


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    log_event(
        LOGGER,
        "service.error",
        level=logging.WARNING if exc.status_code < 500 else logging.ERROR,
        method=request.method,
        path=request.url.path,
        status_code=exc.status_code,
        request_id=exc.request_id,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            exc.error_code,
            exc.message,
            request_id=exc.request_id,
            details=exc.details,
        ),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = getattr(exc, "body", None)
    request_id = body.get("request_id") if isinstance(body, dict) else None
    details: dict[str, Any] = {"errors": exc.errors()}
    if isinstance(body, dict) and body.get("return_debug") is True:
        details["request_body"] = body
    log_fields: dict[str, Any] = {
        "method": request.method,
        "path": request.url.path,
        "status_code": 422,
        "request_id": request_id,
        "errors": exc.errors(),
    }
    if getattr(request.app.state, "log_payloads", False):
        log_fields["request_body"] = body
    log_event(LOGGER, "request.validation_error", level=logging.WARNING, **log_fields)
    return JSONResponse(
        status_code=422,
        content=error_payload(
            "VALIDATION_ERROR",
            "request validation failed",
            request_id=request_id,
            details=details,
        ),
    )
