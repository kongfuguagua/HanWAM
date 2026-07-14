"""FastAPI app factory for the generic API servicer."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import time

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool

from .config import ApiServiceConfig
from .controllers.base import ControllerAdapter
from .errors import ServiceError, service_error_handler, validation_error_handler
from .observability import log_event
from .schemas import HealthResponse, PlanRequest, PlanResponse, ReadyResponse, ResetResponse


LOGGER = logging.getLogger("servicer.api")


def create_app(config: ApiServiceConfig, controller: ControllerAdapter) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log_event(
            LOGGER,
            "service.started",
            service=config.service.as_dict(),
            logging=config.logging.as_dict(),
            controller=controller.metadata(),
        )
        yield
        log_event(LOGGER, "service.stopped", controller_type=controller.controller_type)

    app = FastAPI(title="Control API Servicer", version="1.0.0", lifespan=lifespan)
    app.state.log_payloads = config.logging.payloads
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.middleware("http")
    async def log_http(request: Request, call_next):
        started = time.perf_counter()
        client = request.client.host if request.client is not None else None
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event(
                LOGGER,
                "http.error",
                level=logging.ERROR,
                method=request.method,
                path=request.url.path,
                client=client,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error=exc,
            )
            raise
        log_event(
            LOGGER,
            "http.completed",
            method=request.method,
            path=request.url.path,
            client=client,
            status_code=response.status_code,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        return response

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    def readyz() -> ReadyResponse:
        metadata = controller.metadata()
        return ReadyResponse(
            status="ready",
            controller_type=controller.controller_type,
            model_id=str(metadata["model_id"]),
        )

    @app.get("/v1/metadata")
    def metadata() -> dict:
        return {
            "status": "ok",
            "service": config.service.as_dict(),
            "logging": config.logging.as_dict(),
            "controller": controller.metadata(),
        }

    @app.post("/v1/plan", response_model=PlanResponse)
    async def plan(http_request: Request, request: PlanRequest) -> PlanResponse:
        raw_body = await http_request.body()
        request_body = json.loads(raw_body)
        body_sha256 = hashlib.sha256(raw_body).hexdigest()
        if config.logging.payloads:
            log_event(
                LOGGER,
                "plan.request",
                request_id=request.request_id,
                unit_id=request.unit_id,
                body_sha256=body_sha256,
                body_bytes=len(raw_body),
                request_body=request_body,
                validated_request=request.model_dump(mode="json"),
            )
        try:
            response = await run_in_threadpool(controller.plan, request)
        except ServiceError as exc:
            if request.return_debug:
                exc.details.update(
                    {
                        "request_body": request_body,
                        "validated_request": request.model_dump(mode="json"),
                        "request_body_sha256": body_sha256,
                        "request_body_bytes": len(raw_body),
                    }
                )
            raise
        if request.return_debug:
            response.debug.update(
                {
                    "request_body": request_body,
                    "validated_request": request.model_dump(mode="json"),
                    "request_body_sha256": body_sha256,
                    "request_body_bytes": len(raw_body),
                }
            )
        if config.logging.payloads:
            log_event(
                LOGGER,
                "plan.response",
                request_id=request.request_id,
                unit_id=request.unit_id,
                response=response.model_dump(mode="json"),
            )
        return response

    @app.post("/v1/reset", response_model=ResetResponse)
    def reset() -> ResetResponse:
        response = controller.reset()
        log_event(LOGGER, "controller.reset", response=response)
        return response

    return app
