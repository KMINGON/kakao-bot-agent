"""Fast ACK webhook service; no model work runs in this process."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from psycopg_pool import PoolTimeout
from procrastinate.exceptions import ConnectorException

from .contracts import normalize_message
from .settings import Settings
from .storage import DuplicateConflict, Storage

logger = logging.getLogger(__name__)


def execution_snapshot(settings: Settings) -> dict[str, Any]:
    from .runtime import load_pack
    snapshot = dict(load_pack(settings.packs_root, settings.active_pack).snapshot)
    snapshot.update(runtime_revision=settings.runtime_revision, model_name=settings.model_name, mode=settings.mode)
    return snapshot


def create_app(settings: Settings | None = None, *, storage: Any = None, task: Any = None,
               snapshot: dict[str, Any] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        if len(settings.webhook_secret) < 24:
            raise ValueError("KAKAO_WEBHOOK_SECRET must contain at least 24 characters")
        if not settings.ops_token or len(settings.ops_token) < 24:
            raise ValueError("OPS_TOKEN must contain at least 24 characters")
        app.state.snapshot = snapshot if snapshot is not None else execution_snapshot(settings)
        app.state.storage = storage or Storage(settings)
        if storage is None:
            await app.state.storage.open()
        if task is None:
            from .worker import process_event, queue_app
            app.state.task = process_event
            await queue_app.open_async()
        else:
            app.state.task = task
        try:
            yield
        finally:
            if task is None:
                await queue_app.close_async()
            if storage is None:
                await app.state.storage.close()

    app = FastAPI(title="Kakao Agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authenticate(request: Request, *, ops: bool = False) -> None:
        supplied = request.headers.get("Authorization", "") if ops else request.headers.get(settings.webhook_header_name, "")
        expected = f"Bearer {settings.ops_token}" if ops else settings.webhook_secret
        if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "unauthorized")

    @app.post("/webhooks/kakao")
    async def webhook(request: Request) -> Response:
        authenticate(request)
        started = time.monotonic()
        try:
            async with asyncio.timeout(settings.admission_timeout_seconds):
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > settings.max_body_bytes:
                        raise HTTPException(413, "body too large")
                payload = json.loads(body)
                event = normalize_message(payload, channel_only=settings.channel_only)
                if event is None:
                    return Response(status_code=204)
                admission = await app.state.storage.admit(event, app.state.snapshot, app.state.task)
                logger.info("webhook_admitted event_id=%s duplicate=%s elapsed_ms=%.1f",
                            admission["event_id"], admission["duplicate"], (time.monotonic() - started) * 1000)
                return Response(status_code=204)
        except (ValidationError, ValueError, UnicodeError) as error:
            if isinstance(error, DuplicateConflict):
                raise HTTPException(409, "conflicting event ID") from None
            raise HTTPException(400, "invalid event") from None
        except (TimeoutError, psycopg.Error, PoolTimeout, ConnectorException):
            logger.warning("webhook_admission_unavailable elapsed_ms=%.1f", (time.monotonic() - started) * 1000)
            raise HTTPException(503, "temporarily unavailable") from None

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        try:
            async with asyncio.timeout(0.7):
                if await app.state.storage.ready():
                    return {"status": "ready"}
        except (TimeoutError, psycopg.Error, PoolTimeout):
            pass
        raise HTTPException(503, "not ready")

    @app.get("/ops/status")
    async def status(request: Request) -> dict[str, Any]:
        authenticate(request, ops=True)
        async with asyncio.timeout(3):
            return await app.state.storage.status()

    return app


app = create_app()
