"""Single-replica durable worker, with exclusive ownership during recovery."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import procrastinate
import psycopg

from .contracts import render_callback
from .kakao import KakaoClient
from .settings import Settings
from .storage import Storage

settings = Settings.from_env()
queue_app = procrastinate.App(connector=procrastinate.PsycopgConnector(conninfo=settings.database_url, min_size=1, max_size=4))
_storage: Storage | None = None
_runtime: Any = None
_kakao: KakaoClient | None = None
logger = logging.getLogger(__name__)
CALLBACK_RESERVE_SECONDS = 12
CALLBACK_TIMEOUT_SECONDS = 10


@queue_app.task(name="kakao.process_event", queue="kakao", retry=procrastinate.RetryStrategy(max_attempts=3, wait=1))
async def process_event(event_id: str) -> None:
    assert _storage is not None and _kakao is not None
    event = await _storage.get_event(event_id)
    if not event or event["delivery_status"] in {"sent", "unknown", "rejected", "expired"}:
        return
    if event["delivery_status"] == "sending":
        await _storage.finish_send(event_id, "unknown", "previous_attempt_interrupted")
        return
    remaining = (event["callback_deadline_at"] - datetime.now(UTC)).total_seconds()
    if remaining <= CALLBACK_TIMEOUT_SECONDS:
        await _storage.expire(event_id)
        return
    event = await _storage.start_run(event_id)
    if not event:
        return
    if not event["response_payload"]:
        try:
            spec = event["execution_spec"]
            if spec.get("runtime_revision") != settings.runtime_revision or spec.get("model_name") != settings.model_name:
                raise RuntimeError("execution_revision_mismatch")
            if spec.get("mode") == "echo":
                text = f"연결 확인: {event['input_envelope']['content']}"
                usage, trace = {}, []
            else:
                if _runtime is None:
                    raise RuntimeError("agent_runtime_not_available")
                run_deadline = event["callback_deadline_at"] - timedelta(seconds=CALLBACK_RESERVE_SECONDS)
                run_remaining = min(settings.run_timeout_seconds, (run_deadline - datetime.now(UTC)).total_seconds())
                if run_remaining <= 0:
                    raise TimeoutError("Queue exhausted response budget")
                async with asyncio.timeout(run_remaining):
                    result = await _runtime.run(
                        text=event["input_envelope"]["content"], thread_id=event["graph_thread_id"],
                        event_id=event_id, pack_ref=spec, deadline=run_deadline,
                    )
                text, usage, trace = result.text, result.usage, result.trace
            await _storage.save_response(event_id, render_callback(text), usage=usage, trace=trace)
        except asyncio.CancelledError:
            # Preserve the checkpoint for the next owner; never append the input again.
            raise
        except Exception as error:
            code = type(error).__name__
            logger.error("event_run_failed event_id=%s code=%s", event_id, code)
            if _runtime is not None:
                try:
                    async with asyncio.timeout(1):
                        await _runtime.abandon(thread_id=event["graph_thread_id"], event_id=event_id)
                except Exception:
                    logger.error("event_abandon_failed event_id=%s", event_id)
            failure_text = "이번 요청의 결과를 끝까지 확인하지 못했습니다. 대화와 확인된 기록은 유지됩니다. 접수를 시도한 경우 결과 미확인이며, 이미 처리됐을 수 있어 완료·실패를 단정할 수 없습니다."
            await _storage.save_response(
                event_id,
                render_callback(failure_text),
                failed=True, code=code, usage=getattr(error, "usage", {}), trace=getattr(error, "trace", []),
            )
    attempt = await _storage.begin_send(event_id)
    if not attempt:
        await _storage.expire(event_id)
        return
    try:
        remaining = (event["callback_deadline_at"] - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            await _storage.finish_send(event_id, "unknown", "deadline_before_send")
            return
        async with asyncio.timeout(min(CALLBACK_TIMEOUT_SECONDS, remaining)):
            result = await _kakao.send(attempt["callback_token"], attempt["response_payload"])
        await _storage.finish_send(event_id, result.status, result.code, result.counts)
        logger.info("event_delivery event_id=%s status=%s", event_id, result.status)
    except TimeoutError:
        await _storage.finish_send(event_id, "unknown", "callback_timeout")
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await asyncio.shield(_storage.finish_send(event_id, "unknown", "cancelled_during_send"))
        raise


async def watch_owner(connection: psycopg.AsyncConnection) -> None:
    while True:
        async with asyncio.timeout(5):
            await connection.execute("SELECT 1")
        await asyncio.sleep(3)


async def main() -> None:
    global _storage, _runtime, _kakao
    if not settings.kakao_bot_token:
        raise ValueError("KAKAO_BOT_TOKEN or KAKAO_BOT_TOKEN_FILE is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # A live PostgreSQL session lock proves the previous worker is gone before recovery.
    owner_key = int.from_bytes(hashlib.sha256(f"kakao-worker:{settings.channel_scope}".encode()).digest()[:8], "big") & ((1 << 63) - 1)
    async with contextlib.AsyncExitStack() as stack:
        owner = await stack.enter_async_context(await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True))
        cursor = await owner.execute("SELECT pg_try_advisory_lock(%s)", (owner_key,))
        if not (await cursor.fetchone())[0]:
            raise RuntimeError("another_worker_owns_this_channel")
        _storage = Storage(settings)
        await _storage.open()
        stack.push_async_callback(_storage.close)
        _kakao = KakaoClient(settings.kakao_bot_token)
        stack.push_async_callback(_kakao.close)
        await stack.enter_async_context(queue_app.open_async())
        if settings.mode == "agent":
            from .runtime import AgentRuntime, RuntimeConfig
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY or OPENAI_API_KEY_FILE is required in agent mode")
            config = RuntimeConfig(
                database_url=settings.database_url, openai_api_key=settings.openai_api_key,
                model_name=settings.model_name, packs_root=settings.packs_root,
                max_model_calls=settings.max_model_calls, max_tool_calls=settings.max_tool_calls,
                max_input_tokens=settings.max_input_tokens, max_output_tokens=settings.max_output_tokens,
                summary_trigger_tokens=settings.summary_trigger_tokens,
                run_timeout_seconds=settings.run_timeout_seconds,
            )
            _runtime = await stack.enter_async_context(AgentRuntime(config))
        recovery = await _storage.recover(queue_app, process_event)
        logger.info("worker_recovery %s", json.dumps(recovery))
        watchdog = asyncio.create_task(watch_owner(owner))
        worker = asyncio.create_task(queue_app.run_worker_async(
            queues=["kakao"], concurrency=settings.worker_concurrency, shutdown_graceful_timeout=20,
            fetch_job_polling_interval=0.5,
        ))
        try:
            done, _ = await asyncio.wait([watchdog, worker], return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                await finished
        finally:
            for task in (worker, watchdog):
                if not task.done():
                    task.cancel()
            await asyncio.gather(worker, watchdog, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
