"""Whole-event deadline behavior without PostgreSQL, model calls, or Kakao."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kakao_bot import worker
from kakao_bot.contracts import render_callback
from kakao_bot.kakao import DeliveryResult
from kakao_bot.settings import Settings


@pytest.fixture
def harness(monkeypatch):
    received = datetime(2026, 10, 31, 3, tzinfo=UTC)
    clock = SimpleNamespace(now=received)
    cfg = Settings(database_url="postgresql://unused", run_timeout_seconds=40)
    event = {
        "delivery_status": "not_ready", "response_payload": None,
        "received_at": received, "callback_deadline_at": received + timedelta(seconds=55),
        "execution_spec": {"runtime_revision": cfg.runtime_revision,
                           "model_name": cfg.model_name, "mode": "agent"},
        "input_envelope": {"content": "접수 결과를 알려주세요"},
        "graph_thread_id": "kept-conversation", "callback_token": "never-send",
    }
    saved = []

    async def save_response(event_id, payload, **kwargs):
        event.update(response_payload=payload, delivery_status="ready")
        saved.append(kwargs)

    async def finish_send(event_id, status, code, counts=None):
        event["delivery_status"] = status

    async def expire(event_id):
        event["delivery_status"] = "expired"

    store = SimpleNamespace(
        get_event=AsyncMock(return_value=event), start_run=AsyncMock(return_value=event),
        save_response=AsyncMock(side_effect=save_response), begin_send=AsyncMock(return_value=event),
        finish_send=AsyncMock(side_effect=finish_send), expire=AsyncMock(side_effect=expire),
    )
    runtime = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(text="접수 결과를 확인했습니다.", usage={}, trace=[])),
        abandon=AsyncMock(return_value=True),
    )
    kakao = SimpleNamespace(send=AsyncMock(return_value=DeliveryResult("sent", "ok", {})))
    monkeypatch.setattr(worker, "settings", cfg)
    monkeypatch.setattr(worker, "datetime", SimpleNamespace(now=lambda zone: clock.now))
    monkeypatch.setattr(worker, "_storage", store)
    monkeypatch.setattr(worker, "_runtime", runtime)
    monkeypatch.setattr(worker, "_kakao", kakao)
    return SimpleNamespace(event=event, clock=clock, store=store, runtime=runtime,
                           kakao=kakao, saved=saved)


async def test_queue_and_start_run_time_reduce_generation_budget(harness, monkeypatch):
    h = harness
    h.clock.now += timedelta(seconds=20)

    async def delayed_start(event_id):
        h.clock.now += timedelta(seconds=7)
        return h.event

    timeouts = []

    @asynccontextmanager
    async def capture_timeout(seconds):
        timeouts.append(seconds)
        yield

    h.store.start_run.side_effect = delayed_start
    monkeypatch.setattr(worker.asyncio, "timeout", capture_timeout)
    await worker.process_event.func("event")
    assert h.runtime.run.await_args.kwargs["deadline"] == h.event["callback_deadline_at"] - timedelta(seconds=12)
    assert 16 in timeouts  # 55 seconds - 20 in queue - 7 claiming - 12 reserved.
    assert max(timeouts) <= 16
    h.kakao.send.assert_awaited_once()


async def test_queue_exhausted_generation_budget_still_sends_fallback(harness):
    h = harness
    h.clock.now = h.event["callback_deadline_at"] - timedelta(seconds=11)
    await worker.process_event.func("event")
    h.runtime.run.assert_not_awaited()
    h.kakao.send.assert_awaited_once()
    assert h.event["delivery_status"] == "sent"
    assert h.saved[0]["failed"]


async def test_queue_without_callback_budget_expires_without_calls(harness):
    h = harness
    h.clock.now = h.event["callback_deadline_at"] - timedelta(seconds=10)
    await worker.process_event.func("event")
    h.runtime.run.assert_not_awaited()
    h.kakao.send.assert_not_awaited()
    assert h.event["delivery_status"] == "expired"


async def test_failed_run_preserves_context_and_reports_unconfirmed_outcome(harness):
    h = harness
    h.runtime.run.side_effect = TimeoutError("model deadline")
    await worker.process_event.func("event")
    h.runtime.abandon.assert_awaited_once_with(thread_id="kept-conversation", event_id="event")
    text = h.event["response_payload"]["skillResponse"]["template"]["outputs"][0]["simpleText"]["text"]
    assert "초기화" not in text
    assert any(term in text for term in ["미확인", "확인하지 못", "확인되지", "알 수"])
    assert len(text) <= 500
    assert h.saved[0]["failed"]
    h.kakao.send.assert_awaited_once()


async def test_callback_has_total_timeout_and_is_never_retried(harness, monkeypatch):
    h = harness
    h.event.update(delivery_status="ready", response_payload=render_callback("준비된 답변"))
    actual_timeout = asyncio.timeout
    timeouts = []

    def shorten_timeout(seconds):
        timeouts.append(seconds)
        return actual_timeout(0.005)

    async def hung_send(*args):
        await asyncio.sleep(0.05)
        return DeliveryResult("sent", "ok", {})

    monkeypatch.setattr(worker.asyncio, "timeout", shorten_timeout)
    h.kakao.send.side_effect = hung_send
    await worker.process_event.func("event")
    await worker.process_event.func("event")
    assert 10 in timeouts
    assert h.event["delivery_status"] == "unknown"
    h.runtime.run.assert_not_awaited()
    h.kakao.send.assert_awaited_once()


async def test_deadline_elapsed_while_claiming_send_does_not_send(harness):
    h = harness
    h.event.update(delivery_status="ready", response_payload=render_callback("준비된 답변"))

    async def delayed_claim(event_id):
        h.clock.now = h.event["callback_deadline_at"] + timedelta(seconds=1)
        h.event["delivery_status"] = "sending"
        return h.event

    h.store.begin_send.side_effect = delayed_claim
    await worker.process_event.func("event")
    h.kakao.send.assert_not_awaited()
    assert h.event["delivery_status"] in {"unknown", "rejected", "expired"}
