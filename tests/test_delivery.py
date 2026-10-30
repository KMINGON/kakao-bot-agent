"""Transport correctness without spending model tokens or contacting Kakao."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import procrastinate
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kakao_bot.app import create_app
from kakao_bot.contracts import normalize_message, render_callback
from kakao_bot.kakao import DeliveryResult, KakaoClient, classify_response
from kakao_bot.settings import Settings
from kakao_bot.storage import DuplicateConflict, Storage, migrate


def wire(*, message_id: int = 558185822630215770, content: str = "의자 재고를 알려줘", token: str | None = "synthetic-never-send", **changes):
    data = {"id": message_id, "content": content, "timestamp": datetime.now(UTC).isoformat(),
            "isChannelChatroom": True, "botGroupKey": "test-room", "botUserKey": "test-actor", "callbackToken": token}
    data.update(changes)
    return {"op": 0, "t": "MESSAGE_CREATE", "d": data}


def settings(**changes):
    return Settings(database_url="postgresql://unused", webhook_secret="w" * 32, ops_token="o" * 32, **changes)


def test_long_ids_and_token_boundary():
    event = normalize_message(wire())
    assert event.envelope["message_id"] == "558185822630215770"
    assert "synthetic-never-send" not in repr(event)
    assert "callbackToken" not in event.envelope
    assert str(558185822630215770) in event.upstream_key


@pytest.mark.parametrize("field,value", [("id", True), ("id", 1.5), ("id", "123"), ("isChannelChatroom", "true"), ("timestamp", "2026-10-31T12:00:00"), ("botUserKey", "")])
def test_strict_wire_fields(field, value):
    with pytest.raises((ValidationError, ValueError)):
        normalize_message(wire(**{field: value}))


def test_deadline_uses_earlier_timestamp():
    now = datetime.now(UTC)
    event = normalize_message(wire(timestamp=(now - timedelta(seconds=10)).isoformat()), now=now)
    assert event.deadline == now + timedelta(seconds=285)
    with pytest.raises(ValueError):
        normalize_message(wire(timestamp=(now + timedelta(seconds=60)).isoformat()), now=now)


def test_no_token_and_channel_filter():
    assert normalize_message(wire(token=None)) is None
    assert normalize_message(wire(isChannelChatroom=False)) is None
    assert normalize_message({"op": 0, "t": "ENTRANCE", "d": {}}) is None


def test_users_in_same_room_have_separate_private_context():
    a = normalize_message(wire(botUserKey="actor-a"))
    b = normalize_message(wire(botUserKey="actor-b"))
    assert a.conversation_key != b.conversation_key


def test_callback_is_one_short_text_bubble():
    result = render_callback("가" * 1500)
    outputs = result["skillResponse"]["template"]["outputs"]
    assert len(outputs) == 1
    assert len(outputs[0]["simpleText"]["text"]) <= 1000
    with pytest.raises(ValueError):
        render_callback(" \n")


@pytest.mark.parametrize("code,body,expected", [
    (200, {"totalCount": 1, "sendCount": 1, "errorCount": 0}, "sent"),
    (200, {"totalCount": 1, "sendCount": 0, "errorCount": 1}, "rejected"),
    (200, {}, "unknown"),
    (200, {"totalCount": True, "sendCount": True, "errorCount": 0}, "unknown"),
    (200, {"totalCount": 1, "sendCount": 0, "errorCount": "1"}, "unknown"),
    (400, {"code": -881}, "unknown"),
    (401, {"code": -401}, "rejected"),
    (500, {}, "unknown"),
])
def test_callback_counts_not_http_alone(code, body, expected):
    assert classify_response(httpx.Response(code, json=body)).status == expected


async def test_callback_timeout_does_not_retry():
    count = 0

    async def failed(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("simulated", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
        result = await KakaoClient("test", client=client).send("fake", render_callback("hello"))
    assert result.status == "unknown"
    assert count == 1


class FakeStorage:
    def __init__(self):
        self.events = []

    async def admit(self, event, snapshot, task):
        self.events.append(event)
        return {"event_id": "fake", "duplicate": False}

    async def ready(self):
        return True

    async def status(self):
        return {"db": "ok"}


def test_webhook_auth_validation_ack_and_no_sync_model():
    store = FakeStorage()
    app = create_app(settings(), storage=store, task=object(), snapshot={"mode": "echo"})
    headers = {"X-Bot-Secret": "w" * 32}
    with TestClient(app) as client:
        assert client.post("/webhooks/kakao", json=wire()).status_code == 401
        assert client.post("/webhooks/kakao", headers=headers, content="{").status_code == 400
        assert client.post("/webhooks/kakao", headers=headers, content="x" * 65537).status_code == 413
        response = client.post("/webhooks/kakao", headers=headers, json=wire())
        assert response.status_code == 204 and response.content == b""
        assert len(store.events) == 1
        assert client.post("/webhooks/kakao", headers=headers, json=wire(token=None)).status_code == 204
        assert len(store.events) == 1
        assert client.get("/ops/status").status_code == 401
        assert client.get("/ops/status", headers={"Authorization": "Bearer " + "o" * 32}).status_code == 200
        assert client.get("/docs").status_code == 404


def test_enqueue_database_error_returns_503_without_success_ack():
    from procrastinate.exceptions import ConnectorException

    class Unavailable(FakeStorage):
        async def admit(self, *args):
            raise ConnectorException()

    app = create_app(settings(), storage=Unavailable(), task=object(), snapshot={})
    with TestClient(app) as client:
        response = client.post("/webhooks/kakao", headers={"X-Bot-Secret": "w" * 32}, json=wire())
        assert response.status_code == 503


@pytest_asyncio.fixture
async def pg():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for disposable PostgreSQL integration")
    scope = "td-" + uuid4().hex[:20]
    cfg = settings(channel_scope=scope, mode="echo")
    cfg = replace(cfg, database_url=url)
    await migrate(cfg)
    store = Storage(cfg)
    await store.open()
    queue = procrastinate.App(connector=procrastinate.PsycopgConnector(conninfo=url, min_size=1, max_size=3))

    @queue.task(name=f"test.{scope}", queue=scope)
    async def task(event_id: str):
        return None

    async with queue.open_async():
        yield cfg, store, queue, task
        async with store.pool.connection() as conn:
            await conn.execute("DELETE FROM bot_event WHERE channel_scope=%s", (scope,))
            await conn.execute("DELETE FROM bot_conversation WHERE channel_scope=%s", (scope,))
            await conn.execute("DELETE FROM procrastinate_jobs WHERE queue_name=%s", (scope,))
    await store.close()


def snapshot(cfg):
    return {"pack_id": "test", "version": "1", "digest": "test-pack", "dataset_version": "1",
            "runtime_revision": cfg.runtime_revision, "model_name": cfg.model_name, "mode": "echo"}


@pytest.mark.integration
async def test_pg_atomic_enqueue_rollback_and_duplicate(pg):
    cfg, store, queue, task = pg
    event = normalize_message(wire())

    class BrokenTask:
        def configure(self, **kwargs):
            configured = task.configure(**kwargs)

            async def fail(**args):
                await configured.defer_async(**args)
                raise RuntimeError("after enqueue before commit")

            return SimpleNamespace(defer_async=fail)

    with pytest.raises(RuntimeError):
        await store.admit(event, snapshot(cfg), BrokenTask())
    async with store.pool.connection() as conn:
        assert (await (await conn.execute("SELECT count(*) AS n FROM bot_event WHERE channel_scope=%s", (cfg.channel_scope,))).fetchone())["n"] == 0
        assert (await (await conn.execute("SELECT count(*) AS n FROM procrastinate_jobs WHERE queue_name=%s", (cfg.channel_scope,))).fetchone())["n"] == 0
    results = await asyncio.gather(*(store.admit(event, snapshot(cfg), task) for _ in range(8)))
    assert len({r["event_id"] for r in results}) == 1
    assert sum(not r["duplicate"] for r in results) == 1
    conflicting = replace(event, input_hash="different")
    with pytest.raises(DuplicateConflict):
        await store.admit(conflicting, snapshot(cfg), task)
    row = await store.get_event(results[0]["event_id"])
    assert row["seq"] == 1 and row["job_id"] is not None


@pytest.mark.integration
async def test_pg_order_pack_switch_and_actor_isolation(pg):
    cfg, store, queue, task = pg
    events = [normalize_message(wire(message_id=1000+i)) for i in range(5)]
    receipts = await asyncio.gather(*(store.admit(e, snapshot(cfg), task) for e in events))
    rows = [await store.get_event(r["event_id"]) for r in receipts]
    ordered = sorted(rows, key=lambda row: row["seq"])
    assert [r["seq"] for r in ordered] == list(range(1, 6))
    assert [r["job_id"] for r in ordered] == sorted(r["job_id"] for r in ordered)
    first = await store.start_run(str(ordered[0]["id"]))
    with pytest.raises(RuntimeError, match="earlier_event_unfinished"):
        await store.start_run(str(ordered[1]["id"]))
    await store.save_response(str(first["id"]), render_callback("one"))
    second = await store.start_run(str(ordered[1]["id"]))
    assert first["graph_thread_id"] == second["graph_thread_id"]
    other = await store.admit(normalize_message(wire(message_id=9999, botUserKey="other")), snapshot(cfg), task)
    other_row = await store.start_run(other["event_id"])
    assert other_row["graph_thread_id"] != first["graph_thread_id"]


@pytest.mark.integration
async def test_pg_unknown_callback_never_retried(pg, monkeypatch):
    from kakao_bot import worker

    cfg, store, queue, task = pg
    calls = []

    async def send(token, payload):
        calls.append(payload)
        return DeliveryResult("unknown", "simulated_timeout", {})

    monkeypatch.setattr(worker, "settings", cfg)
    monkeypatch.setattr(worker, "_storage", store)
    monkeypatch.setattr(worker, "_runtime", None)
    monkeypatch.setattr(worker, "_kakao", SimpleNamespace(send=send))
    receipt = await store.admit(normalize_message(wire()), snapshot(cfg), task)
    await worker.process_event.func(receipt["event_id"])
    await worker.process_event.func(receipt["event_id"])
    row = await store.get_event(receipt["event_id"])
    assert row["delivery_status"] == "unknown" and row["send_attempts"] == 1
    assert row["callback_token"] is None and len(calls) == 1


@pytest.mark.integration
async def test_pg_interrupted_send_is_unknown_and_expired_never_calls(pg, monkeypatch):
    from kakao_bot import worker

    cfg, store, queue, task = pg
    receipt = await store.admit(normalize_message(wire()), snapshot(cfg), task)
    await store.start_run(receipt["event_id"])
    await store.save_response(receipt["event_id"], render_callback("prepared"))
    assert await store.begin_send(receipt["event_id"])
    await store.recover(queue, task)
    row = await store.get_event(receipt["event_id"])
    assert row["delivery_status"] == "unknown" and row["callback_token"] is None
    expired = normalize_message(wire(message_id=7777, timestamp=(datetime.now(UTC)-timedelta(minutes=6)).isoformat()))
    expired_receipt = await store.admit(expired, snapshot(cfg), task)

    async def forbidden(*args):
        raise AssertionError("expired event must not invoke Kakao")

    monkeypatch.setattr(worker, "settings", cfg)
    monkeypatch.setattr(worker, "_storage", store)
    monkeypatch.setattr(worker, "_kakao", SimpleNamespace(send=forbidden))
    await worker.process_event.func(expired_receipt["event_id"])
    assert (await store.get_event(expired_receipt["event_id"]))["delivery_status"] == "expired"


@pytest.mark.integration
async def test_real_queue_executes_in_conversation_order(pg, monkeypatch):
    from kakao_bot import worker

    cfg, store, queue, task = pg
    delivered = []

    async def send(token, payload):
        await asyncio.sleep(0.01)
        delivered.append(payload["skillResponse"]["template"]["outputs"][0]["simpleText"]["text"])
        return DeliveryResult("sent", "ok", {"totalCount": 1, "sendCount": 1, "errorCount": 0})

    monkeypatch.setattr(worker, "settings", cfg)
    monkeypatch.setattr(worker, "_storage", store)
    monkeypatch.setattr(worker, "_runtime", None)
    monkeypatch.setattr(worker, "_kakao", SimpleNamespace(send=send))
    monkeypatch.setattr(task, "func", worker.process_event.func)
    ids = []
    for i in range(4):
        receipt = await store.admit(normalize_message(wire(message_id=8000+i, content=f"message-{i}")), snapshot(cfg), task)
        ids.append(receipt["event_id"])
    running = asyncio.create_task(queue.run_worker_async(queues=[cfg.channel_scope], concurrency=2, fetch_job_polling_interval=0.05, install_signal_handlers=False))
    try:
        async with asyncio.timeout(5):
            while True:
                rows = [await store.get_event(event_id) for event_id in ids]
                if all(row["delivery_status"] == "sent" for row in rows):
                    break
                await asyncio.sleep(0.02)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert delivered == [f"연결 확인: message-{i}" for i in range(4)]
    for event_id in ids:
        row = await store.get_event(event_id)
        assert row["delivery_status"] == "sent" and row["send_attempts"] == 1


@pytest.mark.integration
async def test_real_queue_recovers_failed_job_same_event_and_thread(pg, monkeypatch):
    from kakao_bot import worker

    cfg, store, queue, task = pg
    receipt = await store.admit(normalize_message(wire()), snapshot(cfg), task)

    async def interrupted(event_id):
        await store.start_run(event_id)
        raise RuntimeError("simulated worker interruption after claiming event")

    monkeypatch.setattr(task, "func", interrupted)
    await queue.run_worker_async(queues=[cfg.channel_scope], wait=False, install_signal_handlers=False)
    before = await store.get_event(receipt["event_id"])
    assert before["run_status"] == "running"
    recovered = await store.recover(queue, task)
    assert recovered["recovered_jobs"] == 1
    calls = []

    async def send(token, payload):
        calls.append(payload)
        return DeliveryResult("sent", "ok", {"totalCount": 1, "sendCount": 1, "errorCount": 0})

    monkeypatch.setattr(worker, "settings", cfg)
    monkeypatch.setattr(worker, "_storage", store)
    monkeypatch.setattr(worker, "_runtime", None)
    monkeypatch.setattr(worker, "_kakao", SimpleNamespace(send=send))
    monkeypatch.setattr(task, "func", worker.process_event.func)
    await queue.run_worker_async(queues=[cfg.channel_scope], wait=False, install_signal_handlers=False)
    after = await store.get_event(receipt["event_id"])
    assert after["job_id"] == before["job_id"]
    assert after["graph_thread_id"] == before["graph_thread_id"]
    assert after["run_attempts"] == 2
    assert after["delivery_status"] == "sent" and len(calls) == 1


@pytest.mark.integration
async def test_recovery_preserves_other_channel_scope(pg):
    cfg, store, queue, task = pg
    other_cfg = replace(cfg, channel_scope=cfg.channel_scope + "-other")
    other = Storage(other_cfg)
    await other.open()
    try:
        receipt = await other.admit(normalize_message(wire()), snapshot(other_cfg), task)
        await other.start_run(receipt["event_id"])
        await other.save_response(receipt["event_id"], render_callback("other bot sending"))
        row = await other.begin_send(receipt["event_id"])
        async with other.pool.connection() as conn:
            await conn.execute("UPDATE procrastinate_jobs SET status='doing' WHERE id=%s", (row["job_id"],))
        assert await store.recover(queue, task) == {"interrupted_sends": 0, "recovered_jobs": 0}
        untouched = await other.get_event(receipt["event_id"])
        assert untouched["delivery_status"] == "sending"
        assert untouched["callback_token"] is not None
        async with other.pool.connection() as conn:
            job = await (await conn.execute("SELECT status FROM procrastinate_jobs WHERE id=%s", (row["job_id"],))).fetchone()
            assert job["status"] == "doing"
    finally:
        async with other.pool.connection() as conn:
            await conn.execute("DELETE FROM bot_event WHERE channel_scope=%s", (other_cfg.channel_scope,))
            await conn.execute("DELETE FROM bot_conversation WHERE channel_scope=%s", (other_cfg.channel_scope,))
        await other.close()
