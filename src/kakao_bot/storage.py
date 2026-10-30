"""PostgreSQL event ledger and transactional Procrastinate admission."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .contracts import IncomingEvent
from .settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_conversation (
    id uuid PRIMARY KEY,
    channel_scope text NOT NULL,
    conversation_key text NOT NULL,
    next_seq bigint NOT NULL DEFAULT 1 CHECK(next_seq > 0),
    generation bigint NOT NULL DEFAULT 0,
    active_pack_digest text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(channel_scope, conversation_key)
);
CREATE TABLE IF NOT EXISTS bot_event (
    id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES bot_conversation(id),
    channel_scope text NOT NULL,
    upstream_key text NOT NULL,
    input_hash text NOT NULL,
    seq bigint NOT NULL CHECK(seq > 0),
    received_at timestamptz NOT NULL,
    source_timestamp timestamptz NOT NULL,
    callback_deadline_at timestamptz NOT NULL,
    callback_token text,
    input_envelope jsonb NOT NULL,
    execution_spec jsonb NOT NULL CHECK(jsonb_typeof(execution_spec) = 'object'),
    graph_thread_id text,
    job_id bigint UNIQUE,
    run_status text NOT NULL DEFAULT 'queued' CHECK(run_status IN ('queued','running','succeeded','failed','expired','needs_review')),
    delivery_status text NOT NULL DEFAULT 'not_ready' CHECK(delivery_status IN ('not_ready','ready','sending','sent','rejected','unknown','expired')),
    response_payload jsonb,
    usage jsonb NOT NULL DEFAULT '{}',
    trace jsonb NOT NULL DEFAULT '[]',
    run_attempts integer NOT NULL DEFAULT 0 CHECK(run_attempts >= 0),
    send_attempts integer NOT NULL DEFAULT 0 CHECK(send_attempts BETWEEN 0 AND 1),
    send_started_at timestamptz,
    sent_at timestamptz,
    last_error_code text,
    callback_counts jsonb,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(channel_scope, upstream_key),
    UNIQUE(conversation_id, seq),
    CHECK(run_status <> 'succeeded' OR response_payload IS NOT NULL),
    CHECK(delivery_status <> 'sent' OR (sent_at IS NOT NULL AND send_started_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS bot_event_open ON bot_event(conversation_id, seq)
    WHERE run_status IN ('queued','running','needs_review');
CREATE INDEX IF NOT EXISTS bot_event_deadline ON bot_event(callback_deadline_at)
    WHERE delivery_status IN ('not_ready','ready','sending');
"""


class DuplicateConflict(ValueError):
    pass


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = AsyncConnectionPool(
            settings.database_url, min_size=1, max_size=5, open=False, timeout=0.5,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        )

    async def open(self) -> None:
        await self.pool.open(wait=True, timeout=20)

    async def close(self) -> None:
        await self.pool.close()

    async def get_event(self, event_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT * FROM bot_event WHERE id=%s", (event_id,))
            return await cursor.fetchone()

    async def admit(self, event: IncomingEvent, snapshot: dict[str, Any], task: Any) -> dict[str, Any]:
        event_id = str(uuid4())
        try:
            async with self.pool.connection() as conn, conn.transaction():
                await conn.execute("SET LOCAL lock_timeout = '250ms'")
                await conn.execute("SET LOCAL statement_timeout = '500ms'")
                cursor = await conn.execute(
                    """INSERT INTO bot_conversation(id,channel_scope,conversation_key)
                       VALUES (%s,%s,%s) ON CONFLICT(channel_scope,conversation_key)
                       DO UPDATE SET updated_at=now() RETURNING *""",
                    (str(uuid4()), self.settings.channel_scope, event.conversation_key),
                )
                conversation = await cursor.fetchone()
                cursor = await conn.execute(
                    "SELECT id,input_hash FROM bot_event WHERE channel_scope=%s AND upstream_key=%s",
                    (self.settings.channel_scope, event.upstream_key),
                )
                existing = await cursor.fetchone()
                if existing:
                    return self._duplicate(existing, event)
                await conn.execute(
                    "UPDATE bot_conversation SET next_seq=next_seq+1 WHERE id=%s", (conversation["id"],),
                )
                await conn.execute(
                    """INSERT INTO bot_event(id,conversation_id,channel_scope,upstream_key,input_hash,seq,
                       received_at,source_timestamp,callback_deadline_at,callback_token,input_envelope,execution_spec)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (event_id, conversation["id"], self.settings.channel_scope, event.upstream_key,
                     event.input_hash, conversation["next_seq"], event.received_at, event.source_timestamp,
                     event.deadline, event.callback_token, Jsonb(event.envelope), Jsonb(snapshot)),
                )
                job_id = await task.configure(connection=conn, lock=f"conversation:{conversation['id']}").defer_async(event_id=event_id)
                await conn.execute("UPDATE bot_event SET job_id=%s WHERE id=%s", (job_id, event_id))
            return {"event_id": event_id, "duplicate": False}
        except UniqueViolation:
            async with self.pool.connection() as conn:
                cursor = await conn.execute(
                    "SELECT id,input_hash FROM bot_event WHERE channel_scope=%s AND upstream_key=%s",
                    (self.settings.channel_scope, event.upstream_key),
                )
                existing = await cursor.fetchone()
                if existing:
                    return self._duplicate(existing, event)
            raise

    @staticmethod
    def _duplicate(existing: dict[str, Any], event: IncomingEvent) -> dict[str, Any]:
        if existing["input_hash"] != event.input_hash:
            raise DuplicateConflict("same message ID with different content")
        return {"event_id": str(existing["id"]), "duplicate": True}

    async def start_run(self, event_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn, conn.transaction():
            cursor = await conn.execute("SELECT * FROM bot_event WHERE id=%s FOR UPDATE", (event_id,))
            event = await cursor.fetchone()
            if not event or event["delivery_status"] in {"sent", "unknown", "rejected", "expired"}:
                return None
            if event["response_payload"]:
                return event
            cursor = await conn.execute("SELECT * FROM bot_conversation WHERE id=%s FOR UPDATE", (event["conversation_id"],))
            conversation = await cursor.fetchone()
            cursor = await conn.execute(
                """SELECT id FROM bot_event WHERE conversation_id=%s AND seq<%s
                   AND run_status IN ('queued','running','needs_review') LIMIT 1""",
                (event["conversation_id"], event["seq"]),
            )
            if await cursor.fetchone():
                raise RuntimeError("earlier_event_unfinished")
            digest = event["execution_spec"].get("digest", "echo")
            generation = conversation["generation"]
            if not event["graph_thread_id"]:
                if conversation["active_pack_digest"] != digest:
                    generation += 1
                    await conn.execute(
                        "UPDATE bot_conversation SET generation=%s,active_pack_digest=%s WHERE id=%s",
                        (generation, digest, conversation["id"]),
                    )
                thread = hashlib.sha256(f"{conversation['id']}:{generation}:{digest}".encode()).hexdigest()
            else:
                thread = event["graph_thread_id"]
            cursor = await conn.execute(
                """UPDATE bot_event SET run_status='running',run_attempts=run_attempts+1,
                   graph_thread_id=%s,updated_at=now() WHERE id=%s RETURNING *""", (thread, event_id),
            )
            return await cursor.fetchone()

    async def save_response(self, event_id: str, payload: dict[str, Any], *, usage: dict | None = None,
                            trace: list | None = None, failed: bool = False, code: str | None = None) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE bot_event SET response_payload=%s,usage=%s,trace=%s,run_status=%s,
                   delivery_status='ready',last_error_code=%s,updated_at=now()
                   WHERE id=%s AND delivery_status='not_ready'""",
                (Jsonb(payload), Jsonb(usage or {}), Jsonb(trace or []), "failed" if failed else "succeeded", code, event_id),
            )

    async def expire(self, event_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE bot_event SET run_status=CASE WHEN response_payload IS NULL THEN 'expired' ELSE run_status END,
                   delivery_status='expired',callback_token=NULL,last_error_code='deadline',updated_at=now()
                   WHERE id=%s AND delivery_status IN ('not_ready','ready')""", (event_id,),
            )

    async def begin_send(self, event_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """UPDATE bot_event SET delivery_status='sending',send_attempts=1,send_started_at=now(),updated_at=now()
                   WHERE id=%s AND delivery_status='ready' AND send_attempts=0
                   AND callback_deadline_at>now()+interval '10 seconds' RETURNING *""", (event_id,),
            )
            return await cursor.fetchone()

    async def finish_send(self, event_id: str, status: str, code: str, counts: dict | None = None) -> None:
        if status not in {"sent", "unknown", "rejected"}:
            raise ValueError("invalid delivery result")
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE bot_event SET delivery_status=%s,last_error_code=%s,callback_counts=%s,
                   sent_at=CASE WHEN %s='sent' THEN now() ELSE NULL END,callback_token=NULL,updated_at=now()
                   WHERE id=%s AND delivery_status='sending'""",
                (status, code, Jsonb(counts or {}), status, event_id),
            )

    async def recover(self, queue_app: Any, task: Any) -> dict[str, int]:
        """Call only while holding the exclusive worker session lock."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """UPDATE bot_event SET delivery_status='unknown',callback_token=NULL,
                   last_error_code='interrupted_send',updated_at=now()
                   WHERE delivery_status='sending' AND channel_scope=%s RETURNING id""",
                (self.settings.channel_scope,),
            )
            unknown = len(await cursor.fetchall())
            cursor = await conn.execute(
                """SELECT e.id,e.job_id,j.status AS job_status,e.conversation_id FROM bot_event e
                   JOIN procrastinate_jobs j ON j.id=e.job_id
                   WHERE e.channel_scope=%s AND j.status IN ('doing','failed','aborted') ORDER BY j.id""",
                (self.settings.channel_scope,),
            )
            candidates = await cursor.fetchall()
        retried = 0
        for event in candidates:
            if event["job_status"] in {"doing", "failed"}:
                await queue_app.job_manager.retry_job_by_id_async(event["job_id"], retry_at=datetime.now(UTC))
            else:
                async with self.pool.connection() as conn, conn.transaction():
                    job_id = await task.configure(connection=conn, lock=f"conversation:{event['conversation_id']}").defer_async(event_id=str(event["id"]))
                    await conn.execute("UPDATE bot_event SET job_id=%s WHERE id=%s", (job_id, event["id"]))
            retried += 1
        return {"interrupted_sends": unknown, "recovered_jobs": retried}

    async def ready(self) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT to_regclass('bot_event') IS NOT NULL AND to_regclass('procrastinate_jobs') IS NOT NULL AS ready")
            return bool((await cursor.fetchone())["ready"])

    async def status(self) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT delivery_status,count(*) AS count FROM bot_event GROUP BY delivery_status")
            counts = {row["delivery_status"]: row["count"] for row in await cursor.fetchall()}
            cursor = await conn.execute("SELECT EXTRACT(EPOCH FROM now()-max(last_heartbeat)) AS age FROM procrastinate_workers")
            age = (await cursor.fetchone())["age"]
            cursor = await conn.execute("SELECT count(*) AS count FROM bot_event WHERE run_status IN ('queued','running')")
            pending = (await cursor.fetchone())["count"]
            cursor = await conn.execute("SELECT COALESCE(sum((usage->>'estimated_cost_usd')::numeric),0) AS cost FROM bot_event")
            cost = (await cursor.fetchone())["cost"]
        return {"revision": self.settings.runtime_revision, "mode": self.settings.mode, "db": "ok",
                "worker_heartbeat_age_seconds": float(age) if age is not None else None,
                "pending": pending, "delivery_counts": counts, "estimated_model_cost_usd": float(cost)}


async def migrate(settings: Settings | None = None) -> None:
    import procrastinate
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    settings = settings or Settings.from_env()
    async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        cursor = await conn.execute("SELECT to_regclass('procrastinate_jobs')")
        if (await cursor.fetchone())[0] is None:
            schema = procrastinate.App(connector=procrastinate.PsycopgConnector(conninfo=settings.database_url)).schema_manager.get_schema()
            await conn.execute(schema, prepare=False)
        await conn.execute(SCHEMA, prepare=False)
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()


if __name__ == "__main__":
    asyncio.run(migrate())
    print(json.dumps({"migration": "ok"}))
