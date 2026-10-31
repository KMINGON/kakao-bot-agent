"""Kakao wire validation and deterministic transport identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator

MAX_RESPONSE_CHARS = 500
RESPONSE_DEADLINE_SECONDS = 55


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: StrictInt = Field(ge=0, le=9223372036854775807)
    content: StrictStr = Field(min_length=1, max_length=16000)
    timestamp: StrictStr
    isChannelChatroom: StrictBool
    botGroupKey: StrictStr = Field(min_length=1, max_length=512)
    botUserKey: StrictStr = Field(min_length=1, max_length=512)
    callbackToken: StrictStr | None = Field(default=None, max_length=8192)

    @field_validator("timestamp")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp must contain a timezone")
        return value


class Dispatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    op: StrictInt
    t: StrictStr
    d: dict[str, Any]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class IncomingEvent:
    upstream_key: str
    conversation_key: str
    input_hash: str
    envelope: dict[str, Any]
    callback_token: str = field(repr=False)
    received_at: datetime
    source_timestamp: datetime
    deadline: datetime


def normalize_message(payload: Any, *, channel_only: bool = True, now: datetime | None = None) -> IncomingEvent | None:
    dispatch = Dispatch.model_validate(payload)
    if dispatch.op != 0 or dispatch.t != "MESSAGE_CREATE":
        return None
    message = Message.model_validate(dispatch.d)
    if not message.callbackToken or (channel_only and not message.isChannelChatroom):
        return None
    received_at = now or datetime.now(UTC)
    source = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00")).astimezone(UTC)
    if source > received_at + timedelta(seconds=30):
        raise ValueError("timestamp is in the future")
    envelope = {
        "message_id": str(message.id),
        "content": message.content,
        "actor_id": message.botUserKey,
        "group_key": message.botGroupKey,
        "is_channel": message.isChannelChatroom,
        "timestamp": source.isoformat(),
    }
    # Actor is part of the private conversation boundary, even in a shared room.
    conversation_key = stable_json([message.isChannelChatroom, message.botGroupKey, message.botUserKey])
    return IncomingEvent(
        upstream_key=stable_json(["MESSAGE_CREATE", message.botGroupKey, str(message.id)]),
        conversation_key=conversation_key,
        input_hash=hashlib.sha256(stable_json(envelope).encode()).hexdigest(),
        envelope=envelope,
        callback_token=message.callbackToken,
        received_at=received_at,
        source_timestamp=source,
        deadline=min(received_at + timedelta(seconds=RESPONSE_DEADLINE_SECONDS),
                     source + timedelta(seconds=295)),
    )


def render_callback(text: str) -> dict[str, Any]:
    clean = text.strip()
    if not clean:
        raise ValueError("empty response")
    if len(clean) > MAX_RESPONSE_CHARS:
        clean = clean[:MAX_RESPONSE_CHARS - 1].rstrip() + "…"
    return {"skillResponse": {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": clean}}]}}}
