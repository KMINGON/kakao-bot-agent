"""Environment configuration. API processes do not require worker-only secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


def secret(name: str, *, default: str = "") -> str:
    path = os.getenv(f"{name}_FILE")
    value = Path(path).read_text(encoding="utf-8").strip() if path else os.getenv(name, default)
    return value


def integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    webhook_secret: str = field(default="", repr=False)
    ops_token: str = field(default="", repr=False)
    kakao_bot_token: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    webhook_header_name: str = "X-Bot-Secret"
    channel_scope: str = "1630499"
    channel_only: bool = True
    active_pack: str = "equipment_demo"
    packs_root: Path = Path("domain_packs")
    mode: str = "agent"
    model_name: str = "gpt-5.4-mini-2026-03-17"
    service_tier: str | None = None
    runtime_revision: str = "v1"
    worker_concurrency: int = 1
    run_timeout_seconds: int = 40
    max_model_calls: int = 8
    max_tool_calls: int = 12
    max_input_tokens: int = 24000
    max_output_tokens: int = 2048
    summary_trigger_tokens: int = 18000
    max_body_bytes: int = 65536
    admission_timeout_seconds: float = 0.75

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = secret("DATABASE_URL")
        if not database_url:
            password = secret("POSTGRES_PASSWORD")
            user = quote(os.getenv("DATABASE_USER", "kakao"), safe="")
            host = os.getenv("DATABASE_HOST", "postgres")
            port = integer("DATABASE_PORT", 5432, 1, 65535)
            name = quote(os.getenv("DATABASE_NAME", "kakao"), safe="")
            database_url = f"postgresql://{user}:{quote(password, safe='')}@{host}:{port}/{name}"
        mode = os.getenv("BOT_MODE", "agent")
        if mode not in {"agent", "echo"}:
            raise ValueError("BOT_MODE must be agent or echo")
        service_tier = os.getenv("MODEL_SERVICE_TIER", "").strip() or None
        if service_tier not in {None, "default", "fast", "priority", "ultrafast"}:
            raise ValueError("MODEL_SERVICE_TIER must be default, fast, priority, ultrafast, or empty")
        return cls(
            database_url=database_url,
            webhook_secret=secret("KAKAO_WEBHOOK_SECRET"),
            ops_token=secret("OPS_TOKEN"),
            kakao_bot_token=secret("KAKAO_BOT_TOKEN"),
            openai_api_key=secret("OPENAI_API_KEY"),
            webhook_header_name=os.getenv("KAKAO_WEBHOOK_HEADER_NAME", "X-Bot-Secret"),
            channel_scope=os.getenv("CHANNEL_SCOPE", "1630499"),
            channel_only=os.getenv("CHANNEL_ONLY", "true").lower() == "true",
            active_pack=os.getenv("ACTIVE_PACK", "equipment_demo"),
            packs_root=Path(os.getenv("PACKS_ROOT", "domain_packs")),
            mode=mode,
            model_name=os.getenv("MODEL_NAME", "gpt-5.4-mini-2026-03-17"),
            service_tier=service_tier,
            runtime_revision=os.getenv("RUNTIME_REVISION", "v1"),
            worker_concurrency=integer("WORKER_CONCURRENCY", 1, 1, 8),
            run_timeout_seconds=integer("AGENT_RUN_CAP_SECONDS", 40, 5, 40),
            max_model_calls=integer("MAX_MODEL_CALLS", 8, 1, 32),
            max_tool_calls=integer("MAX_TOOL_CALLS", 12, 1, 64),
            max_input_tokens=integer("MAX_INPUT_TOKENS", 24000, 2000, 100000),
            max_output_tokens=integer("MAX_OUTPUT_TOKENS", 2048, 128, 8192),
            summary_trigger_tokens=integer("SUMMARY_TRIGGER_TOKENS", 18000, 1000, 100000),
        )
