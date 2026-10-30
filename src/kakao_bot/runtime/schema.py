from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="확인된 업무 사실. 근거가 없으면 주장하지 않는다.")
    evidence_ids: list[str] = Field(description="실제로 조회한 출처 ID")


class AnswerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "needs_input", "uncertain"]
    answer: str = Field(description="사용자에게 보낼 간결한 한국어 답변, 850자 이내")
    claims: list[Claim]
    evidence_ids: list[str]
    missing_fields: list[str]


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str
    openai_api_key: str = field(repr=False)
    model_name: str = "gpt-5.4-mini-2026-03-17"
    packs_root: Path = Path("domain_packs")
    max_model_calls: int = 8
    max_tool_calls: int = 12
    max_input_tokens: int = 24000
    max_output_tokens: int = 2048
    run_timeout_seconds: float = 180
    tool_timeout_seconds: float = 15
    model_timeout_seconds: float = 60
    summary_trigger_tokens: int = 12000
    summary_keep_messages: int = 8


@dataclass(frozen=True)
class RunContext:
    thread_id: str
    event_id: str
    pack_digest: str
    deadline: datetime


@dataclass
class RuntimeResult:
    text: str
    usage: dict
    trace: list[dict]
    answer: dict = field(default_factory=dict)
    recovered: bool = False
