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
    claims: list[Claim]
    evidence_ids: list[str]
    missing_fields: list[str] = Field(description="고객에게 지금 확인할 필수 정보. 목록에 있으면 answer에서도 실제 질문한다.")
    status: Literal["answered", "needs_input", "uncertain"]
    answer: str = Field(description="위에서 확인한 사실·정책 판단·고객 선택지·필수 질문·실제 접수 결과를 빠뜨리지 않는 간결한 한국어 답변, 430~450자 이내. 출처 목록은 서버가 붙이므로 중복하지 않는다. 출처 포함 최종 답변은 500자 이내")


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str
    openai_api_key: str = field(repr=False)
    model_name: str = "gpt-5.4-mini-2026-03-17"
    service_tier: str | None = None
    packs_root: Path = Path("domain_packs")
    max_model_calls: int = 8
    max_tool_calls: int = 12
    max_input_tokens: int = 24000
    max_output_tokens: int = 2048
    run_timeout_seconds: float = 40
    tool_timeout_seconds: float = 15
    model_timeout_seconds: float = 40
    summary_trigger_tokens: int = 18000
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
