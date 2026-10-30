"""Per-run budgets at the final provider boundary, including summarization."""
from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

import tiktoken
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunBudget:
    max_model_calls: int
    max_tool_calls: int
    max_input_tokens: int
    deadline_monotonic: float
    tool_timeout_seconds: float = 15
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    trace: list[dict] = field(default_factory=list)

    def remaining(self) -> float:
        left = self.deadline_monotonic - time.monotonic()
        if left <= 0:
            raise TimeoutError("Agent deadline expired")
        return left

    def model_start(self, request: dict) -> None:
        self.remaining()
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("Model call limit exceeded")
        # Count the final request, including system prompt and bound tool schemas.
        # JSON tokenization is a conservative local estimate, not provider billing.
        estimated = len(_ENCODER.encode(json.dumps(request, ensure_ascii=False, default=str))) + 512
        if estimated > self.max_input_tokens:
            raise BudgetExceeded(f"Final request exceeds input budget ({estimated})")
        self.model_calls += 1
        self.trace.append({"kind": "model_start", "call": self.model_calls,
                           "estimated_input_tokens": estimated})

    def usage(self) -> dict:
        uncached = max(0, self.input_tokens - self.cached_input_tokens)
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "estimated_cost_usd": round((uncached * .75 + self.cached_input_tokens * .075
                                              + self.output_tokens * 4.5) / 1_000_000, 8),
                "pricing_model": "gpt-5.4-mini-2026-03-17"}


_ENCODER = tiktoken.get_encoding("o200k_base")
active_budget: ContextVar[RunBudget | None] = ContextVar("agent_budget", default=None)


class BudgetedChatOpenAI(ChatOpenAI):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = active_budget.get()
        if budget is None:
            raise RuntimeError("Provider invocation requires an active run budget")
        budget.model_start(self._get_request_payload(messages, stop=stop, **kwargs))
        started = time.monotonic()
        try:
            async with asyncio.timeout(budget.remaining()):
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except BaseException:
            budget.trace.append({"kind": "model_error", "call": budget.model_calls,
                                 "elapsed_ms": round((time.monotonic() - started) * 1000)})
            raise
        for generation in result.generations:
            usage = generation.message.usage_metadata or {}
            budget.input_tokens += usage.get("input_tokens", 0)
            budget.output_tokens += usage.get("output_tokens", 0)
            budget.cached_input_tokens += usage.get("input_token_details", {}).get("cache_read", 0)
        budget.trace.append({"kind": "model_end", "elapsed_ms": round((time.monotonic() - started) * 1000)})
        return result

    def _generate(self, *args, **kwargs):
        raise RuntimeError("Only budgeted asynchronous model calls are supported")


class ToolPolicyMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        budget = active_budget.get()
        if budget is None:
            raise RuntimeError("Tool invocation requires an active run budget")
        budget.remaining()
        if budget.tool_calls >= budget.max_tool_calls:
            raise BudgetExceeded("Tool call limit exceeded")
        budget.tool_calls += 1
        name = request.tool_call["name"]
        if name in {"execute", "task", "delete", "write_todos"}:
            raise ValueError("Tool is disabled")
        started = time.monotonic()
        try:
            async with asyncio.timeout(min(budget.tool_timeout_seconds, budget.remaining())):
                result = await handler(request)
        except (BudgetExceeded, asyncio.CancelledError):
            raise
        except Exception as error:
            # Provider/transport exception strings can contain credentials and URLs.
            budget.trace.append({"kind": "tool_error", "tool": name, "error_type": type(error).__name__})
            return ToolMessage(content=json.dumps({"status": "failed", "error": type(error).__name__,
                               "message": "도구 조회에 실패했습니다. 결과를 추정하지 마세요."}, ensure_ascii=False),
                               tool_call_id=request.tool_call["id"], name=name, status="error")
        budget.trace.append({"kind": "tool", "tool": name,
                             "elapsed_ms": round((time.monotonic() - started) * 1000)})
        if isinstance(result, ToolMessage):
            content = result.content if isinstance(result.content, str) else json.dumps(result.content, ensure_ascii=False)
            if len(_ENCODER.encode(content)) > 2000:
                # The caller must query a smaller slice; a truncated response is never a complete source.
                result = result.model_copy(update={"content": json.dumps({"status": "uncertain",
                    "truncated": True, "message": "조회 범위를 좁혀 다시 요청하세요.",
                    "preview": _ENCODER.decode(_ENCODER.encode(content)[:1700])}, ensure_ascii=False)})
        return result
