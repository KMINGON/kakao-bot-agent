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
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI


class BudgetExceeded(RuntimeError):
    pass


def estimated_model_cost(model: str, input_tokens: int, cached_input_tokens: int,
                         output_tokens: int, service_tier: str | None = None) -> float | None:
    # Standard short-context USD per 1M tokens; runtime input limit is below 272K.
    rates = {"gpt-5.4-mini": (.75, .075, 4.5),
             "gpt-5.4-mini-2026-03-17": (.75, .075, 4.5),
             "gpt-6.1-sol": (2, .10, 10)}.get(model)
    if rates is None:
        return None
    multiplier = {None: 1, "default": 1}.get(service_tier)
    if model == "gpt-6.1-sol":
        multiplier = {None: 1, "default": 1, "fast": 2, "priority": 2, "ultrafast": 6}.get(service_tier)
    if multiplier is None:
        return None
    return round((max(0, input_tokens - cached_input_tokens) * rates[0]
                  + cached_input_tokens * rates[1] + output_tokens * rates[2]) * multiplier / 1_000_000, 8)


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
    model_name: str = "gpt-5.4-mini-2026-03-17"
    service_tier: str | None = None

    def remaining(self) -> float:
        left = self.deadline_monotonic - time.monotonic()
        if left <= 0:
            raise TimeoutError("Agent deadline expired")
        return left

    def model_start(self, request: dict, *, reasoning_tokens: dict[str, int] | None = None) -> None:
        self.remaining()
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("Model call limit exceeded")
        # Responses ciphertext is opaque transport state, not tokenized model text.
        # Charge its recorded reasoning tokens (or a full output allowance), preserving the wire payload.
        estimated_request, opaque_tokens = request, 0
        if isinstance(request.get("input"), list):
            inputs = []
            for item in request["input"]:
                if isinstance(item, dict) and item.get("type") == "reasoning" and item.get("encrypted_content"):
                    inputs.append({key: value for key, value in item.items() if key != "encrypted_content"})
                    opaque_tokens += (reasoning_tokens or {}).get(item.get("id"), 2048)
                else:
                    inputs.append(item)
            estimated_request = {**request, "input": inputs}
        estimated = len(_ENCODER.encode(json.dumps(estimated_request, ensure_ascii=False, default=str))) + opaque_tokens + 512
        if estimated > self.max_input_tokens:
            raise BudgetExceeded(f"Final request exceeds input budget ({estimated})")
        self.model_calls += 1
        self.trace.append({"kind": "model_start", "call": self.model_calls,
                           "estimated_input_tokens": estimated})

    def usage(self) -> dict:
        # Image analysis retains its own mini model and is already in token totals.
        media = [item for item in self.trace if item.get("kind") == "media_usage"
                 and item.get("included_in_primary_token_totals")]
        media_input = sum(item["usage"].get("input_tokens", 0) for item in media)
        media_output = sum(item["usage"].get("output_tokens", 0) for item in media)
        media_cached = sum((item["usage"].get("input_tokens_details") or {}).get("cached_tokens", 0)
                           for item in media)
        calls = [item for item in self.trace if item.get("kind") == "model_usage"]
        cost = estimated_model_cost(self.model_name,
            self.input_tokens - media_input - sum(item["input_tokens"] for item in calls),
            self.cached_input_tokens - media_cached - sum(item["cached_input_tokens"] for item in calls),
            self.output_tokens - media_output - sum(item["output_tokens"] for item in calls), self.service_tier)
        # With complete recorded calls, an unknown configured tier does not erase known actual billing.
        if calls and self.input_tokens == media_input + sum(item["input_tokens"] for item in calls) \
                and self.output_tokens == media_output + sum(item["output_tokens"] for item in calls):
            cost = 0.0
        for item in [*media, *calls]:
            extra = item.get("estimated_cost_usd")
            cost = None if cost is None or extra is None else round(cost + extra, 8)
        tiers = {item["service_tier"] for item in calls} or {self.service_tier or "default"}
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "estimated_cost_usd": cost,
                "pricing_model": self.model_name,
                "pricing_service_tier": next(iter(tiers)) if len(tiers) == 1 else "mixed"}


_ENCODER = tiktoken.get_encoding("o200k_base")
active_budget: ContextVar[RunBudget | None] = ContextVar("agent_budget", default=None)


def count_context_tokens(messages, *, tools=None) -> int:
    """Count summary context without tokenizing opaque Responses transport state."""
    visible, opaque_tokens = [], 0
    for message in messages:
        if isinstance(message.content, list):
            content = []
            usage = getattr(message, "usage_metadata", None) or {}
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "reasoning" and block.get("encrypted_content"):
                    content.append({key: value for key, value in block.items() if key != "encrypted_content"})
                    opaque_tokens += usage.get("output_token_details", {}).get(
                        "reasoning", usage.get("output_tokens", 2048))
                else:
                    content.append(block)
            message = message.model_copy(update={"content": content})
        visible.append(message)
    return count_tokens_approximately(visible, tools=tools, use_usage_metadata_scaling=True) + opaque_tokens


class BudgetedChatOpenAI(ChatOpenAI):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = active_budget.get()
        if budget is None:
            raise RuntimeError("Provider invocation requires an active run budget")
        reasoning_tokens = {}
        for message in messages:
            usage = getattr(message, "usage_metadata", None) or {}
            for block in message.content if isinstance(message.content, list) else []:
                if isinstance(block, dict) and block.get("type") == "reasoning" and block.get("encrypted_content"):
                    reasoning_tokens[block.get("id")] = usage.get("output_token_details", {}).get(
                        "reasoning", usage.get("output_tokens", self.max_tokens or 2048))
        budget.model_start(self._get_request_payload(messages, stop=stop, **kwargs), reasoning_tokens=reasoning_tokens)
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
            metadata = generation.message.response_metadata or {}
            tier = metadata.get("service_tier") or budget.service_tier
            model = metadata.get("model_name") or metadata.get("model") or budget.model_name
            incoming, outgoing = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            cached = usage.get("input_token_details", {}).get("cache_read", 0)
            budget.trace.append({"kind": "model_usage", "model": model,
                "service_tier": tier or "default", "input_tokens": incoming,
                "output_tokens": outgoing, "cached_input_tokens": cached,
                "estimated_cost_usd": estimated_model_cost(model, incoming, cached, outgoing, tier)})
        budget.trace.append({"kind": "model_end", "elapsed_ms": round((time.monotonic() - started) * 1000)})
        return result

    def _generate(self, *args, **kwargs):
        raise RuntimeError("Only budgeted asynchronous model calls are supported")


class ToolPolicyMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        # The graph input schema excludes structured_response, so clear it here.
        return {"structured_response": None}

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
            return ToolMessage(content=json.dumps({"status": "unknown", "error": type(error).__name__,
                               "message": "도구 실행 결과를 확인하지 못했습니다. 접수·변경 요청은 이미 처리됐을 수 있으므로 성공·실패를 단정하거나 본문을 바꿔 재시도하지 마세요."}, ensure_ascii=False),
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
