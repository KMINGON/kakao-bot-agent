import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from kakao_bot.runtime import AgentRuntime
from kakao_bot.runtime.policy import BudgetExceeded, BudgetedChatOpenAI, RunBudget, active_budget, count_context_tokens
from test_runtime import ScriptedModel, answer, config, tool_call


@pytest.mark.parametrize("model, expected", [
    ("gpt-5.4-mini-2026-03-17", .001065),
    ("gpt-6.1-sol", .00262),
    ("unknown-model", None),
])
def test_model_specific_cost_with_cached_input(model, expected):
    budget = RunBudget(1, 1, 1000, time.monotonic() + 5, model_name=model,
                       input_tokens=1000, cached_input_tokens=200, output_tokens=100)
    assert budget.usage()["estimated_cost_usd"] == expected
    assert budget.usage()["pricing_model"] == model


def test_sol_cost_keeps_mini_image_analysis_separate():
    budget = RunBudget(2, 1, 1000, time.monotonic() + 5, model_name="gpt-6.1-sol",
                       input_tokens=1100, cached_input_tokens=200, output_tokens=110)
    budget.trace.append({"kind": "media_usage", "included_in_primary_token_totals": True,
        "usage": {"input_tokens": 100, "output_tokens": 10}, "estimated_cost_usd": .00012})
    assert budget.usage()["estimated_cost_usd"] == .00274


def test_checkpoint_cost_prices_each_recorded_model_and_preserves_unknown():
    from kakao_bot.runtime.agent import event_checkpoint_usage, merge_checkpoint_usage
    def reply(model):
        return AIMessage(content="ok", response_metadata={"model_name": model},
            usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
                            "input_token_details": {"cache_read": 200}})
    messages = [HumanMessage(content="hi", id="event"), reply("gpt-5.4-mini-2026-03-17"),
                reply("gpt-6.1-sol")]
    usage = event_checkpoint_usage(messages, "event", "gpt-6.1-sol")
    assert usage["estimated_cost_usd"] == .003685
    assert usage["pricing_model"] == "mixed"
    prior = event_checkpoint_usage([HumanMessage(content="hi", id="event"), reply("unknown-model")], "event")
    merge_checkpoint_usage(usage, prior)
    assert usage["estimated_cost_usd"] is None
    assert usage["model_calls"] == 3


def test_opaque_reasoning_uses_recorded_tokens_without_mutating_payload():
    request = {"input": [{"type": "reasoning", "id": "rs_test", "encrypted_content": "opaque" * 10000},
                         {"role": "user", "content": "질문"}], "tools": [{"description": "등록된 도구"}]}
    before = json.dumps(request)
    budget = RunBudget(1, 1, 1000, time.monotonic() + 5)
    budget.model_start(request, reasoning_tokens={"rs_test": 50})
    assert 550 <= budget.trace[0]["estimated_input_tokens"] < 1000
    assert json.dumps(request) == before


def test_unknown_reasoning_keeps_a_conservative_allowance():
    budget = RunBudget(1, 1, 1000, time.monotonic() + 5)
    with pytest.raises(BudgetExceeded, match="input budget"):
        budget.model_start({"input": [{"type": "reasoning", "id": "rs_test", "encrypted_content": "opaque"}]})


@pytest.mark.parametrize("payload", [
    {"input": [{"role": "user", "content": "긴 실제 질문 " * 3000}]},
    {"tools": [{"description": "긴 실제 도구 설명 " * 3000}]},
    {"input": [{"type": "message", "encrypted_content": "ordinary field " * 3000}]},
])
def test_visible_input_and_tools_still_obey_token_budget(payload):
    with pytest.raises(BudgetExceeded, match="input budget"):
        RunBudget(1, 1, 1000, time.monotonic() + 5).model_start(payload)


async def test_provider_receives_unchanged_reasoning_and_usage_is_retained(monkeypatch):
    encrypted = "opaque" * 10000
    message = AIMessage(content=[{"type": "reasoning", "id": "rs_test", "summary": [],
                                 "encrypted_content": encrypted}],
                        usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                                        "output_token_details": {"reasoning": 15}})
    messages = [message, HumanMessage(content="이어서 확인해 주세요.")]

    async def fake_generate(self, actual_messages, **kwargs):
        assert actual_messages is messages
        assert actual_messages[0].content[0]["encrypted_content"] == encrypted
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="확인했습니다.",
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}))])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake_generate)
    model = BudgetedChatOpenAI(api_key="offline", use_responses_api=True)
    budget = RunBudget(1, 1, 1500, time.monotonic() + 5)
    token = active_budget.set(budget)
    try:
        await model._agenerate(messages)
    finally:
        active_budget.reset(token)
    assert budget.model_calls == 1 and budget.input_tokens == 100 and budget.output_tokens == 10


async def test_twelve_model_calls_have_room_for_graph_terminal_hooks():
    model = ScriptedModel(responses=[
        *[tool_call("calculate", {"expression": "2+3"}, f"call-{index}") for index in range(11)],
        AIMessage(content=json.dumps(answer("5", []))),
    ])
    async with AgentRuntime(config(max_model_calls=12, max_tool_calls=24),
                            checkpointer=InMemorySaver(), model=model) as runtime:
        result = await runtime.run(text="계산을 확인해 주세요.", thread_id="twelve", event_id="event",
            pack_ref="equipment_demo", deadline=datetime.now(UTC) + timedelta(seconds=10))
        assert result.text == "5" and len(model.requests) == 12


def test_summary_counter_ignores_ciphertext_but_counts_reasoning_and_preserves_messages():
    message = AIMessage(content=[{"type": "reasoning", "id": "rs_summary", "summary": [],
                                 "encrypted_content": "opaque" * 10000}],
        usage_metadata={"input_tokens": 10, "output_tokens": 50, "total_tokens": 60,
                        "output_token_details": {"reasoning": 40}})
    before = message.model_dump()
    assert 40 <= count_context_tokens([message]) < 100
    assert message.model_dump() == before
    assert count_context_tokens([AIMessage(content=message.content)]) >= 2048
    assert count_context_tokens([HumanMessage(content="visible " * 8000)]) > 12000
    assert count_context_tokens([message], tools=[{"description": "visible tool " * 5000}]) > 12000


def test_summary_middleware_uses_corrected_counter_without_changing_trigger():
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware

    middleware = SummarizationMiddleware(model=ScriptedModel(responses=[]), backend=StateBackend(),
        trigger=("tokens", 12000), token_counter=count_context_tokens)
    message = AIMessage(content=[{"type": "reasoning", "encrypted_content": "opaque" * 10000}],
        usage_metadata={"input_tokens": 10, "output_tokens": 50, "total_tokens": 60,
                        "output_token_details": {"reasoning": 40}})
    assert middleware._count_tokens([message], None, []) < 12000
    assert middleware._count_tokens([HumanMessage(content="visible " * 8000)], None, []) > 12000


def test_default_summary_waits_for_18k_but_provider_cap_stays_24k():
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware

    limits = config()
    assert limits.summary_trigger_tokens == 18000
    assert limits.max_input_tokens - limits.summary_trigger_tokens == 6000
    middleware = SummarizationMiddleware(model=ScriptedModel(responses=[]), backend=StateBackend(),
        trigger=("tokens", limits.summary_trigger_tokens), token_counter=count_context_tokens)
    history = [HumanMessage(content="확인"), AIMessage(content="알겠습니다.")] * 6
    assert not middleware._should_summarize(history, 16631)
    assert not middleware._should_summarize(history, 17999)
    assert middleware._should_summarize(history, 18000)
    assert middleware._should_summarize(history, 24000)
    with pytest.raises(BudgetExceeded, match="input budget"):
        RunBudget(1, 1, limits.max_input_tokens, time.monotonic() + 5).model_start(
            {"input": [{"role": "user", "content": "실제 긴 입력 " * 12000}]})
