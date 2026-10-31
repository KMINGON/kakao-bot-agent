"""Real SDK interruption recovery, without network or paid model calls."""
import asyncio
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from kakao_bot.runtime import AgentRuntime, load_pack, lock_pack
from kakao_bot.runtime.action_api import ActionAPI
from kakao_bot.runtime.policy import BudgetExceeded
from test_runtime import ScriptedModel, answer, config, tool_call


@pytest.mark.parametrize("explicit_repair", [True, False])
async def test_unfinished_tool_is_closed_and_next_turn_keeps_history(explicit_repair):
    saver = InMemorySaver()
    bad = ScriptedModel(responses=[tool_call("calculate", {"expression": "2+3"}, "unfinished")])
    deadline = datetime.now(UTC) + timedelta(seconds=20)
    async with AgentRuntime(config(max_tool_calls=0), checkpointer=saver, model=bad) as runtime:
        with pytest.raises(BudgetExceeded):
            await runtime.run(text="내 별칭은 파랑새. 2+3?", thread_id="failed-tool", event_id="first",
                              pack_ref="equipment_demo", deadline=deadline)
        if explicit_repair:
            assert await runtime.abandon("failed-tool", "first")
            assert not await runtime.abandon("failed-tool", "other-event")
    good = ScriptedModel(responses=[AIMessage(content=json.dumps(answer("파랑새님, 이어서 확인합니다.", [])))])
    async with AgentRuntime(config(), checkpointer=saver, model=good) as runtime:
        await runtime.run(text="내 별칭을 기억해?", thread_id="failed-tool", event_id="second",
                          pack_ref="equipment_demo", deadline=deadline)
        state = await runtime.inspect("failed-tool", "equipment_demo")
        assert not state.next
        assert any(m.id == "first" and "파랑새" in m.content for m in state.values["messages"])
        pending = next(m for m in state.values["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        assert pending.tool_calls[0]["args"] == {"expression": "2+3"}
        recovered = next(m for m in state.values["messages"] if isinstance(m, ToolMessage))
        assert recovered.tool_call_id == "unfinished"
        assert json.loads(recovered.content)["status"] == "unknown"
        assert "파랑새" in str(good.requests[-1])


async def test_timed_out_post_preserves_tc_customer_policy_journal_and_exact_body(tmp_path, monkeypatch):
    shutil.copytree("domain_packs/gyodong", tmp_path / "gyodong", ignore=shutil.ignore_patterns("__pycache__"))
    lock_pack(tmp_path / "gyodong")
    pack = load_pack(tmp_path, "gyodong")
    monkeypatch.setenv("ACTION_API_WRITES_ENABLED", "true")
    calls = []

    async def stalled(self, api_id, **kwargs):
        if api_id == "API-13":
            return {"api_id": api_id, "parameters": kwargs["parameters"], "status": "ok", "data": [
                {"member_id": "MB-A", "name": "고객 A", "phone": "019-1111-1234", "status": "ACTIVE"}]}
        if api_id == "API-07":
            return {"api_id": api_id, "parameters": kwargs["parameters"], "status": "ok", "data":
                {"order_no": "100000001", "member_id": "MB-A", "amount": 32000}}
        calls.append((api_id, kwargs))
        await asyncio.Event().wait()

    monkeypatch.setattr(ActionAPI, "request", stalled)
    submit = {"api_id": "API-29", "parameters": {},
              "body": {"order_no": "100000001", "amount": 32000, "reason": "원문 사유 그대로"},
              "policy_source_ids": ["D008"], "policy_reason": "확인된 하자에 대한 환불 요청",
              "customer_confirmation": "전액 환불해 주세요."}
    model = ScriptedModel(responses=[AIMessage(content=json.dumps(answer("확인했습니다.", []))),
                                     tool_call("action_submit", submit, "pending-post")])
    saver = InMemorySaver()
    old_journal = {"already-submitted": {"canonical": "preserve-exact-bytes", "result": {"status": "submitted"}}}
    async with AgentRuntime(config(packs_root=tmp_path),
                            checkpointer=saver, model=model) as runtime:
        await runtime.run(text="TC ID: L-11\ncase metadata\n- id: L-11\n- 상담 인입 일시: 2037-06-30 10:00\n- 인입창구: 채팅\n- 인입 계정: 회원 로그인 · 고객 A · 019-****-1234\n\n고객 A입니다.", thread_id="post", event_id="intro",
                          pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        graph, _ = await runtime.graph_for(pack.snapshot)
        await graph.aupdate_state({"configurable": {"thread_id": "post"}}, {
            "policy_reads": {"D008": True}, "action_requests": old_journal,
            "action_observations": {"order": {"status": "ok", "data": {"order_no": "100000001", "amount": 32000}}},
        })
        runtime.config = replace(runtime.config, run_timeout_seconds=.5)
        with pytest.raises(TimeoutError):
            await runtime.run(text="주문 100000001 전액 환불해 주세요.", thread_id="post", event_id="refund",
                              pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        assert len(calls) == 1
        assert await runtime.abandon("post", "refund")
        state = await runtime.inspect("post", pack.snapshot)
        assert not state.next
        assert state.values["tc_id"] == "L-11"
        assert "고객 A" in state.values["case_brief"]
        assert state.values["policy_reads"] == {"D008": True}
        assert state.values["action_requests"] == old_journal
        assert state.values["action_observations"]["order"]["data"]["order_no"] == "100000001"
        pending = next(m for m in state.values["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        assert pending.tool_calls[0]["args"] == submit
        recovered = next(m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "pending-post")
        assert json.loads(recovered.content)["status"] == "unknown"
        assert any(isinstance(m, HumanMessage) and m.id == "refund" for m in state.values["messages"])
    good = ScriptedModel(responses=[AIMessage(content=json.dumps(answer("이전 요청의 접수 결과는 아직 미확인입니다.", [])))])
    async with AgentRuntime(config(packs_root=tmp_path), checkpointer=saver, model=good) as runtime:
        result = await runtime.run(text="접수됐나요?", thread_id="post", event_id="followup",
                                   pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        assert "미확인" in result.text
        assert len(calls) == 1


async def test_new_turn_never_reuses_previous_structured_answer_after_failure():
    saver = InMemorySaver()
    model = ScriptedModel(responses=[AIMessage(content=json.dumps(answer("첫 번째 답변", [])))])
    async with AgentRuntime(config(), checkpointer=saver, model=model) as runtime:
        await runtime.run(text="first", thread_id="stale", event_id="first", pack_ref="equipment_demo",
                          deadline=datetime.now(UTC) + timedelta(seconds=20))
        with pytest.raises(AssertionError):
            await runtime.run(text="second", thread_id="stale", event_id="second", pack_ref="equipment_demo",
                              deadline=datetime.now(UTC) + timedelta(seconds=20))
        state = await runtime.inspect("stale", "equipment_demo")
        assert state.values.get("structured_response") is None
