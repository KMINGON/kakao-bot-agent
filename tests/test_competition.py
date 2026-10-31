import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from kakao_bot.runtime import AgentRuntime, load_pack, lock_pack
from kakao_bot.runtime.action_api import ActionAPI
from kakao_bot.runtime.competition import extract_tc_id, recover_action_requests, submission_identity
from test_runtime import ScriptedModel, config, tool_call


def final(text="확인했습니다.", sources=None):
    return AIMessage(content=json.dumps({"status":"answered","answer":text,
        "claims":[],"evidence_ids":sources or [],"missing_fields":[]}))


def official_brief(tc, name="계정 A", text="주문 100000001 전액 환불해 주세요."):
    return (f"TC ID: {tc}\ncase metadata\n- id: {tc}\n- 상담 인입 일시: 2037-06-26 10:00\n"
            f"- 인입창구: 채팅\n- 인입 계정: 회원 로그인 · {name} · 019-****-1234\n\n{text}")


def own_lookup(api_id, parameters, *, name="계정 A", member_id="MB-A"):
    data = ([{"member_id":member_id,"name":name,"phone":"019-1111-1234","status":"ACTIVE"}]
            if api_id=="API-13" else {"order_no":"100000001","member_id":member_id,"amount":32000})
    return {"status":"ok","api_id":api_id,"parameters":parameters,"data":data,
            "evidence":[{"source_id":api_id}],"effect":"read_only"}


@pytest.fixture
def pack(tmp_path):
    shutil.copytree("domain_packs/gyodong", tmp_path / "gyodong", ignore=shutil.ignore_patterns("__pycache__"))
    lock_pack(tmp_path / "gyodong")
    return load_pack(tmp_path,"gyodong")


@pytest.mark.parametrize("text,expected",[
    ("TC ID: L-47\n환불 문의", "L-47"),
    ("[TC: Y-07-b] 상담입니다", "Y-07-b"),
    ("테스트 케이스 ID: PRIVATE_901.2", "PRIVATE_901.2"),
    ('{"tc_id":"Hidden-123","message":"문의"}', "Hidden-123"),
    ("[R-05-a]\n문의", "R-05-a"),
    ("L-47\n문의", "L-47"),
    ("다른 L-47 사례처럼 해주세요", None),
    ("TC ID: ../../etc", None),
    ("TC ID: L-47/other", None),
])
def test_tc_is_explicit_not_guessed(text,expected):
    assert extract_tc_id(text)==expected


@pytest.mark.asyncio
async def test_same_tc_followup_restart_summary_and_new_tc_isolation(pack,monkeypatch):
    calls=[]
    async def fake(self, api_id, *, tc_id, parameters, **kwargs):
        calls.append((tc_id,parameters))
        if api_id=="API-13":
            return own_lookup(api_id,parameters,name=parameters["name"],member_id="MB-A" if tc_id=="L-47" else "MB-B")
        return {"status":"ok","api_id":api_id,"tc_id":tc_id,"data":[],
                "evidence":[{"source_id":api_id}],"effect":"read_only"}
    monkeypatch.setattr(ActionAPI,"request",fake)
    saver=InMemorySaver()
    model=ScriptedModel(responses=[tool_call("action_get",{"api_id":"API-08","parameters":{"member_id":"MB-A"}}),final(sources=["API-08"])])
    async with AgentRuntime(config(packs_root=pack.root.parent),checkpointer=saver,model=model) as runtime:
        await runtime.run(text=official_brief("L-47"),thread_id="customer",event_id="one",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        state=await runtime.inspect("customer",pack.snapshot)
        assert state.values["tc_id"]=="L-47" and state.values["action_observations"]
        # Summary/history trimming cannot remove the durable TC and initial case envelope.
        graph,_=await runtime.graph_for(pack.snapshot)
        await graph.aupdate_state({"configurable":{"thread_id":"customer"}},
            {"messages":[RemoveMessage(id=REMOVE_ALL_MESSAGES),HumanMessage(content="요약된 이전 대화",id="summary")]})
    model=ScriptedModel(responses=[tool_call("case_context",{}),final(sources=["CASE_CONTEXT"]),
        tool_call("action_get",{"api_id":"API-08","parameters":{"member_id":"MB-B"}}),final(sources=["API-08"])])
    async with AgentRuntime(config(packs_root=pack.root.parent),checkpointer=saver,model=model) as runtime:
        await runtime.run(text="이어서 확인해 주세요",thread_id="customer",event_id="two",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        state=await runtime.inspect("customer",pack.snapshot)
        assert state.values["tc_id"]=="L-47" and "2037-06-26" in state.values["case_brief"]
        assert any("MB-A" in key for key in state.values["action_observations"])
        await runtime.run(text=official_brief("Y-16", "계정 B"),thread_id="customer",event_id="three",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        state=await runtime.inspect("customer",pack.snapshot)
        assert state.values["tc_id"]=="Y-16"
        assert all("MB-A" not in key for key in state.values["action_observations"])
        assert "MB-A" not in str(state.values["action_observations"])
        assert "계정 A" not in state.values["case_brief"]
        assert all("계정 A" not in str(m.content) for m in state.values["messages"])
        assert state.values["policy_reads"]=={} and state.values["action_requests"]=={}
    assert calls==[("L-47",{"name":"계정 A"}),("L-47",{"member_id":"MB-A"}),
                  ("Y-16",{"name":"계정 B"}),("Y-16",{"member_id":"MB-B"})]


@pytest.mark.asyncio
async def test_missing_tc_cannot_call_api_and_evidence_is_not_fabricated(pack,monkeypatch):
    async def forbidden(*args,**kwargs):
        raise AssertionError("No HTTP without TC")
    monkeypatch.setattr(ActionAPI,"request",forbidden)
    model=ScriptedModel(responses=[tool_call("action_get",{"api_id":"API-07","parameters":{"order_no":"100"}}),final("TC ID를 알려 주세요.")])
    async with AgentRuntime(config(packs_root=pack.root.parent),checkpointer=InMemorySaver(),model=model) as runtime:
        result=await runtime.run(text="주문을 확인해주세요",thread_id="missing",event_id="missing",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        state=await runtime.inspect("missing",pack.snapshot)
        result_json=json.loads(next(m for m in state.values["messages"] if isinstance(m,ToolMessage)).content)
        assert result_json["status"]=="needs_input" and result_json["evidence"]==[]
        assert result.answer["evidence_ids"]==[]


@pytest.mark.asyncio
async def test_policy_tool_result_registers_seen_source_in_real_graph(pack):
    model=ScriptedModel(responses=[tool_call("read_policy",{"source_id":"D001","start_line":1}),final(sources=["D001"])])
    async with AgentRuntime(config(packs_root=pack.root.parent),checkpointer=InMemorySaver(),model=model) as runtime:
        result=await runtime.run(text="TC: L-32\n이관 기준",thread_id="policy",event_id="p1",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        assert result.answer["evidence_ids"]==["D001"]
        state=await runtime.inspect("policy",pack.snapshot)
        assert state.values["policy_reads"]=={"D001":True}


@pytest.mark.asyncio
async def test_post_requires_policy_target_consent_and_reuses_receipt(pack,monkeypatch):
    monkeypatch.setenv("ACTION_API_WRITES_ENABLED","true")
    calls=[]
    target_available=False
    async def fake(self,api_id,**kwargs):
        if api_id=="API-13" or api_id=="API-07" and target_available:
            return own_lookup(api_id,kwargs["parameters"])
        if api_id=="API-07":
            return {"status":"needs_input","evidence":[]}
        calls.append(kwargs)
        return {"status":"submitted","data":{"refund_id":"RF1","request_status":"SUBMITTED"},"evidence":[{"source_id":api_id}]}
    monkeypatch.setattr(ActionAPI,"request",fake)
    submit={t.name:t for t in pack.python_tools()}["action_submit"]
    state={"tc_id":"L-11","messages":[HumanMessage(content="주문 100000001 전액 환불해 주세요.")],
           "case_brief":official_brief("L-11"), "policy_reads":{},"action_observations":{},"action_requests":{}}
    runtime=SimpleNamespace(state=state,tool_call_id="submit")
    args={"api_id":"API-29","parameters":{},"body":{"order_no":"100000001","amount":32000,"reason":"하자 환불"},
          "policy_source_ids":["D008"],"policy_reason":"확인된 하자와 전액 환불 요청을 확인했습니다.",
          "customer_confirmation":"전액 환불해 주세요.","runtime":runtime}
    blocked=await submit.coroutine(**args)
    assert not calls and json.loads(blocked.update["messages"][0].content)["status"]=="needs_input"
    state["policy_reads"]={"D008":True}
    blocked=await submit.coroutine(**args)
    assert not calls
    target_available=True
    state["action_observations"]={"order":own_lookup("API-07",{"order_no":"100000001"})}
    state["messages"].append(HumanMessage(content="환불하지 마세요. 주소만 변경해주세요."))
    wrong=await submit.coroutine(**{**args,"customer_confirmation":"없던 요청"})
    assert not calls
    state["messages"].append(HumanMessage(content="전액 환불해 주세요."))
    accepted=await submit.coroutine(**args)
    state["action_requests"].update(accepted.update["action_requests"])
    reused=await submit.coroutine(**args)
    assert len(calls)==1 and json.loads(reused.update["messages"][0].content)["reused_receipt"]
    changed=await submit.coroutine(**{**args,"body":{**args["body"],"reason":"조금 바뀐 사유"}})
    assert len(calls)==1 and json.loads(changed.update["messages"][0].content)["status"]=="needs_input"


@pytest.mark.asyncio
async def test_new_tc_drops_prior_policy_permission(pack):
    model=ScriptedModel(responses=[tool_call("read_policy",{"source_id":"D008"}),final(sources=["D008"]),
        final("새 상담입니다.")])
    async with AgentRuntime(config(packs_root=pack.root.parent),checkpointer=InMemorySaver(),model=model) as runtime:
        for tc,event in [("L-11","old"),("L-47","new")]:
            await runtime.run(text=f"TC ID: {tc}",thread_id="switch",event_id=event,pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        state=await runtime.inspect("switch",pack.snapshot)
        assert state.values["policy_reads"]=={}


@pytest.mark.asyncio
async def test_same_lane_distinct_escalations_and_exact_retry_body(pack,monkeypatch):
    monkeypatch.setenv("ACTION_API_WRITES_ENABLED","true")
    calls=[]
    async def fake(self,api_id,**kwargs):
        calls.append(kwargs)
        return {"status":"submitted","data":{"escalation_id":f"ESC-{len(calls)}","request_status":"SUBMITTED"},
                "evidence":[{"source_id":api_id}]}
    monkeypatch.setattr(ActionAPI,"request",fake)
    submit={t.name:t for t in pack.python_tools()}["action_submit"]
    state={"tc_id":"L-11","messages":[HumanMessage(content="환불 요청과 안전 확인이 필요합니다.")],
           "policy_reads":{"D001":True},"action_requests":{}}
    runtime=SimpleNamespace(state=state,tool_call_id="escalate")
    args={"api_id":"API-35","parameters":{},"policy_source_ids":["D001"],
          "policy_reason":"동일 로트 복수 신고와 공급자 안전 확인이 필요합니다.",
          "customer_confirmation":"","runtime":runtime}
    for issue,lane,reason in [("100:안전","LANE2_HANDOFF","100 주문의 로트 안전 확인"),
                               ("200:안전","LANE2_HANDOFF","200 주문의 별도 로트 안전 확인"),
                               ("100:광고","LANE2_HANDOFF","100 주문 상품 광고 정정"),
                               ("100:배상","LANE3_CLAIM","100 주문 별도 손해배상 검토")]:
        result=await submit.coroutine(**args,issue_key=issue,body={"lane":lane,"reason":reason})
        state["action_requests"].update(result.update["action_requests"])
    assert len(calls)==4
    reused=await submit.coroutine(**args,issue_key="100:안전",body={"lane":"LANE2_HANDOFF","reason":"100 주문의 로트 안전 확인"})
    assert len(calls)==4 and json.loads(reused.update["messages"][0].content)["reused_receipt"]
    rejected=await submit.coroutine(**args,issue_key="100:안전",replace_previous=True,
        body={"lane":"LANE2_HANDOFF","reason":"같은 사안을 재생성한 다른 문장"})
    assert len(calls)==4 and json.loads(rejected.update["messages"][0].content)["status"]=="needs_input"
    assert all("issue_key" not in call["body"] for call in calls)


@pytest.mark.asyncio
async def test_real_sdk_summary_and_scratch_are_cleared_on_new_tc(pack):
    model=ScriptedModel(responses=[
        tool_call("write_file",{"file_path":"/scratch/private.txt","content":"이전고객의 비공개 배송지 OLD-CUSTOMER"}),
        final("기록했습니다."), final("새 상담입니다.")])
    async with AgentRuntime(config(packs_root=pack.root.parent,summary_trigger_tokens=1000,summary_keep_messages=2),
                            checkpointer=InMemorySaver(),model=model) as runtime:
        await runtime.run(text="TC ID: L-47\nOLD-CUSTOMER 상담기록 " + "이전 고객의 주문 설명 "*600,
            thread_id="private-summary",event_id="old",pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        before=await runtime.inspect("private-summary",pack.snapshot)
        assert model.summaries>=1 and before.values.get("_summarization_event")
        assert "/scratch/private.txt" in before.values["files"]
        boundary=len(model.requests)
        await runtime.run(text="TC ID: Y-16\n새 고객 문의",thread_id="private-summary",event_id="new",
            pack_ref=pack.snapshot,deadline=datetime.now(UTC)+timedelta(seconds=20))
        after=await runtime.inspect("private-summary",pack.snapshot)
        assert after.values.get("_summarization_event") is None
        assert not after.values.get("files")
        assert all("OLD-CUSTOMER" not in str(request) and "실제 수량은 원본 도구로" not in str(request)
                   for request in model.requests[boundary:])


def refund_arguments():
    return {"api_id": "API-29", "parameters": {},
            "body": {"order_no": "100", "amount": 32000, "reason": "원문 사유 그대로"},
            "policy_source_ids": ["D008"], "policy_reason": "확인된 하자에 대한 전액 환불 요청",
            "customer_confirmation": "전액 환불해 주세요."}


@pytest.mark.parametrize("status", [None, "unknown", "uncertain", "failed", "needs_input", "submitted"])
def test_recovery_only_journals_unknown_requests_and_preserves_existing(status):
    args = refund_arguments()
    original = "TC: L-11\n주문 100 전액 환불해 주세요."
    messages = [HumanMessage(content=original), tool_call("action_submit", args, "interrupted")]
    if status:
        messages.append(ToolMessage(content=json.dumps({"status": status}), tool_call_id="interrupted"))
    state = {"tc_id": "L-11", "messages": messages, "action_requests": {}}
    recovered = recover_action_requests(state)
    if status not in {None, "unknown", "uncertain"}:
        assert recovered == {}
        return
    key, canonical = submission_identity(args["api_id"], args["parameters"], args["body"])
    record = recovered[key]
    assert record["canonical"] == canonical
    assert record["policy_source_ids"] == args["policy_source_ids"]
    assert record["policy_reason"] == args["policy_reason"]
    assert record["customer_confirmation"] == original
    assert messages[1].tool_calls[0]["args"] == args
    assert record["result"]["status"] in {"unknown", "uncertain"}
    state["action_requests"][key] = {"canonical": canonical, "result": {"status": "submitted"}}
    assert recover_action_requests(state) == {}
    state["action_requests"] = {}
    state["tc_id"] = "Y-16"
    assert recover_action_requests(state) == {}


def test_recovery_keeps_distinct_issue_keys_in_same_escalation_lane():
    args = {**refund_arguments(), "api_id": "API-35", "body": {"lane": "LANE2_HANDOFF", "reason": "안전 확인"}}
    calls = [tool_call("action_submit", {**args, "issue_key": issue}, issue)
             for issue in ("100:안전", "200:안전")]
    recovered = recover_action_requests({"tc_id": "L-11", "messages": calls})
    assert len(recovered) == 2
    assert {record["issue_key"] for record in recovered.values()} == {"100:안전", "200:안전"}


@pytest.mark.parametrize("model_quote", ["", "환불 요청"])
def test_recovery_of_automatic_quote_keeps_call_time_customer_and_original_arguments(model_quote):
    original = "이 주문을 전액 환불해 주세요."
    args = {**refund_arguments(), "customer_confirmation": model_quote}
    call = tool_call("action_submit", args, "interrupted-auto")
    state = {"tc_id": "L-11", "messages": [HumanMessage(content=original), call,
        ToolMessage(content=json.dumps({"status": "unknown"}), tool_call_id="interrupted-auto"),
        HumanMessage(content="지금 접수됐나요?")], "action_requests": {}}
    record = next(iter(recover_action_requests(state).values()))
    assert record["customer_confirmation"] == original
    assert call.tool_calls[0]["args"]["customer_confirmation"] == model_quote
    assert json.loads(record["canonical"]) == {"parameters": args["parameters"], "body": args["body"]}


@pytest.mark.asyncio
async def test_real_sdk_interrupted_submit_restores_exact_retry_and_isolates_new_tc(pack, monkeypatch):
    monkeypatch.setenv("ACTION_API_WRITES_ENABLED", "true")
    calls = []

    async def fake(self, api_id, **kwargs):
        if api_id in {"API-13", "API-07"}:
            return own_lookup(api_id, kwargs["parameters"])
        calls.append(kwargs)
        return {"status": "submitted", "data": {"refund_id": "RF-1", "request_status": "SUBMITTED"},
                "evidence": [{"source_id": api_id}]}

    monkeypatch.setattr(ActionAPI, "request", fake)
    args = refund_arguments()
    args = {**args, "body": {**args["body"], "order_no": "100000001"}}
    changed = {**args, "body": {**args["body"], "reason": "달라진 사유"}, "replace_previous": True}
    model = ScriptedModel(responses=[final(), tool_call("action_submit", changed, "changed"),
        tool_call("case_context", {}, "unknown-context"), tool_call("action_submit", args, "exact-retry"),
        final(sources=["API-29"]), final(), final()])
    async with AgentRuntime(config(packs_root=pack.root.parent), checkpointer=InMemorySaver(), model=model) as runtime:
        await runtime.run(text=official_brief("L-11"), thread_id="recover-submit", event_id="intro",
                          pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        graph, _ = await runtime.graph_for(pack.snapshot)
        await graph.aupdate_state({"configurable": {"thread_id": "recover-submit"},
                                  "metadata": {"event_id": "intro", "pack_digest": pack.digest}}, {
            "messages": [tool_call("action_submit", args, "pending-post")],
            "policy_reads": {"D008": True},
            "action_observations": {"order": own_lookup("API-07", {"order_no": "100000001"})},
        })
        # The real core repair produces the unknown marker; no submission journal was checkpointed.
        assert await runtime.abandon("recover-submit", "intro")
        before = await runtime.inspect("recover-submit", pack.snapshot)
        assert not before.values["action_requests"]
        marker = next(m for m in before.values["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "pending-post")
        assert json.loads(marker.content)["error"] == "interrupted_tool_call"
        await runtime.run(text="접수를 다시 확인해 주세요.", thread_id="recover-submit", event_id="retry",
                          pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        after = await runtime.inspect("recover-submit", pack.snapshot)
        results = {m.tool_call_id: json.loads(m.content) for m in after.values["messages"] if isinstance(m, ToolMessage)}
        assert results["changed"]["status"] == "needs_input"
        unknown = results["unknown-context"]["data"]["receipts"][0]
        assert unknown["result"]["status"] == "unknown"
        assert unknown["request"] == {"parameters": args["parameters"], "body": args["body"]}
        assert len(calls) == 1 and calls[0]["body"] == args["body"]
        key, _ = submission_identity(args["api_id"], args["parameters"], args["body"])
        assert after.values["action_requests"][key]["result"]["status"] == "submitted"
        for event, text in [("new", "TC: Y-16\n새 상담"), ("new-followup", "이어서 확인해 주세요")]:
            await runtime.run(text=text, thread_id="recover-submit", event_id=event,
                              pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
            current = await runtime.inspect("recover-submit", pack.snapshot)
            assert current.values["tc_id"] == "Y-16" and current.values["action_requests"] == {}
