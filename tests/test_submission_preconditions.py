import json
import shutil
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from kakao_bot.runtime import load_pack, lock_pack
from kakao_bot.runtime.action_api import ActionAPI


@pytest.mark.parametrize("shipping,allowed", [
    (None, False),
    ({"api_id": "API-11", "status": "failed", "parameters": {"order_no": "100000001"}, "data": []}, False),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "other"}, "data": []}, False),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "100000001"}, "data": []}, True),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "100000001"},
      "data": [{"order_no": "100000001", "status": "READY", "picked_up_at": None}]}, True),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "100000001"},
      "data": [{"order_no": "100000001", "status": "READY", "picked_up_at": "2037-06-30T09:00:00+09:00"}]}, False),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "100000001"},
      "data": [{"order_no": "100000001", "status": "IN_TRANSIT", "picked_up_at": None}]}, False),
    ({"api_id": "API-11", "status": "ok", "parameters": {"order_no": "100000001"},
      "data": [{"order_no": "other", "status": "READY", "picked_up_at": None}]}, False),
])
async def test_address_change_requires_same_order_shipping_evidence(tmp_path, monkeypatch, shipping, allowed):
    shutil.copytree("domain_packs/gyodong", tmp_path / "gyodong", ignore=shutil.ignore_patterns("__pycache__"))
    lock_pack(tmp_path / "gyodong")
    pack = load_pack(tmp_path, "gyodong")
    monkeypatch.setenv("ACTION_API_WRITES_ENABLED", "true")
    calls = []

    async def fake_request(self, api_id, **kwargs):
        if api_id == "API-13":
            return {"api_id": api_id, "parameters": kwargs["parameters"], "status": "ok", "data": [
                {"member_id": "MB-A", "name": "고객 A", "phone": "019-1111-1234", "status": "ACTIVE"}]}
        if api_id == "API-07":
            return {"api_id": api_id, "parameters": kwargs["parameters"], "status": "ok", "data":
                {"order_no": "100000001", "member_id": "MB-A", "status": "PAID", "amount": 32000}}
        calls.append(kwargs)
        return {"status": "submitted", "data": {"request_id": "ACR-test", "request_status": "SUBMITTED"}}

    monkeypatch.setattr(ActionAPI, "request", fake_request)
    brief = "TC ID: L-03\ncase metadata\n- id: L-03\n- 상담 인입 일시: 2037-06-30 10:00\n- 인입창구: 채팅\n- 인입 계정: 회원 로그인 · 고객 A · 019-****-1234\n\n주문 100000001"
    state = {"case_brief": brief, "tc_id": "L-03", "messages": [HumanMessage(content="이 주소로 변경해주세요.")],
             "policy_reads": {"D015": True}, "action_requests": {},
             "action_observations": {"order": {"status": "ok", "data": {"order_no": "100000001", "status": "PAID"}}}}
    if shipping is not None:
        state["action_observations"]["shipping"] = shipping
    submit = next(tool for tool in pack.python_tools() if tool.name == "action_submit")
    result = await submit.coroutine(api_id="API-24", parameters={"order_no": "100000001"},
        body={"new_address": "경기 성남시 테스트로 1, 101호"}, policy_source_ids=["D015"],
        policy_reason="고객이 확인한 주문의 집하 전 배송지 변경 요청입니다.",
        customer_confirmation="이 주소로 변경해주세요.", runtime=SimpleNamespace(state=state, tool_call_id="change"))
    value = json.loads(result.update["messages"][0].content)
    assert len(calls) == int(allowed)
    assert value["status"] == ("submitted" if allowed else "needs_input")
