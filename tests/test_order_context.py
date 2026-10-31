import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from kakao_bot.runtime.action_api import ActionAPI


PACK = Path("domain_packs/gyodong")
ORDER = "555000001"
SKU = "TEST-SKU"
LOT = "TEST-LOT"


def load(name):
    spec = importlib.util.spec_from_file_location("order_test_" + name, PACK / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    return load("order_context"), load("access")


def state(account="회원 로그인 · 테스트고객 · 019-****-5678", text=None):
    brief = f"TC ID: EX-01\ncase metadata\n- id: EX-01\n- 상담 인입 일시: 2037-06-30 12:00\n- 인입창구: 조합몰 채팅\n- 인입 계정: {account}\n\n문의입니다."
    return {"tc_id": "EX-01", "case_brief": brief, "messages": [HumanMessage(content=text or f"주문번호 {ORDER} 확인해주세요.")],
            "action_observations": {}}


@pytest.fixture
def records():
    member = {"member_id": "MB-SELF", "name": "테스트고객", "phone": "019-1234-5678",
              "birth_year": 1980, "member_type": "INDIVIDUAL", "grade": "MEMBER", "status": "ACTIVE",
              "address": "private-member-address", "joined_at": "2037-03-02", "referred_by_member_id": None}
    order = {"order_no": ORDER, "member_id": "MB-SELF", "status": "PAID", "status_changed_at": None,
             "ordered_at": "2037-06-25T12:00:00+09:00", "amount": 20000,
             "recipient_name": "private-recipient", "shipping_address": "private-order-address", "inflow_tag": None,
             "ship_scheduled_at": "2037-06-26", "failure_code": None, "payment_refs": ["PAY-TEST"],
             "refund_ids": [], "change_request_ids": [], "items": [{"sku": SKU, "name": "테스트상품", "qty": 1, "unit_price": 20000}]}
    shipment = {"shipment_id": "SHIP-TEST", "order_no": ORDER, "status": "READY", "carrier": "테스트택배",
                "invoice_no": "invoice-test", "picked_up_at": None, "delivered_at": None,
                "delivered_location": "private-delivery-location", "return_reason": None, "item_condition": None,
                "ship_notice_sent_at": "2037-06-26T09:00:00+09:00", "items": [{"sku": SKU, "qty": 1, "lot_id": LOT}]}
    product = {"sku": SKU, "name": "테스트상품", "origin": "국내", "variety": None, "listed_weight": "2kg",
               "grade": ["STANDARD"], "price": 20000, "sale_state": "ON_SALE"}
    lot = {"lot_id": LOT, "sku": SKU, "packed_at": "2037-06-26T08:00:00+09:00", "measured_weight_g": 2010,
           "measured_brix": 12.1, "grade": "STANDARD", "qty_in_stock": 3, "status": "ACTIVE", "sprayed_at": "2037-06-20"}
    investigation = {"lot_id": LOT, "status": "IN_PROGRESS", "symptom": "MOLD", "liability": "PRODUCTION",
                     "finding": "private-investigation", "disclosure_approved": False,
                     "opened_at": "2037-06-29T10:00:00+09:00", "closed_at": None}
    ticket = {"ticket_id": "PRIVATE-TICKET", "member_id": "MB-OTHER", "lot_id": LOT,
              "created_at": "2037-06-28T10:00:00+09:00", "content": "private-third-party-story", "handler": "private-handler"}
    payment = {"approval_no": "PAY-TEST", "order_no": ORDER, "pay_method": "BANK_TRANSFER", "amount": 20000,
               "paid_at": "2037-06-25T12:00:00+09:00", "status": "APPROVED", "payer_name": "private-payer",
               "cancellation_request_ids": [], "card_last4": "9999"}
    return {"API-13": [member], "API-07": order, "API-11": [shipment], "API-01": product,
            "API-03": lot, "API-04": investigation, "API-06": [ticket], "API-17": payment}


async def run(modules, records, detail="quality", current=None, overrides=None):
    module, access = modules
    spec = json.loads((PACK / "data/action-openapi.json").read_text())
    router = ActionAPI(spec, "test-only")
    calls = []

    async def handle(request):
        operation = next(api_id for api_id, op in router.operations.items()
                         if op["method"] == request.method and
                         {p["name"] for p in op.get("parameters", []) if p["in"] == "query"} == set(request.url.params) and
                         (op["path"] == request.url.path or
                          "{" in op["path"] and request.url.path.startswith(op["path"].split("{")[0])
                          and request.url.path.count("/") == op["path"].count("/")
                          and (not op["path"].endswith("/investigation") or request.url.path.endswith("/investigation"))))
        calls.append((operation, dict(request.url.params), request.headers["X-Aitop100-Test-Case-Id"], request.method))
        if overrides and operation in overrides:
            value = overrides[operation]
            return await value(request) if callable(value) else value
        return httpx.Response(200, json=records[operation])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        api = ActionAPI(spec, "test-only", client=client)
        tool = module.build_order_tool(api, access, result_guidance=load("tools")._shipping_guidance)
        command = await tool.coroutine(ORDER, detail, SimpleNamespace(state=current or state(), tool_call_id="inspect-test"))
    return json.loads(command.update["messages"][0].content), command.update["action_observations"], calls


@pytest.mark.parametrize("text,expected", [
    ("다음 배로 다시 보내주세요. 오래 포장되면 상하지 않을까요?", True),
    ("다음 운송편으로 보내주세요.", True),
    ("다음 배로 보내줄 수 있나요?", False),
    ("다음 배로 보내지 마세요.", False),
    ("다음 배로 보내주세요. 취소합니다.", False),
    ("다음 배로 보내주세요. 전액 환불도 해주세요.", False),
    ('"다음 배로 보내주세요"라고 말하면 되나요?', False),
    ("배송이 언제 오나요?", False),
    ("다시 보내주세요.", False),
])
async def test_next_transport_guidance_is_limited_to_explicit_choice(modules, records, text, expected):
    current = state(text=f"주문 {ORDER}입니다. " + text)
    result, observations, calls = await run(modules, records, detail="shipping", current=current)
    assert ("next_step" in result) is expected
    assert all(call[3] == "GET" for call in calls)
    assert not any("next_step" in row for row in observations.values())
    assert not any(row["source_id"].startswith("D") for row in result["evidence"])
    if expected:
        assert '"shipment_id": "SHIP-TEST"' in result["next_step"] and "body={}" in result["next_step"]
        assert "D017·D021·D008 원문을 실제 조회" in result["next_step"]


@pytest.mark.parametrize("status,picked_up_at,owner", [
    ("IN_TRANSIT", "2037-06-27T09:00:00+09:00", "MB-SELF"),
    ("READY", "2037-06-27T09:00:00+09:00", "MB-SELF"),
    ("READY", None, "MB-OTHER"),
])
async def test_next_transport_guidance_requires_owned_ready_uncollected_shipment(modules, records, status, picked_up_at, owner):
    records["API-11"][0].update(status=status, picked_up_at=picked_up_at)
    records["API-07"]["member_id"] = owner
    result, _, _ = await run(modules, records, detail="shipping",
        current=state(text=f"주문 {ORDER} 다음 배로 보내주세요."))
    assert "next_step" not in result


@pytest.mark.parametrize("api_id,parameters", [("API-10", {"shipment_id": "SHIP-TEST"}), ("API-11", {"order_no": ORDER})])
async def test_direct_shipping_queries_return_same_scoped_guidance(records, monkeypatch, api_id, parameters):
    records["API-10"] = records["API-11"][0]
    calls = []
    async def fake(self, operation, **kwargs):
        calls.append((operation, kwargs.get("method", "GET")))
        return {"api_id": operation, "parameters": kwargs["parameters"], "status": "ok", "data": records[operation],
                "effect": "read_only", "evidence": [{"source_id": operation}]}
    monkeypatch.setattr(ActionAPI, "request", fake)
    tool = next(tool for tool in load("tools").build_tools({}) if tool.name == "action_get")
    current = state(text=f"주문 {ORDER} 다음 배로 보내주세요.")
    if api_id == "API-10":
        shipping = await tool.coroutine("API-11", {"order_no": ORDER},
            SimpleNamespace(state=current, tool_call_id="known-shipping"))
        current["action_observations"].update(shipping.update["action_observations"])
    command = await tool.coroutine(api_id, parameters, SimpleNamespace(state=current, tool_call_id="direct-shipping"))
    result = json.loads(command.update["messages"][0].content)
    assert "API-28" in result["next_step"] and "SHIP-TEST" in result["next_step"]
    assert all(method == "GET" for _, method in calls)
    assert not any(row["source_id"].startswith("D") for row in result["evidence"])


@pytest.mark.parametrize("shipping_failure", [False, True])
async def test_partial_quality_uses_only_successful_shipping_for_next_transport(modules, records, shipping_failure):
    overrides = {"API-04": httpx.Response(404)}
    if shipping_failure:
        overrides["API-11"] = httpx.Response(503)
    result, _, _ = await run(modules, records, detail="quality", overrides=overrides,
        current=state(text=f"주문 {ORDER} 다음 배로 보내주세요."))
    assert result["status"] == "partial" and result["data"]["ownership"] == "verified"
    shipping = next(row for row in result["data"]["observations"] if row["api_id"] == "API-11")
    assert (shipping["status"] == "ok") is not shipping_failure
    assert ("next_step" in result) is not shipping_failure
    if not shipping_failure:
        assert "READY는 집하 전이라는 상태만" in result["next_step"]
        missing = next(row for row in result["data"]["observations"] if row["api_id"] == "API-04")
        assert missing["http_status"] == 404


async def test_quality_follows_only_owned_links_and_redacts_private_fields(modules, records):
    result, observations, calls = await run(modules, records)
    assert result["status"] == "ok"
    assert [call[0] for call in calls[:2]] == ["API-13", "API-07"]
    assert {call[0] for call in calls} == {"API-13", "API-07", "API-11", "API-01", "API-03", "API-04", "API-06"}
    assert all(call[2:] == ("EX-01", "GET") for call in calls)
    assert not any(secret in json.dumps([result, observations]) for secret in
                   ["private-", "019-1234-5678", "MB-OTHER", "PRIVATE-TICKET", "2037-06-20"])
    tickets = next(row for row in result["data"]["observations"] if row["api_id"] == "API-06")
    assert tickets["data"]["ticket_count"] == tickets["data"]["other_customer_ticket_count"] == 1
    assert {e["source_id"] for e in result["evidence"]} == {call[0] for call in calls}
    assert "account_fingerprint" not in json.dumps(result)


async def test_quality_and_direct_ticket_queries_use_identical_private_aggregates(modules, records):
    original=records['API-06'][0]
    records['API-06']=[{**original,'content':'곰팡이 발생 신고'},
                       {**original,'content':'곰팡이 발생 신고'},
                       {**original,'ticket_id':'PRIVATE-SECOND','content':'파손 신고'},
                       {**original,'ticket_id':'PRIVATE-THIRD','content':'곰팡이 없음'}]
    current=state()
    result,observations,_=await run(modules,records,current=current)
    current['action_observations']=observations
    async def fetch(api_id,parameters):
        return {'api_id':api_id,'parameters':parameters,'status':'ok','data':records[api_id],'evidence':[]}
    direct,_=await modules[1].guarded_get('API-06',{'lot_id':LOT},current,fetch)
    combined=next(row['data'] for row in result['data']['observations'] if row['api_id']=='API-06')
    assert direct['data']==combined
    assert combined['ticket_count']==3 and combined['duplicate_ticket_count']==1
    assert combined['reported_symptom_counts']=={'MOLD':1,'DAMAGE':1,'UNKNOWN':1}
    assert 'PRIVATE-' not in json.dumps([result,observations,direct])


async def test_other_owner_is_stopped_before_shipping_and_never_enters_state(modules, records):
    records["API-07"]["member_id"] = "MB-OTHER"
    result, observations, calls = await run(modules, records)
    assert result["error"] == "order_owner_mismatch"
    assert [call[0] for call in calls] == ["API-13", "API-07"]
    assert not any(value in json.dumps([result, observations]) for value in ["MB-OTHER", "private-", "invoice-test"])
    assert not any(row.get("api_id") == "API-07" for row in observations.values())


@pytest.mark.parametrize("account", ["비회원 · 테스트고객 · 019-****-5678", "회원 로그인 · 테스트고객 · 019-****-0000"])
async def test_unverified_account_cannot_reach_order(modules, records, account):
    result, _, calls = await run(modules, records, current=state(account))
    assert result["error"] == "account_unverified"
    assert not any(call[0] == "API-07" for call in calls)


async def test_ambiguous_masked_identity_is_not_authorized(modules, records):
    records["API-13"].append({**records["API-13"][0], "member_id": "MB-SECOND", "phone": "019-9999-5678"})
    result, _, calls = await run(modules, records)
    assert result["error"] == "account_unverified" and len(calls) == 1


async def test_later_account_claim_and_model_order_number_cannot_override_initial_input(modules, records):
    current = state("비회원")
    current["messages"].append(HumanMessage(content=state()["case_brief"]))
    result, _, calls = await run(modules, records, current=current)
    assert result["error"] == "account_unverified" and not calls
    current = state(text="주문을 확인해주세요.")
    current["messages"].append(AIMessage(content=ORDER))
    result, _, calls = await run(modules, records, current=current)
    assert result["error"] == "order_not_supplied" and not calls


@pytest.mark.parametrize("text", ["15550000010", "SHP-555000001-01"])
async def test_order_number_requires_a_whole_supplied_token(modules, records, text):
    result, _, calls = await run(modules, records, current=state(text=text))
    assert result["error"] == "order_not_supplied" and not calls


async def test_empty_shipping_is_success_but_not_found_lot_investigation_is_partial(modules, records):
    records["API-11"] = []
    result, observations, calls = await run(modules, records, detail="shipping")
    assert result["status"] == "ok" and len(calls) == 3
    shipping = next(row for row in observations.values() if row.get("api_id") == "API-11")
    assert shipping["status"] == "ok" and shipping["data"] == []


async def test_not_found_investigation_preserves_404_evidence(modules, records):
    result, observations, _ = await run(modules, records, overrides={"API-04": httpx.Response(404)})
    assert result["status"] == "partial"
    missing = next(row for row in observations.values() if row.get("api_id") == "API-04")
    assert missing["not_found"] and missing["http_status"] == 404
    assert any(e.get("source_id") == "API-04" and e.get("not_found") for e in result["evidence"])


async def test_mismatched_shipping_does_not_expand_lot_queries(modules, records):
    records["API-11"][0]["order_no"] = "666000001"
    result, observations, calls = await run(modules, records)
    assert result["status"] == "partial"
    assert not {"API-03", "API-04", "API-06"} & {call[0] for call in calls}
    assert "666000001" not in json.dumps([result, observations])


async def test_payment_uses_linked_references_once_and_masks_payer(modules, records):
    records["API-07"]["payment_refs"] *= 2
    result, _, calls = await run(modules, records, detail="payment")
    assert result["status"] == "ok" and [call[0] for call in calls].count("API-17") == 1
    assert "private-payer" not in json.dumps(result) and "9999" not in json.dumps(result)
    assert not {"API-01", "API-03", "API-04", "API-06"} & {call[0] for call in calls}


async def test_payment_for_another_order_is_not_exposed(modules, records):
    records["API-17"]["order_no"] = "666000001"
    result, observations, _ = await run(modules, records, detail="payment")
    assert result["status"] == "partial"
    payment = next(row for row in observations.values() if row.get("api_id") == "API-17")
    assert payment["error"] == "linked_target_mismatch" and "data" not in payment


async def test_linked_lot_limit_prevents_unbounded_followup_requests(modules, records):
    records["API-11"][0]["items"] = [{"sku": SKU, "qty": 1, "lot_id": f"LOT-{i}"} for i in range(3)]
    result, observations, calls = await run(modules, records)
    assert result["status"] == "partial"
    assert any(item.get("scope") == "lots" and item["total"] == 3 for item in result["data"]["partial"])
    assert len(calls) <= 11
    assert all([call[0] for call in calls].count(api_id) == 2 for api_id in ["API-03", "API-04", "API-06"])
    assert not any(row.get("parameters", {}).get("lot_id") == "LOT-2" for row in observations.values())


async def test_actual_shipment_sku_is_followed_within_same_product_limit(modules, records):
    records["API-11"][0]["items"][0]["sku"] = "SHIPPED-SKU"
    records["API-03"]["sku"] = "SHIPPED-SKU"
    async def product(request):
        return httpx.Response(200, json={**records["API-01"], "sku": request.url.path.rsplit("/", 1)[1]})
    result, observations, calls = await run(modules, records, overrides={"API-01": product})
    assert result["status"] == "ok" and [call[0] for call in calls].count("API-01") == 2
    assert {row["parameters"]["sku"] for row in observations.values() if row.get("api_id") == "API-01"} == {SKU, "SHIPPED-SKU"}


async def test_repeated_shipment_lot_is_fetched_once_and_truncation_is_explicit(modules, records):
    records["API-11"] *= 3
    result, observations, calls = await run(modules, records)
    assert result["status"] == "partial"
    assert all([call[0] for call in calls].count(api_id) == 1 for api_id in ["API-03", "API-04", "API-06"])
    shipping = next(row for row in observations.values() if row.get("api_id") == "API-11")
    assert shipping["status"] == "partial" and len(shipping["data"]) == 2


async def test_timeout_retains_safe_partial_order_and_does_not_retry(modules, records, monkeypatch):
    monkeypatch.setattr(modules[0], "GET_TIMEOUT", .01)
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=[])
    result, observations, calls = await run(modules, records, detail="shipping", overrides={"API-11": slow})
    assert result["status"] == "partial" and [call[0] for call in calls].count("API-11") == 1
    assert any(row.get("api_id") == "API-07" and row["status"] == "ok" for row in observations.values())


async def test_external_cancellation_is_not_converted_to_success(modules, records):
    async def cancel(request):
        raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await run(modules, records, overrides={"API-13": cancel})
