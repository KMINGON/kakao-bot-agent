import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import ToolMessage

from kakao_bot.runtime import load_pack, lock_pack
from kakao_bot.runtime.action_api import ActionAPI
from kakao_bot.runtime.agent import checkpoint_evidence, validate_answer


SPEC = json.loads(Path("domain_packs/gyodong/data/action-openapi.json").read_text())


def api(client=None):
    return ActionAPI(SPEC, "test-token", client=client)


def sample(schema):
    if "$ref" in schema:
        return sample(SPEC["components"]["schemas"][schema["$ref"].split("/")[-1]])
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next(t for t in kind if t != "null")
    if kind == "object":
        return {k: sample(schema.get("properties", {})[k]) for k in schema.get("required", [])}
    if kind == "array":
        return [sample(schema["items"])]
    if kind == "integer":
        return 100
    if kind == "boolean":
        return True
    if schema.get("format") == "date":
        return "2037-06-30"
    if schema.get("format") == "date-time":
        return "2037-06-30T12:00:00+09:00"
    return "confirmed-target"


def test_all_operations_are_schema_bound_and_reset_is_excluded():
    client = api()
    assert len(client.operations) == 40
    assert sum(op["method"] == "POST" for op in client.operations.values()) == 17
    for api_id, op in client.operations.items():
        params = {p["name"]: "target" for p in op["parameters"] if p["in"] in {"path", "query"}}
        schema = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
        body = sample(schema) if schema else None
        client.prepare(api_id, params, body, op["method"])
        with pytest.raises(ValueError):
            client.prepare(api_id, {**params, "unsupported": "x"}, body, op["method"])
        if schema:
            with pytest.raises(ValueError):
                client.prepare(api_id, params, {**body, "unsupported": "x"}, "POST")
            for required in schema.get("required", []):
                incomplete = dict(body); incomplete.pop(required)
                with pytest.raises(ValueError):
                    client.prepare(api_id, params, incomplete, "POST")
    with pytest.raises(ValueError):
        client.prepare("resetMyActionData", {}, {}, "POST")


@pytest.mark.parametrize("body", [
    {"order_no": "100", "amount": True, "reason": "하자"},
    {"order_no": "100", "amount": 1.2, "reason": "하자"},
    {"order_no": "100", "amount": 2**40, "reason": "하자"},
])
def test_receipt_rejects_invalid_money_types(body):
    with pytest.raises(ValueError):
        api().prepare("API-29", {}, body, "POST")


@pytest.mark.asyncio
async def test_query_once_and_concurrent_tc_headers_are_isolated():
    requests = []
    async def handle(request):
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json=[])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = api(http)
        result = await asyncio.gather(*[client.request("API-08", tc_id=tc,
            parameters={"member_id": member}) for tc, member in [("L-03", "MB-1"), ("Y-16", "MB-2")]])
    assert all(r["status"] == "ok" for r in result)
    assert {(r.headers["X-Aitop100-Test-Case-Id"], r.url.params["member_id"]) for r in requests} == {
        ("L-03", "MB-1"), ("Y-16", "MB-2")}
    assert all(len(r.url.params.multi_items()) == 1 for r in requests)
    assert "X-Aitop100-Test-Case-Id" not in http.headers


@pytest.mark.asyncio
async def test_receipt_retry_preserves_exact_body_and_is_only_submitted():
    requests = []
    async def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"refund_id": "RF-1", "order_no": "100", "request_status": "SUBMITTED",
                                        "requested_at": "2037-06-30T12:00:00+09:00"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        result = await api(http).request("API-29", tc_id="L-11", parameters={},
            body={"order_no": "100", "amount": 32000, "reason": "고객이 확인한 곰팡이 하자"}, method="POST")
    assert len(requests) == 2 and requests[0].content == requests[1].content
    assert result["status"] == "submitted"
    assert result["data"]["request_status"] == "SUBMITTED"
    assert result["evidence"][0]["source_id"] == "API-29"


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [(400,None),(404,None),(429,{"code":"RATE_LIMIT_EXCEEDED","message":"누적 API 호출 한도를 초과했습니다."})])
async def test_permanent_errors_are_not_retried(status, body):
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(status,json=body) if body else httpx.Response(status)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        result=await api(http).request("API-07",tc_id="L-03",parameters={"order_no":"100"})
    assert len(calls)==1 and result["status"]=="failed"
    assert result["http_status"]==status


@pytest.mark.asyncio
async def test_invalid_receipt_is_uncertain_not_a_success():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200,json={"ok":True}))) as http:
        result=await api(http).request("API-22",tc_id="L-03",parameters={"order_no":"100"},body={},method="POST")
    assert result["status"]=="uncertain" and result["evidence"]==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("api_id", [key for key,op in api().operations.items() if op["method"]=="POST"])
async def test_all_seventeen_receipt_shapes_match_contract(api_id):
    operation=api().operations[api_id]
    request_schema=operation["requestBody"]["content"]["application/json"]["schema"]
    response_schema=operation["responses"]["200"]["content"]["application/json"]["schema"]
    body=sample(request_schema)
    params={p["name"]:"confirmed-target" for p in operation["parameters"] if p["in"]=="path"}
    response=sample(response_schema)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=response))) as http:
        result=await api(http).request(api_id,tc_id="L-03",parameters=params,body=body,method="POST")
    assert result["status"]=="submitted"
    assert result["data"]["request_status"]=="SUBMITTED"
    assert result["evidence"][0]["source_id"]==api_id


@pytest.mark.asyncio
async def test_receipt_for_different_order_is_not_accepted():
    receipt={"refund_id":"RF-wrong","order_no":"different-order","request_status":"SUBMITTED",
             "requested_at":"2037-06-30T12:00:00+09:00"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=receipt))) as http:
        result=await api(http).request("API-29",tc_id="L-11",parameters={},
            body={"order_no":"requested-order","amount":100,"reason":"하자"},method="POST")
    assert result["status"]=="uncertain" and result["evidence"]==[]


@pytest.mark.asyncio
async def test_single_attempt_smoke_never_retries_transient_failure():
    requests=[]
    def fail(request):
        requests.append(request)
        return httpx.Response(503)
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
        result=await api(http).request("API-01",tc_id="L-47",parameters={"sku":"WTM-STD-8"},max_attempts=1)
    assert len(requests)==1 and result["status"]=="failed"


@pytest.mark.asyncio
async def test_get_404_is_citable_negative_evidence_but_not_a_target(tmp_path, monkeypatch):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(404))) as http:
        result = await api(http).request("API-20", tc_id="L-20", parameters={"code": "MELON99"})
    assert result["status"] == "failed" and result["not_found"] is True
    assert result["parameters"] == {"code": "MELON99"} and "data" not in result
    assert result["evidence"][0] == {
        "source_id": "API-20", "locator": "/api/promos/MELON99", "http_status": 404,
        "not_found": True, "retrieved_at": result["evidence"][0]["retrieved_at"]}

    shutil.copytree("domain_packs/gyodong", tmp_path / "gyodong", ignore=shutil.ignore_patterns("__pycache__"))
    lock_pack(tmp_path / "gyodong")
    pack = load_pack(tmp_path, "gyodong")
    async def observed(*args, **kwargs):
        return result
    monkeypatch.setattr(ActionAPI, "request", observed)
    get = next(tool for tool in pack.python_tools() if tool.name == "action_get")
    command = await get.coroutine(api_id="API-20", parameters={"code": "MELON99"},
        runtime=SimpleNamespace(state={"tc_id": "L-20"}, tool_call_id="missing-code"))
    assert command.update["action_observations"] == {}
    seen, _ = checkpoint_evidence(command.update["messages"], pack, "event")
    answer = {"status": "answered", "answer": "조회한 MELON99 코드는 없습니다.",
              "claims": [{"text": "조회한 MELON99 코드가 없습니다.", "evidence_ids": ["API-20"]}],
              "evidence_ids": ["API-20"], "missing_fields": []}
    assert validate_answer(answer, pack, seen).evidence_ids == ["API-20"]
    with pytest.raises(ValueError, match="not retrieved"):
        validate_answer({**answer, "evidence_ids": ["API-07"]}, pack, seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 503, None])
async def test_other_get_failures_are_not_negative_evidence(status):
    def fail(request):
        if status is None:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(status)
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
        result = await api(http).request("API-20", tc_id="L-20", parameters={"code": "MELON99"}, max_attempts=1)
    assert result["status"] == "failed" and result["evidence"] == []
    assert result.get("not_found") is not True
    seen, _ = checkpoint_evidence([ToolMessage(content=json.dumps(result), tool_call_id="failure")],
                                 SimpleNamespace(evidence_ids={"API-20"}), "event")
    assert seen == set()


@pytest.mark.asyncio
async def test_post_404_does_not_create_negative_query_evidence():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(404))) as http:
        result = await api(http).request("API-22", tc_id="L-03", parameters={"order_no": "100"},
                                        body={}, method="POST")
    assert result["status"] == "failed" and result["evidence"] == []
    assert result.get("not_found") is not True
