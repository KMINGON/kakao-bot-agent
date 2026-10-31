"""Bounded, read-only joins for an order belonging to the official incoming account."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from kakao_bot.runtime.policy import active_budget


MAX_LINKS = 2
MAX_ITEMS = 4
TOTAL_TIMEOUT = 14
GET_TIMEOUT = 3


def _key(api_id: str, parameters: dict) -> str:
    return api_id + ":" + json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def _failure(error: str, message: str) -> dict:
    return {"status": "needs_input", "error": error, "message": message,
            "effect": "read_only", "evidence": []}


async def inspect_order_context(api, access, order_no: str, detail: str, state: dict) -> tuple[dict, dict]:
    if detail not in {"shipping", "quality", "payment"}:
        return _failure("invalid_detail", "배송·품질·결제 중 필요한 조회 범위를 지정하세요."), {}
    budget = active_budget.get()
    deadline = time.monotonic() + min(TOTAL_TIMEOUT, budget.remaining() - 1 if budget else TOTAL_TIMEOUT)
    requests, observations, partial = {}, {}, []
    semaphore = asyncio.Semaphore(4)

    async def fetch(api_id, parameters):
        async def once():
            async with semaphore:
                left = deadline - time.monotonic()
                if left <= 0:
                    return {"api_id": api_id, "parameters": parameters, "status": "failed",
                            "error": "lookup_deadline", "effect": "read_only", "evidence": []}
                try:
                    async with asyncio.timeout(min(GET_TIMEOUT, left)):
                        return await api.request(api_id, tc_id=state["tc_id"], parameters=parameters,
                                                 method="GET", max_attempts=1)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    return {"api_id": api_id, "parameters": parameters, "status": "failed",
                            "error": type(error).__name__, "effect": "read_only", "evidence": []}
        key = _key(api_id, parameters)
        if key not in requests:
            requests[key] = asyncio.create_task(once())
        return await requests[key]

    def remember(response, data=None, *, complete=True):
        safe = {**response, "_access_member_id": member_id, "tc_id": state["tc_id"]}
        if data is not None:
            safe["data"] = data
        if not complete:
            safe["status"] = "partial"
            safe["truncated"] = True
        observations[_key(safe["api_id"], safe["parameters"])] = safe
        if safe.get("status") != "ok":
            partial.append({"api_id": safe["api_id"], "parameters": safe["parameters"],
                            "status": safe.get("status"), "error": safe.get("error"),
                            "not_found": safe.get("not_found", False)})

    def invalid(response):
        return {key: value for key, value in {**response, "status": "uncertain",
                "error": "linked_target_mismatch", "evidence": []}.items() if key != "data"}

    def finish():
        evidence = []
        for response in observations.values():
            for entry in response.get("evidence", []):
                if entry not in evidence:
                    evidence.append(entry)
        return {"status": "partial" if partial else "ok", "effect": "read_only",
                "data": {"order_no": order_no, "detail": detail, "ownership": "verified",
                         "observations": [{key: item for key, item in value.items()
                                           if not key.startswith("_") and key != "account_fingerprint"}
                                          for name, value in observations.items() if name != "AUTH_ACCOUNT"],
                         "partial": partial,
                         "api_calls": len(requests)}, "evidence": evidence,
                "message": "조회만 수행했습니다. 빈 목록·404·조회 실패·부분 확인을 구분하고 고객 선택이나 접수 완료로 해석하지 마세요."}, observations

    order, authorized = await access.get_owned_order(order_no, state, fetch)
    observations.update(authorized)
    if order.get("status") != "ok":
        return {**order, "effect": "read_only"}, observations
    member_id = order["data"]["member_id"]
    order_data = {key: value for key, value in order["data"].items()
                  if key not in {"recipient_name", "shipping_address"}}
    items = order_data.get("items", [])
    order_data["items"] = items[:MAX_ITEMS]
    remember(order, order_data, complete=len(items) <= MAX_ITEMS)

    def limited(values, label):
        unique = list(dict.fromkeys(value for value in values if isinstance(value, str) and value))
        if len(unique) > MAX_LINKS:
            partial.append({"scope": label, "total": len(unique), "checked": MAX_LINKS})
        return unique[:MAX_LINKS]

    skus = limited([item["sku"] for item in items], "products") if detail == "quality" else []
    payments = limited(order_data.get("payment_refs", []), "payments") if detail == "payment" else []
    batch = [("API-11", {"order_no": order_no})]
    batch += [("API-01", {"sku": sku}) for sku in skus]
    batch += [("API-17", {"approval_no": approval}) for approval in payments]
    results = await asyncio.gather(*(fetch(api_id, parameters) for api_id, parameters in batch))
    lot_skus = {}
    for response in results:
        if response.get("status") != "ok":
            remember(response)
            continue
        data = response.get("data")
        api_id = response["api_id"]
        if api_id == "API-11":
            if not isinstance(data, list) or any(row.get("order_no") != order_no for row in data):
                remember(invalid(response))
                continue
            selected = data[:MAX_LINKS]
            shipping = []
            for shipment in selected:
                safe = {key: value for key, value in shipment.items() if key != "delivered_location"}
                safe["items"] = shipment["items"][:MAX_ITEMS]
                shipping.append(safe)
                for item in safe["items"]:
                    lot_skus.setdefault(item["lot_id"], set()).add(item["sku"])
            complete = len(data) <= MAX_LINKS and all(len(row["items"]) <= MAX_ITEMS for row in selected)
            remember(response, shipping, complete=complete)
        elif api_id == "API-01":
            remember(response if isinstance(data, dict) and data.get("sku") == response["parameters"]["sku"]
                     else invalid(response))
        elif api_id == "API-17":
            if not isinstance(data, dict) or data.get("order_no") != order_no:
                remember(invalid(response))
            else:
                remember(response, {key: value for key, value in data.items() if key not in {"payer_name", "card_last4"}})
    if detail == "quality":
        connected_skus = limited(skus + [sku for group in lot_skus.values() for sku in sorted(group)], "products")
        lots = limited(lot_skus, "lots")
        linked = [("API-01", {"sku": sku}) for sku in connected_skus if sku not in skus]
        linked += [(api_id, {"lot_id": lot}) for lot in lots for api_id in ("API-03", "API-04", "API-06")]
        for response in await asyncio.gather(*(fetch(api_id, parameters) for api_id, parameters in linked)):
            if response.get("status") != "ok":
                remember(response)
                continue
            if response["api_id"] == "API-01":
                remember(response if response.get("data", {}).get("sku") == response["parameters"]["sku"]
                         else invalid(response))
                continue
            data, lot, api_id = response.get("data"), response["parameters"]["lot_id"], response["api_id"]
            if api_id == "API-06":
                if not isinstance(data, list) or any(row.get("lot_id") != lot for row in data):
                    remember(invalid(response))
                else:
                    remember(response, access.aggregate_lot_tickets(data, lot, member_id))
            elif not isinstance(data, dict) or data.get("lot_id") != lot:
                remember(invalid(response))
            elif api_id == "API-03":
                if data.get("sku") in lot_skus[lot]:
                    remember(response, {key: value for key, value in data.items() if key != "sprayed_at"})
                else:
                    remember(invalid(response))
            else:
                safe = dict(data)
                if safe.get("disclosure_approved") is not True:
                    safe.pop("finding", None)
                    safe.pop("liability", None)
                    safe["disclosure_restricted"] = True
                remember(response, safe)
    return finish()


def build_order_tool(api, access):
    @tool
    async def inspect_order(order_no: str, detail: Literal["shipping", "quality", "payment"],
                            runtime: ToolRuntime) -> Command:
        """고객이 직접 준 주문번호를 공식 로그인 계정과 대조한 뒤 연결 기록을 조회한다. shipping은 주문·배송, quality는 품질·신선도·로트·조사·신고 건수, payment는 결제 승인 기록까지 읽는다. 실행·접수하지 않는다. 다른 계정 주문이나 추정 번호는 조회·공개하지 않으며 부분 결과를 전체 부재로 해석하지 않는다."""
        result, observations = await inspect_order_context(api, access, order_no, detail, runtime.state)
        return Command(update={"action_observations": observations, "messages": [ToolMessage(
            content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]})
    return inspect_order
