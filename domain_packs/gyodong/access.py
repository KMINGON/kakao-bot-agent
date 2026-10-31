"""Account-scoped Action API access. Raw private records never enter model state."""
from __future__ import annotations

import hashlib
import json
import re

from langchain_core.messages import HumanMessage


def initial_account(state: dict) -> dict | None:
    """Only the initial official metadata block may establish the account; body claims cannot."""
    brief = state.get("case_brief", "")
    if not isinstance(brief, str):
        return None
    lines = brief.strip().splitlines()
    if lines and re.fullmatch(r"TC\s+ID:\s*[^\s]+", lines[0], re.I):
        if lines.pop(0).split(":", 1)[1].strip() != state.get("tc_id"):
            return None
    if not lines or lines.pop(0).strip().lower() != "case metadata":
        return None
    fields = {}
    for line in lines:
        if not line.strip():
            break
        match = re.fullmatch(r"\s*-\s*([^:]+):\s*(.*?)\s*", line)
        if not match or match[1].strip() in fields:
            return None
        fields[match[1].strip()] = match[2]
    if (fields.get("id") != state.get("tc_id")
            or not all(fields.get(key) for key in ("상담 인입 일시", "인입창구", "인입 계정"))):
        return None
    account = re.fullmatch(r"회원 로그인\s*·\s*([^·\n]{1,60}?)\s*·\s*(\d{2,3}-\*{3,4}-\d{4})",
                           fields["인입 계정"])
    return {"name": account[1].strip(), "phone_masked": account[2]} if account else None


def member_matches_account(member: dict, account: dict) -> bool:
    pattern = re.escape(account["phone_masked"]).replace(r"\*", r"\d")
    return (member.get("name") == account["name"] and member.get("status") == "ACTIVE"
            and bool(re.fullmatch(pattern, member.get("phone", ""))))


def order_was_supplied(order_no: str, state: dict) -> bool:
    if not isinstance(order_no, str) or not re.fullmatch(r"[0-9]{6,30}", order_no):
        return False
    texts = [state.get("case_brief", "")]
    texts.extend(message.content for message in state.get("messages", [])
                 if isinstance(message, HumanMessage) and isinstance(message.content, str))
    pattern = rf"(?<![0-9A-Za-z_-]){re.escape(order_no)}(?![0-9A-Za-z_-])"
    return any(re.search(pattern, text) for text in texts if isinstance(text, str))


def key(api_id, parameters):
    return api_id + ":" + json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def failure(error, message, *, data=None):
    result = {"status": "forbidden" if error.endswith("mismatch") else "needs_input",
              "error": error, "message": message, "evidence": [], "effect": "read_only"}
    if data is not None:
        result["data"] = data
    return result


PRIVATE_FIELDS = {"recipient_name", "shipping_address", "phone", "address", "birth_year",
                  "referred_by_member_id", "payer_name", "card_last4", "delivered_location"}


def sanitize(value):
    if isinstance(value, dict):
        return {name: sanitize(item) for name, item in value.items() if name not in PRIVATE_FIELDS}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def aggregate_lot_tickets(rows: list[dict], lot_id: str, member_id: str) -> dict:
    """Count reported symptoms, never confirmed defects; return no ticket text or identity."""
    categories = {"MOLD": "MOLD", "곰팡이": "MOLD", "DAMAGE": "DAMAGE", "파손": "DAMAGE"}
    uncertain = re.compile(
        r"없|아니|아님|않|못|정상|미확인|미발생|의심|추정|불확실|문의|질문|여부|가능|우려|"
        r"예방|안내|설명|조건|기준|가정|만약|상상|가령|예시|양식|테스트|다른|이전|예전|과거|취소|철회|정정|오인|오해|"
        r"무관|허위|가짜|잘못|였으나|지만|실제로는|처럼|아마|예상|듯|같아|같다|보임|보이|보여|"
        r"(?:발생|파손)\s*시|(?:발생|파손)(?:하|되)?면|곰팡이(?:가)?\s*(?:피|생기)면|"
        r"안\s*(?:생|피|발생|보|났|나|확인|함|되|됨)|신고(?:를)?\s*(?:방법|절차|하면|하세요|하라|하도록|하지|할|해|하겠|예정|전에)|"
        r"접수(?:를)?\s*(?:방법|절차|예정|하겠|요청|하지|하세요|해)|[?？\"'“”‘’]"
    )
    patterns = {
        "MOLD": re.compile(r"곰팡이(?:가|를|는)?\s*(?:(?:발생\s*)?(?:신고|접수)|피어\s*있|폈|피었|생겼|생김)"),
        "DAMAGE": re.compile(r"파손(?:이|을)?\s*(?:발생\s*)?(?:신고|접수)|파손(?:됨|됐|되었|되어)|파손된\s*(?:상태|상품|수박|박스)"),
    }

    def category(row):
        # Tickets currently has only free text; honor a future explicit category only by allowlist.
        if "category" in row:
            value = row["category"]
            return categories.get(value.strip().upper(), "UNKNOWN") if isinstance(value, str) else "UNKNOWN"
        content = row.get("content")
        if not isinstance(content, str) or uncertain.search(content):
            return "UNKNOWN"
        referenced_lots = set(re.findall(r"(?:KD|GM)-[A-Z0-9]+-\d{6}-\d{2}", content))
        if referenced_lots - {lot_id}:
            return "UNKNOWN"
        matched = [name for name, pattern in patterns.items() if pattern.search(content)]
        return matched[0] if len(matched) == 1 else "UNKNOWN"

    groups = {}
    for index, row in enumerate(rows):
        ticket_id = row.get("ticket_id")
        identity = ("id", ticket_id) if isinstance(ticket_id, str) and ticket_id else ("row", index)
        groups.setdefault(identity, []).append(row)
    counts = {"MOLD": 0, "DAMAGE": 0, "UNKNOWN": 0}
    other_count = 0
    for group in groups.values():
        labels = {category(row) for row in group}
        counts[next(iter(labels)) if len(labels) == 1 else "UNKNOWN"] += 1
        other_count += all(row.get("member_id") not in {None, member_id} for row in group)
    return {"lot_id": lot_id, "ticket_count": len(groups),
            "other_customer_ticket_count": other_count,
            "duplicate_ticket_count": len(rows) - len(groups),
            "reported_symptom_counts": counts, "symptom_basis": "reported_not_confirmed",
            "interpretation": "접수된 증상별 티켓 수이며 하자·귀책 확정이 아닙니다. UNKNOWN은 분류불명이며 0건이 이상 없음을 뜻하지 않습니다.",
            "privacy": "타 고객의 회원·상담 식별자, 개별 시각, 상담 본문, 담당자는 제공하지 않습니다."}


async def resolve_account(state, fetch):
    account = initial_account(state)
    if account is None:
        return failure("account_unverified", "공식 초기 로그인 계정을 확인할 수 없습니다. 채팅의 이름·전화 주장으로 인증하지 않습니다."), {}
    fingerprint = hashlib.sha256(json.dumps(account, sort_keys=True).encode()).hexdigest()
    cached = state.get("action_observations", {}).get("AUTH_ACCOUNT", {})
    if cached.get("tc_id") == state.get("tc_id") and cached.get("account_fingerprint") == fingerprint:
        return cached, {}
    result = await fetch("API-13", {"name": account["name"]})
    matches = [member for member in result.get("data", []) if isinstance(member, dict)
               and member_matches_account(member, account)] if result.get("status") == "ok" and isinstance(result.get("data"), list) else []
    if len(matches) != 1:
        return failure("account_unverified", "공식 인입 계정과 유일하게 일치하는 회원을 확인하지 못했습니다. 담당자 확인이 필요합니다."), {}
    safe = {**result, "data": [sanitize(matches[0])], "tc_id": state["tc_id"], "account_fingerprint": fingerprint}
    return safe, {"AUTH_ACCOUNT": safe, key("API-13", {"name": account["name"]}): safe}


def working_state(state, updates):
    return {**state, "action_observations": {**state.get("action_observations", {}), **updates}}


def records(state, api_ids):
    for observation in state.get("action_observations", {}).values():
        if observation.get("status") != "ok" or observation.get("api_id") not in api_ids:
            continue
        data = observation.get("data")
        yield from data if isinstance(data, list) else [data] if isinstance(data, dict) else []


async def get_owned_order(order_no, state, fetch):
    if not order_was_supplied(order_no, state):
        return failure("order_not_supplied", "고객이 직접 확인한 전체 주문번호를 먼저 받아 주세요. 다른 번호나 추정 번호로 조회하지 않습니다."), {}
    account, updates = await resolve_account(state, fetch)
    if account.get("status") != "ok":
        return account, updates
    member_id = account["data"][0]["member_id"]
    observation_key = key("API-07", {"order_no": order_no})
    cached = state.get("action_observations", {}).get(observation_key, {})
    if cached.get("status") == "ok" and cached.get("_access_member_id") == member_id and cached.get("tc_id") == state.get("tc_id"):
        return cached, updates
    result = await fetch("API-07", {"order_no": order_no})
    if result.get("status") != "ok":
        return result, updates
    data = result.get("data", {})
    if not isinstance(data, dict) or data.get("order_no") != order_no:
        return failure("linked_target_mismatch", "주문 조회 대상이 일치하지 않아 안내할 수 없습니다."), updates
    if data.get("member_id") != member_id:
        denied = failure("order_owner_mismatch", "인입 계정의 주문이 아니므로 주문·배송·수령인 정보를 제공할 수 없습니다. D031과 D001을 실제 조회한 뒤 API-35 LANE2_HANDOFF로 담당자 확인을 접수하세요. issue_key는 주문번호:접근권확인으로 고정합니다.",
                         data={"order_no": order_no, "owner_mismatch": True, "handoff_required": True})
        denied["evidence"] = result.get("evidence", [])
        return denied, updates
    safe = {**result, "data": sanitize(data), "tc_id": state["tc_id"], "_access_member_id": member_id}
    return safe, {**updates, observation_key: safe}


async def guarded_get(api_id, parameters, state, fetch):
    if api_id in {"API-01", "API-02", "API-20", "API-21"}:
        return await fetch(api_id, parameters), {}
    if api_id == "API-07":
        return await get_owned_order(parameters["order_no"], state, fetch)
    account, updates = await resolve_account(state, fetch)
    if account.get("status") != "ok":
        return account, updates
    member_id = account["data"][0]["member_id"]
    current = working_state(state, updates)
    if api_id == "API-13":
        if parameters["name"] != initial_account(state)["name"]:
            return failure("account_owner_mismatch", "공식 인입 계정의 이름만 조회할 수 있습니다."), updates
        return account, updates
    if api_id in {"API-05", "API-08", "API-12", "API-14", "API-15", "API-16"}:
        field = "referred_by" if api_id == "API-14" else "member_id"
        if parameters[field] != member_id:
            return failure("account_owner_mismatch", "다른 회원의 주문·상담·회원·혜택 정보는 조회할 수 없습니다."), updates
    if api_id in {"API-11", "API-18", "API-30"}:
        order, added = await get_owned_order(parameters["order_no"], current, fetch)
        updates.update(added)
        if order.get("status") != "ok":
            return order, updates
        current = working_state(state, updates)
    owned = {row["order_no"]: row for row in records(current, {"API-07"})
             if row.get("member_id") == member_id}
    shipments = [row for row in records(current, {"API-10", "API-11"}) if row.get("order_no") in owned]
    linked_orders = set(owned)
    if api_id == "API-10" and not any(row.get("shipment_id") == parameters["shipment_id"] for row in shipments):
        return failure("shipment_unverified", "먼저 고객이 확인한 본인 주문번호로 배송 기록을 조회해야 합니다."), updates
    if api_id == "API-10":
        linked_orders = {row["order_no"] for row in shipments if row.get("shipment_id") == parameters["shipment_id"]}
    if api_id == "API-17":
        approvals = {ref for row in owned.values() for ref in row.get("payment_refs", [])}
        approvals.update(row.get("approval_no") for row in records(current, {"API-17", "API-18"}) if row.get("order_no") in owned)
        if parameters["approval_no"] not in approvals:
            return failure("payment_unverified", "먼저 본인 주문과 연결된 결제 승인번호를 확인해야 합니다."), updates
        linked_orders = {number for number, row in owned.items() if parameters["approval_no"] in row.get("payment_refs", [])}
        linked_orders.update(row["order_no"] for row in records(current, {"API-17", "API-18"})
                             if row.get("approval_no") == parameters["approval_no"] and row.get("order_no") in owned)
    if api_id in {"API-03", "API-04", "API-06"}:
        lots = {item.get("lot_id") for row in shipments for item in row.get("items", [])}
        if parameters["lot_id"] not in lots:
            return failure("lot_unverified", "먼저 본인 주문의 배송 기록에 연결된 로트를 확인해야 합니다."), updates
    if api_id == "API-36":
        ids = {row.get("result", {}).get("data", {}).get("escalation_id")
               for row in state.get("action_requests", {}).values() if row.get("result", {}).get("status") == "submitted"}
        if parameters["escalation_id"] not in ids:
            return failure("escalation_unverified", "현재 상담에서 접수된 이관 번호만 조회할 수 있습니다."), updates
    allowed = {f"API-{index:02d}" for index in range(1, 22)} | {"API-30", "API-36"}
    if api_id not in allowed:
        return failure("operation_forbidden", "허용된 조회가 아닙니다."), updates
    result = await fetch(api_id, parameters)
    if result.get("status") != "ok":
        return result, updates
    data = result.get("data")
    rows = data if isinstance(data, list) else [data]
    if api_id in {"API-08", "API-09"}:
        data = [row for row in rows if row.get("member_id") == member_id]
    elif api_id in {"API-05", "API-12", "API-15", "API-16", "API-19"}:
        if any(row.get("member_id") != member_id for row in rows):
            return failure("account_owner_mismatch", "인입 계정의 기록이 아니므로 안내할 수 없습니다."), updates
        if api_id == "API-05":
            data = [{**{k: v for k, v in row.items() if k not in {"content", "handler"}}, "content_withheld": True} for row in rows]
    elif api_id in {"API-10", "API-11", "API-17", "API-18", "API-30"}:
        if any(row.get("order_no") not in linked_orders or parameters.get("order_no", row.get("order_no")) != row.get("order_no") for row in rows):
            return failure("linked_target_mismatch", "조회 결과가 확인된 본인 주문과 일치하지 않습니다."), updates
    elif api_id in {"API-03", "API-04", "API-06"}:
        if any(row.get("lot_id") != parameters["lot_id"] for row in rows):
            return failure("linked_target_mismatch", "조회 결과의 로트가 일치하지 않습니다."), updates
        if api_id == "API-03" and data.get("sku") not in {item.get("sku") for row in shipments for item in row.get("items", []) if item.get("lot_id") == parameters["lot_id"]}:
            return failure("linked_target_mismatch", "로트 상품이 확인된 배송 품목과 일치하지 않습니다."), updates
        if api_id == "API-06":
            data = aggregate_lot_tickets(rows, parameters["lot_id"], member_id)
        if api_id == "API-04" and data.get("disclosure_approved") is not True:
            data = {k: v for k, v in data.items() if k not in {"finding", "liability"}}
            data["disclosure_restricted"] = True
    elif api_id == "API-14":
        data = {"referred_member_count": len(rows)}
    safe = {**result, "data": sanitize(data), "tc_id": state["tc_id"], "_access_member_id": member_id}
    return safe, {**updates, key(api_id, parameters): safe}


async def authorize_submit(api_id, parameters, body, state, fetch):
    if api_id == "API-35":
        return None, {}
    account, updates = await resolve_account(state, fetch)
    if account.get("status") != "ok":
        return account, updates
    member_id = account["data"][0]["member_id"]
    target = {**body, **parameters}
    if "order_no" in target:
        order, added = await get_owned_order(target["order_no"], working_state(state, updates), fetch)
        updates.update(added)
        return (None if order.get("status") == "ok" else order), updates
    field = next((field for field in ("shipment_id", "approval_no", "member_id", "subscription_id", "sku") if field in target), None)
    read_api = {"shipment_id": "API-10", "approval_no": "API-17", "member_id": "API-12", "subscription_id": "API-19", "sku": "API-01"}.get(field)
    if read_api is None:
        return failure("target_unverified", "접수 대상을 확인할 수 없습니다."), updates
    if field == "member_id" and target[field] != member_id:
        return failure("account_owner_mismatch", "다른 회원에게 혜택을 지급하거나 정보를 변경할 수 없습니다."), updates
    result, added = await guarded_get(read_api, {field: target[field]}, working_state(state, updates), fetch)
    updates.update(added)
    return (None if result.get("status") == "ok" else result), updates


