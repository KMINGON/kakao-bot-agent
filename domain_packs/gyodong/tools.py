"""Reviewed contest tools. Credentials and TC headers are never model arguments."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from kakao_bot.runtime.action_api import ActionAPI
from kakao_bot.runtime.competition import submission_identity, submission_record
from kakao_bot.runtime.tools import calculate
from kakao_bot.settings import secret


ROOT = Path(__file__).parent


def _message(runtime, result: dict, **updates):
    return Command(update={**updates, "messages": [ToolMessage(
        content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]})


def _failure(message: str, *, status: str = "needs_input") -> dict:
    return {"status": status, "message": message, "evidence": []}


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _target_record(state: dict, field: str, value: str) -> dict | None:
    sources = {"order_no": {"API-07"}, "shipment_id": {"API-10", "API-11"},
               "approval_no": {"API-17", "API-18"}, "member_id": {"API-12", "API-13"},
               "subscription_id": {"API-19"}, "sku": {"API-01", "API-02"}}
    for observation in state.get("action_observations", {}).values():
        if observation.get("status") == "ok" and observation.get("api_id") in sources.get(field, set()):
            data = observation.get("data")
            for obj in data if isinstance(data, list) else [data] if isinstance(data, dict) else []:
                if obj.get(field) == value:
                    return obj
    return None


def _latest_customer_text(state):
    return next((message.content for message in reversed(state.get("messages", []))
                 if isinstance(message, HumanMessage) and isinstance(message.content, str)), "")


def _receipt_guidance(api_id, body, result, state, sources=None):
    if api_id == "API-35" and result.get("status") == "submitted":
        result = {**result, "next_step": "이관은 접수 상태입니다. D001에서 회신 기한을 확인하고 실제 이관번호와 함께 고객에게 안내하세요. 실제 회신 날짜를 계산할 때는 D022의 영업일·공휴일 규칙을 별도로 확인하며 문서의 예시 날짜를 사용하지 마세요. 봇이 이 대화 밖에서 자동으로 결과를 확인해 다시 연락하겠다고 약속하지 마세요."}
        source = (sources or {}).get("D001", {})
        lines = source.get("text", "").splitlines()
        start = next((i for i, line in enumerate(lines) if line == "## 접수·회신·승인"), None)
        if start is not None:
            end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            result["followup_policy"] = {"source_id": "D001", "title": source.get("title", ""),
                "status": source.get("status", "문서확인"), "scope": source.get("scope", ""),
                "start_line": start + 1, "end_line": end, "text": "\n".join(lines[start:end])}
            result["evidence"] = [*result.get("evidence", [])]
            if not any(item.get("source_id") == "D001" for item in result["evidence"]):
                result["evidence"].append({"source_id": "D001", "start_line": start + 1, "end_line": end})
        return result
    if api_id != "API-29" or result.get("status") != "submitted":
        return result
    lots = {item.get("lot_id") for row in _objects(state.get("action_observations", {}))
            if row.get("order_no") == body.get("order_no") and "shipment_id" in row
            for item in row.get("items", [])}
    reports = [row.get("data", {}) for row in state.get("action_observations", {}).values()
               if row.get("api_id") == "API-06" and row.get("parameters", {}).get("lot_id") in lots
               and row.get("status") == "ok"]
    if any(isinstance(row, dict)
           and row.get("other_customer_reported_symptom_counts", {}).get("MOLD", 0) > 0 for row in reports):
        return {**result, "next_step": "환불 접수와 별개로 같은 배송 로트의 타 고객 곰팡이 신고가 확인돼 있습니다. D001 안전 이관 규정을 읽고 조사 확인 상태와 함께 API-35 LANE2_HANDOFF로 안전·판매자/공급 확인을 접수하세요. 이미 같은 사안을 접수했다면 그 번호를 재사용하고, 환불 번호와 안전 이관 번호·회신 기한을 함께 안내하세요. 신고는 하자·원인 확정이 아닙니다."}
    return result


def _requested_actions(text):
    # Only explicit current instructions authorize a new mutation; ambiguous replies need clarification.
    terms = {
        "API-22": r"복구|되살",
        "API-23": r"(?:상품|품목|수량|개수).{0,12}(?:변경|바꾸|바꿔|수정)",
        "API-24": r"주소|배송지",
        "API-25": r"날짜|일정|출고일|배송일|수령일",
        "API-27": r"긴급|빨리|서둘러",
        "API-28": r"재배송|재발송|다시.{0,4}보내|다음.{0,6}(?:배|운송편)",
        "API-29": r"환불",
        "API-31": r"(?:카드|결제|승인).{0,12}취소",
        "API-32": r"영수증|소득공제|지출증빙|발행",
        "API-33": r"쿠폰|차액\s*보상|가격.{0,5}보상",
        "API-34": r"적립금|포인트|리워드",
        "API-37": r"배송비|재배송비",
        "API-38": r"픽업|수거|회수",
        "API-39": r"재입고|입고.{0,5}알림",
        "API-40": r"구독|정기배송",
    }
    selected = set()
    # A separate question does not revoke an instruction. Quoted examples are not consent.
    unquoted = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'|“[^”\n]*”|‘[^’\n]*’', "", text)
    for clause in re.findall(r"[^.!?;,\n]+[.!?;,]?", unquoted):
        if re.search(r"\?|나요|까요|어때요|가능|할\s*수|수\s*있|방법|됐|되었", clause):
            continue
        choice = re.sub(r"(?:환불|재배송|재발송|취소)(?:은|는|을|를)?\s*(?:대신|말고|안\s*해|하지\s*(?:마|말)|취소|철회)", "", clause)
        if (re.search(r"해\s*(?:주|줘)|부탁|요청|원해|원합니다|선택|할게|할래|낼게|하겠|보내\s*(?:주|줘)|받을게|진행|접수|신청|발행|발급|변경", choice)
                or re.fullmatch(r"(?:환불|재배송|재발송)(?:로|이)?(?:요)?[.!;, ]*", choice.strip())):
            selected.update(operation for operation, pattern in terms.items() if re.search(pattern, choice))
    for operation, pattern in terms.items():
        negative = r"지\s*(?:마|말)|않|안\s*(?:해|하|받|보내|할)|싫|보류|철회"
        if operation not in {"API-31", "API-40"}:
            negative += "|취소"
        if re.search(r"(?:" + pattern + r")[^.!?\n]*(?:" + negative + ")", text):
            selected.discard(operation)
    return selected


def _current_choice_guard(api_id, parameters, body, state):
    if api_id in {"API-26", "API-35"}:
        return None
    text = _latest_customer_text(state)
    selected = _requested_actions(text)
    last_clause = next((part.strip() for part in reversed(re.split(r"[.!?\n]+", text)) if part.strip()), "")
    withdrawal = r"지\s*(?:마|말)|취소|철회|보류|그만|안\s*(?:해|하|받|보내|할)|(?:진행|접수|신청|요청|원하|원치|받|보내).{0,6}않|(?:^|\s)하지\s*않"
    if re.search(withdrawal, last_clause) and not _requested_actions(last_clause):
        return "최신 고객이 조치를 철회·보류했습니다. 과거 요청으로 접수하지 마세요."
    if api_id in {"API-28", "API-29"} and {"API-28", "API-29"} <= selected:
        return "환불과 재배송 중 고객의 최종 선택을 확인하세요. 두 조치를 함께 접수하지 않습니다."
    if api_id in selected:
        if api_id == "API-40" and not re.search(r"취소|해지" if body.get("kind") == "CANCEL" else r"건너|쉬|스킵|일시정지", text):
            return "구독 취소와 건너뛰기 중 고객이 선택한 조치를 확인하세요."
        return None
    if selected or re.search(r"지\s*(?:마|말)|취소|철회|보류|그만|않|안\s*(?:해|하|받|보내|할)", text):
        return "최신 고객의 선택·철회와 이 접수 조치가 일치하지 않습니다. 과거 요청을 대신 적용하지 마세요."
    canonical = json.dumps({"parameters": parameters, "body": body}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if any(previous.get("api_id") == api_id and previous.get("canonical") == canonical
           and previous.get("result", {}).get("status") in {"submitted", "unknown", "uncertain"}
           for previous in state.get("action_requests", {}).values()):
        return None
    return "최신 고객 발화에서 이 조치를 요청·선택했는지 명확히 확인해야 합니다. 가능 여부 질문이나 번호 확인만으로 접수하지 마세요."


def _shipping_guidance(result, state):
    text = _latest_customer_text(state)
    verified_partial = (result.get("status") == "partial" and isinstance(result.get("data"), dict)
                        and result["data"].get("ownership") == "verified")
    if ((result.get("status") != "ok" and not verified_partial) or "API-28" not in _requested_actions(text)
            or not re.search(r"다음\s*(?:배|운송편)", text)):
        return result
    rows = [result] if result.get("api_id") else result.get("data", {}).get("observations", [])
    shipments = [shipment for row in rows if row.get("api_id") in {"API-10", "API-11"}
                 and row.get("status") == "ok"
                 for shipment in (row.get("data", []) if isinstance(row.get("data"), list) else [row.get("data")])
                 if isinstance(shipment, dict) and shipment.get("status") == "READY"
                 and "picked_up_at" in shipment and shipment["picked_up_at"] is None
                 and isinstance(shipment.get("shipment_id"), str) and shipment["shipment_id"]]
    if len(shipments) != 1:
        return result
    parameters = {"shipment_id": shipments[0]["shipment_id"]}
    if _current_choice_guard("API-28", parameters, {}, state):
        return result
    return {**result, "next_step": "고객이 다음 배·운송편 재출고를 명시적으로 요청했고 이 배송은 READY·집하 전입니다. READY라는 이유만으로 재출고 접수 불가라고 판단하지 마세요. D017·D021·D008 원문을 실제 조회해 적용 사유·조건과 고객 선택을 확인한 뒤 API-28에 parameters="
            + json.dumps(parameters, ensure_ascii=False) + ", body={}로 접수할 수 있습니다. 희망 날짜는 필수 입력이 아닙니다. READY는 집하 전이라는 상태만 뜻하며 포장 경과 시간·신선도·변질 여부를 증명하지 않습니다. 이 조회는 접수가 아니며 결항 원인·운송 재개일·도착일을 추정하지 마세요."}


def submission_guard(api_id, parameters, body, state, policy_source_ids,
                     customer_confirmation, policy_reason):
    if not state.get("tc_id"):
        return "현재 상담의 TC ID를 먼저 확인하세요."
    if not policy_source_ids or not all(source in state.get("policy_reads", {}) for source in policy_source_ids):
        return "적용 정책 원문을 먼저 조회하고 조회한 정책 출처 ID를 지정하세요."
    if len(policy_reason.strip()) < 10:
        return "대상·적용 조건·예외를 확인한 정책 판단 근거를 적으세요."
    choice_error = _current_choice_guard(api_id, parameters, body, state)
    if choice_error:
        return choice_error
    targets = {**parameters, **body}
    field = next((key for key in ("order_no", "shipment_id", "approval_no", "member_id", "subscription_id", "sku")
                  if key in targets), None)
    record = _target_record(state, field, targets[field]) if field else None
    if api_id != "API-35" and record is None:
        return "처리할 정확한 대상을 먼저 조회하세요. 다른 주문·배송·회원으로 바꾸지 마세요."
    if api_id in {"API-28", "API-29"}:
        order_no = body["order_no"] if api_id == "API-29" else record.get("order_no")
        opposite = "API-28" if api_id == "API-29" else "API-29"
        for previous in state.get("action_requests", {}).values():
            if previous.get("api_id") != opposite or previous.get("result", {}).get("status") not in {"submitted", "unknown", "uncertain"}:
                continue
            try:
                request = json.loads(previous["canonical"])
                if opposite == "API-29":
                    previous_order = request["body"]["order_no"]
                else:
                    shipment = _target_record(state, "shipment_id", request["parameters"]["shipment_id"])
                    previous_order = shipment.get("order_no") if shipment else None
            except (KeyError, TypeError, ValueError):
                previous_order = None
            if previous_order is None or previous_order == order_no:
                return "이미 환불 또는 재배송 요청이 접수됐거나 결과가 미확인입니다. 반대 조치를 함께 접수할 수 없습니다. 고객이 선택을 바꿔도 기존 요청을 철회하는 API가 없으므로 기존 접수번호와 변경 의사를 운영 담당에게 이관하세요."
    if api_id == "API-24":
        shipping = next((observation for observation in state.get("action_observations", {}).values()
            if observation.get("api_id") == "API-11" and observation.get("status") == "ok"
            and observation.get("parameters", {}).get("order_no") == parameters["order_no"]), None)
        if shipping is None or not isinstance(shipping.get("data"), list):
            return "배송지 변경 전에 같은 주문번호로 API-11 배송 조회를 하세요. 결제 완료만으로 집하 전이라고 판단할 수 없습니다."
        if any(shipment.get("order_no") != parameters["order_no"] or shipment.get("picked_up_at")
               or shipment.get("status") != "READY" for shipment in shipping["data"]):
            return "배송 기록에서 집하 전임을 확인할 수 없습니다. 집하 스캔과 배송 상태를 확인하고 운영 담당에게 이관하세요."
    if api_id == "API-29" and (body["amount"] <= 0 or body["amount"] > record.get("amount", 0)):
        return "환불액은 조회한 주문의 실제 결제액 범위에서 정책·산식으로 확인해야 합니다."
    if any(key in body and body[key] <= 0 for key in ("qty", "charge_amount")):
        return "수량·청구액은 0보다 커야 합니다."
    if api_id == "API-35" and body["lane"] not in {"LANE2_HANDOFF", "LANE3_CLAIM"}:
        return "에스컬레이션은 확인한 LANE2_HANDOFF 또는 LANE3_CLAIM 경로를 사용하세요."
    return None


def build_tools(data):
    access_spec = importlib.util.spec_from_file_location(f"{__name__}_access", ROOT / "access.py")
    access = importlib.util.module_from_spec(access_spec)
    sys.modules[access_spec.name] = access
    exec(compile((ROOT / "access.py").read_bytes(), str(ROOT / "access.py"), "exec"), access.__dict__)
    api = ActionAPI(json.loads((ROOT / "data/action-openapi.json").read_text()), secret("ACTION_API_TOKEN"))
    policy_path = ROOT / "data/policies.json"
    sources = {source["id"]: source for source in json.loads(policy_path.read_text())["sources"]}

    @tool
    def case_context(runtime: ToolRuntime) -> dict:
        """현재 상담의 TC ID와 최초 고객/상담 정보 원문을 확인한다. 서버 날짜를 상담 날짜로 쓰지 않는다."""
        return {"status": "ok" if runtime.state.get("tc_id") else "needs_input",
                "data": {"tc_id": runtime.state.get("tc_id"), "initial_message": runtime.state.get("case_brief", ""),
                    "observed_targets": [{"api_id": r.get("api_id"), "parameters": r.get("parameters", {})}
                        for r in runtime.state.get("action_observations", {}).values()],
                    "receipts": [{"request": json.loads(r["canonical"]), "result": r["result"]}
                        for r in runtime.state.get("action_requests", {}).values()]},
                "evidence": [{"source_id": "CASE_CONTEXT"}] if runtime.state.get("tc_id") else [],
                "effect": "read_only"}

    @tool
    def search_policy(query: str, runtime: ToolRuntime) -> Command:
        """정책·문서의 적용 범위와 관련 구간을 검색한다. 짧은 핵심어로 찾고 read_policy로 앞뒤 예외를 확인한다."""
        from math import log1p

        words = list(dict.fromkeys(w.lower() for w in re.split(r"\s+", query.strip()) if w))
        if not words or len(query) > 200:
            return _message(runtime, _failure("검색할 정책 핵심어를 짧게 지정하세요."))
        documents = [(source.get("title", "") + " " + source["text"]).lower() for source in sources.values()]
        frequencies = {word: sum(word in document for document in documents) for word in words}
        weights = {word: log1p((len(documents) - frequency + .5) / (frequency + .5))
                   for word, frequency in frequencies.items()}
        found = []
        for source in sources.values():
            lines = source["text"].splitlines()
            title, best = source.get("title", "").lower(), None
            for start in range(0, len(lines), 12):
                text = "\n".join(lines[start:start + 16])
                # Repeated product names must not crowd out the rarer policy terms.
                score = sum(weights[word] * (min(text.lower().count(word), 2) + 2 * (word in title))
                            for word in words)
                rank = (score, sum(word in text.lower() or word in title for word in words))
                if score and (best is None or rank > best[0]):
                    scope = source.get("scope", "")
                    best = (rank, {"source_id": source["id"], "title": source.get("title", ""),
                        "scope": scope[:600], "scope_truncated": len(scope) > 600,
                        "start_line": start + 1, "text": text[:1400], "status": source.get("status", "문서확인")})
            if best:
                found.append(best)
        hits = [entry for _, entry in sorted(found, key=lambda item: item[0], reverse=True)[:3]]
        ids = {hit["source_id"]: True for hit in hits}
        return _message(runtime, {"status": "ok", "data": hits,
            "evidence": [{"source_id": source_id} for source_id in ids], "effect": "read_only"}, policy_reads=ids)

    @tool
    def read_policy(source_id: str, runtime: ToolRuntime, start_line: int = 1) -> Command:
        """정책 원문을 출처 ID(D001 등)와 1부터 시작하는 줄 번호로 읽는다. next_line이 있으면 필요한 구간을 이어 읽는다."""
        source = sources.get(source_id)
        if source is None or start_line < 1:
            return _message(runtime, _failure("등록된 정책 출처 ID와 양의 줄 번호를 지정하세요."))
        lines = source["text"].splitlines()
        if start_line > len(lines):
            return _message(runtime, _failure(f"문서는 {len(lines)}줄입니다. 범위 안의 줄 번호를 지정하세요."))
        chunk, count = [], 0
        for line in lines[start_line - 1:start_line + 39]:
            if chunk and count + len(line) > 3000:
                break
            chunk.append(line[:3000]); count += len(line) + 1
        end = start_line - 1 + len(chunk)
        return _message(runtime, {"status": "ok", "data": {"source_id": source_id,
            "title": source.get("title", ""), "status": source.get("status", "문서확인"),
            "scope": source.get("scope", ""), "start_line": start_line,
            "text": "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(chunk)),
            "next_line": end + 1 if end < len(lines) else None},
            "evidence": [{"source_id": source_id}], "effect": "read_only"}, policy_reads={source_id: True})

    @tool(description="업무 데이터를 조회한다. 공개 상품·재고(API-01/02/21)와 프로모션(API-20)은 고객의 최종 선택·주문번호가 없어도 공식 자료의 관련 SKU·코드 후보로 조건을 조회할 수 있다. 후보 조회는 해당 고객의 적용 자격·사용·할인 적용 확정이 아니다. 개별 주문·회원·배송·결제는 인입 계정의 권한과 고객이 확인한 대상을 따른다. TC는 대화 상태에서 자동으로 넣는다. parameters에는 아래 이름만 사용한다.\n" + api.describe("GET"))
    async def action_get(api_id: str, parameters: dict, runtime: ToolRuntime) -> Command:
        if not runtime.state.get("tc_id"):
            return _message(runtime, _failure("현재 상담의 TC ID를 알려 주세요. 임의 ID로 호출하지 않습니다."))
        try:
            api.prepare(api_id, parameters, None, "GET")
            async def fetch(operation, targets):
                return await api.request(operation, tc_id=runtime.state["tc_id"], parameters=targets)
            result, observations = await access.guarded_get(api_id, parameters, runtime.state, fetch)
        except ValueError as error:
            return _message(runtime, _failure(str(error), status="failed"))
        key = api_id + ":" + json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        if result["status"] == "ok":
            observations[key] = result
        return _message(runtime, _shipping_guidance(result, runtime.state), action_observations=observations)

    @tool(description="정책·정확한 대상 조회·고객 의사를 확인한 뒤 업무 요청을 접수한다. 정책은 API가 검사하지 않는다. 성공도 SUBMITTED 접수이며 실제 지급·상태 변경 완료가 아니다. customer_confirmation은 서버가 최신 고객 발화 원문으로 기록하며 현재 조치 선택을 검사한다. policy_source_ids는 검색/조회한 정책 출처, policy_reason은 적용 조건·예외 판단이다. 자동 이관/API-35와 메모/API-26은 고객 인용을 비워도 된다. API-35의 issue_key는 주문번호:식품안전 또는 주문번호:광고정정처럼 대상·사안별로 정하고 재시도에 그대로 유지한다. 같은 레인의 서로 다른 사안은 다른 issue_key를 사용한다. issue_key는 API 본문에 넣지 않는다. 재시도는 같은 본문, replace_previous는 고객이 기존 접수 변경을 명시한 때만 true이며 API-35 본문 변경에는 쓰지 못한다.\n" + api.describe("POST"))
    async def action_submit(api_id: str, parameters: dict, body: dict,
                            policy_source_ids: list[str], policy_reason: str,
                            runtime: ToolRuntime, customer_confirmation: str = "",
                            replace_previous: bool = False, issue_key: str = "") -> Command:
        if os.getenv("ACTION_API_WRITES_ENABLED", "false").lower() != "true":
            return _message(runtime, _failure("접수 도구가 운영 설정에서 비활성화되어 있습니다. 접수했다고 안내하지 마세요.", status="failed"))
        try:
            api.prepare(api_id, parameters, body, "POST")
        except ValueError as error:
            return _message(runtime, _failure(str(error), status="failed"))
        customer_confirmation = _latest_customer_text(runtime.state)
        async def fetch(operation, targets):
            return await api.request(operation, tc_id=runtime.state.get("tc_id", ""), parameters=targets)
        denied, observations = await access.authorize_submit(api_id, parameters, body, runtime.state, fetch)
        if denied is not None:
            return _message(runtime, denied, action_observations=observations)
        state = access.working_state(runtime.state, observations)
        blocked = submission_guard(api_id, parameters, body, state, policy_source_ids,
                                   customer_confirmation, policy_reason)
        if blocked:
            return _message(runtime, _failure(blocked), action_observations=observations)
        try:
            key, canonical = submission_identity(api_id, parameters, body, issue_key)
        except ValueError as error:
            return _message(runtime, _failure(str(error)))
        previous = runtime.state.get("action_requests", {}).get(key)
        if previous and previous["canonical"] != canonical and previous["result"]["status"] in {"unknown", "uncertain"}:
            return _message(runtime, _failure("이전 접수 결과가 미확인이라 본문을 변경할 수 없습니다. 기존 본문을 그대로 재시도하세요."))
        if previous and previous["canonical"] != canonical and (not replace_previous or api_id == "API-35"):
            return _message(runtime, _failure("같은 대상의 접수 본문이 달라졌습니다. 기존 본문을 그대로 재시도하거나 고객의 명시적 변경을 확인하세요."))
        if previous and previous["canonical"] == canonical and previous["result"]["status"] == "submitted":
            return _message(runtime, _receipt_guidance(api_id, body,
                {**previous["result"], "reused_receipt": True}, runtime.state, sources))
        result = await api.request(api_id, tc_id=runtime.state["tc_id"], parameters=parameters, body=body, method="POST")
        journal = submission_record({"api_id": api_id, "issue_key": issue_key,
            "policy_source_ids": policy_source_ids, "policy_reason": policy_reason,
            "customer_confirmation": customer_confirmation}, canonical, result)
        return _message(runtime, _receipt_guidance(api_id, body, result, runtime.state, sources),
                        action_requests={key: journal}, action_observations=observations)

    media_path = ROOT / "media.py"
    spec = importlib.util.spec_from_file_location(f"{__name__}_media", media_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(media_path.read_bytes(), str(media_path), "exec"), module.__dict__)
    order_path = ROOT / "order_context.py"
    order_spec = importlib.util.spec_from_file_location(f"{__name__}_order_context", order_path)
    order_module = importlib.util.module_from_spec(order_spec)
    sys.modules[order_spec.name] = order_module
    exec(compile(order_path.read_bytes(), str(order_path), "exec"), order_module.__dict__)
    return [case_context, search_policy, read_policy, action_get, action_submit, calculate,
            module.build_media_tool(secret("OPENAI_API_KEY")),
            order_module.build_order_tool(api, access, result_guidance=_shipping_guidance)]
