"""Checkpointed contest context through the existing LangChain middleware seam."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Overwrite

from .action_api import TC_PATTERN


def merge_records(previous: dict | None, update: dict | None) -> dict:
    return {**(previous or {}), **(update or {})}


class CompetitionState(AgentState):
    tc_id: str
    case_brief: str
    action_observations: Annotated[dict, merge_records]
    action_requests: Annotated[dict, merge_records]
    policy_reads: Annotated[dict, merge_records]


def extract_tc_id(text: str) -> str | None:
    """Accept explicit TC labels, a standalone first-line ID, or JSON envelope metadata."""
    try:
        value = json.loads(text)
        candidate = value.get("test_case_id", value.get("tc_id")) if isinstance(value, dict) else None
        if isinstance(candidate, str) and TC_PATTERN.fullmatch(candidate):
            return candidate
    except ValueError:
        pass
    match = re.search(r"(?:TC\s*(?:ID)?|test[_ -]?case[_ -]?id|테스트\s*케이스\s*(?:ID|아이디))"
                      r"\s*[:=：]\s*[\[\"']?([^\s\]\"',;]+)", text, re.I)
    if match and TC_PATTERN.fullmatch(match.group(1)):
        return match.group(1)
    first = text.strip().splitlines()[0] if text.strip() else ""
    match = re.fullmatch(r"\[?([A-Z][A-Za-z0-9]*-\d+(?:-[A-Za-z0-9]+)?)\]?", first)
    return match.group(1) if match else None


def submission_identity(api_id: str, parameters: dict, body: dict, issue_key: str = "") -> tuple[str, str]:
    if api_id == "API-35":
        if not issue_key.strip() or len(issue_key) > 100:
            raise ValueError("이관 대상·사안을 나타내는 issue_key가 필요합니다. 예: 주문번호:식품안전. 같은 사안의 재시도에는 같은 값을 쓰세요.")
        target = {"lane": body["lane"], "issue_key": issue_key}
    else:
        target = parameters or {"order_no": body["order_no"]}
    key = api_id + ":" + json.dumps(target, sort_keys=True, ensure_ascii=False)
    canonical = json.dumps({"parameters": parameters, "body": body}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return key, canonical


def submission_record(arguments: dict, canonical: str, result: dict) -> dict:
    return {"canonical": canonical, "body_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "api_id": arguments["api_id"], "issue_key": arguments.get("issue_key", ""),
            "policy_source_ids": arguments["policy_source_ids"], "policy_reason": arguments["policy_reason"],
            "customer_confirmation": arguments["customer_confirmation"], "result": result}


def recover_action_requests(state: dict) -> dict:
    if not state.get("tc_id"):
        return {}
    messages = state.get("messages", [])
    results = {message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)}
    recovered = {}
    tc_id = state["tc_id"]
    customer_text = state.get("case_brief", "")
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            tc_id = extract_tc_id(message.content) or tc_id
            customer_text = message.content
        if not isinstance(message, AIMessage) or tc_id != state["tc_id"]:
            continue
        for call in message.tool_calls:
            if call["name"] != "action_submit":
                continue
            result = results.get(call["id"])
            if result is not None:
                try:
                    result = json.loads(result.content)
                except (TypeError, ValueError):
                    continue
                if not isinstance(result, dict) or result.get("status") not in {"unknown", "uncertain"}:
                    continue
            else:
                result = {"status": "unknown", "error": "interrupted_tool_call", "evidence": [],
                          "message": "이전 접수 결과는 미확인입니다. 재시도할 때 기존 요청 본문을 그대로 유지하세요."}
            arguments = call["args"]
            arguments = {**arguments, "customer_confirmation": customer_text}
            try:
                key, canonical = submission_identity(arguments["api_id"], arguments["parameters"],
                                                     arguments["body"], arguments.get("issue_key", ""))
                record = submission_record(arguments, canonical, result)
            except (KeyError, TypeError, ValueError, AttributeError):
                # Invalid tool arguments cannot have reached the HTTP request.
                continue
            if key not in state.get("action_requests", {}) and key not in recovered:
                recovered[key] = record
    return recovered


class ContestContextMiddleware(AgentMiddleware):
    state_schema = CompetitionState

    def before_agent(self, state, runtime):
        latest = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if latest is None or not isinstance(latest.content, str):
            return None
        tc_id = extract_tc_id(latest.content)
        if tc_id is None or tc_id == state.get("tc_id"):
            return None
        return {"tc_id": tc_id, "case_brief": latest.content[:16000],
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), latest],
                "action_observations": Overwrite({}), "action_requests": Overwrite({}),
                "policy_reads": Overwrite({}), "structured_response": None,
                "_summarization_event": None, "_summarization_session_id": None,
                # DeepAgents retains summaries and file-backed scratch separately from messages.
                "files": {path: None for path in state.get("files", {})}}

    def before_model(self, state, runtime):
        recovered = recover_action_requests(state)
        return {"action_requests": recovered} if recovered else None

    async def awrap_model_call(self, request, handler):
        # Avoid concurrent mutations of one receipt target and needless parallel API spending.
        return await handler(request.override(model_settings={**request.model_settings, "parallel_tool_calls": False}))
