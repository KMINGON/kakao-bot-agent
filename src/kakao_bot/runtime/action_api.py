"""The contest's fixed OpenAPI boundary. No arbitrary URLs or reset operation."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator, FormatChecker, validators


BASE_URL = "https://gyodong-mall-action.aitop100.org"
TC_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_STRICT_TYPES = Draft202012Validator.TYPE_CHECKER.redefine(
    "integer", lambda checker, value: type(value) is int)
_Validator = validators.extend(Draft202012Validator, type_checker=_STRICT_TYPES)
_FORMATS = FormatChecker()


@_FORMATS.checks("int32")
def _int32(value):
    return type(value) is int and -(2 ** 31) <= value < 2 ** 31


class ActionAPI:
    def __init__(self, spec: dict, token: str, *, client: httpx.AsyncClient | None = None):
        self.spec, self.token, self.client = spec, token, client
        self.operations = {
            op["operationId"]: {**op, "method": method.upper(), "path": path.split("?", 1)[0]}
            for path, methods in spec["paths"].items()
            for method, op in methods.items()
            if method in {"get", "post"} and re.fullmatch(r"API-\d{2}", op.get("operationId", ""))
        }

    def validate(self, value: Any, schema: dict) -> None:
        errors = list(_Validator({**schema, "components": self.spec.get("components", {})},
                                 format_checker=_FORMATS).iter_errors(value))
        if errors:
            # Values can contain personal information; expose schema locations, not payloads.
            error = errors[0]
            location = ".".join(map(str, error.absolute_path)) or "root"
            raise ValueError(f"Invalid API field {location}: {error.validator}")

    def prepare(self, api_id: str, parameters: dict, body: dict | None, method: str):
        op = self.operations.get(api_id)
        if op is None or op["method"] != method:
            raise ValueError("Unknown or disallowed API operation")
        fields = [p for p in op.get("parameters", []) if p["in"] in {"path", "query"}]
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {p["name"]: p["schema"] for p in fields},
                  "required": [p["name"] for p in fields if p.get("required")]}
        self.validate(parameters, schema)
        path, query = op["path"], {}
        for p in fields:
            value = parameters.get(p["name"])
            if value is None:
                continue
            if not value.strip() or len(value) > 256:
                raise ValueError("Empty or excessive API target")
            if p["in"] == "path":
                if value in {".", ".."}:
                    raise ValueError("Invalid API target")
                path = path.replace("{" + p["name"] + "}", quote(value, safe=""))
            else:
                query[p["name"]] = value
        if method == "GET":
            if body:
                raise ValueError("GET has no request body")
            payload = None
        else:
            payload = {} if body is None else body
            request_schema = op["requestBody"]["content"]["application/json"]["schema"]
            self.validate(payload, request_schema)
        return op, path, query, payload

    def describe(self, method: str) -> str:
        lines = []
        for api_id, op in self.operations.items():
            if op["method"] != method:
                continue
            params = [p["name"] for p in op.get("parameters", []) if p["in"] in {"path", "query"}]
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
            fields = [f"{key}:{value.get('type', 'value')}" for key, value in body.get("properties", {}).items()]
            lines.append(f"{api_id} {op['summary']} parameters={','.join(params) or '{}'}"
                         + (f" body={','.join(fields) or '{}'}" if method == "POST" else ""))
        return "\n".join(lines)

    async def request(self, api_id: str, *, tc_id: str, parameters: dict,
                      body: dict | None = None, method: str = "GET", max_attempts: int = 2) -> dict:
        if max_attempts not in {1, 2}:
            raise ValueError("Action API attempts must be 1 or 2")
        op, path, query, payload = self.prepare(api_id, parameters, body, method)
        if not isinstance(tc_id, str) or not TC_PATTERN.fullmatch(tc_id):
            raise ValueError("A valid conversation TC ID is required")
        if not self.token:
            return {"status": "failed", "error": "action_api_not_configured", "evidence": []}
        # Serialize once: retries must never let a model regenerate the receipt body.
        encoded = None if payload is None else json.dumps(payload, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()
        headers = {"Authorization": f"Bearer {self.token}", "X-Aitop100-Test-Case-Id": tc_id}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        details = {"api_id": api_id, "tc_id": tc_id, "parameters": parameters,
                   "body": payload, "effect": "receipt" if method == "POST" else "read_only"}
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=5.0, follow_redirects=False)
        try:
            for attempt in range(max_attempts):
                try:
                    response = await client.request(method, BASE_URL + path, params=query,
                                                    content=encoded, headers=headers)
                except httpx.TransportError:
                    if attempt + 1 < max_attempts:
                        continue
                    return {**details, "status": "uncertain" if method == "POST" else "failed",
                            "error": "transport_error", "evidence": [],
                            "message": "접수 여부 미확인. 재시도하려면 동일 본문을 유지하세요."}
                if response.status_code == 200:
                    try:
                        if len(response.content) > 250_000:
                            raise ValueError("Oversize response")
                        data = response.json()
                        self.validate(data, op["responses"]["200"]["content"]["application/json"]["schema"])
                        if method == "POST" and data.get("request_status") != "SUBMITTED":
                            raise ValueError("Receipt status missing")
                        expected = {p["name"]: parameters[p["name"]] for p in op.get("parameters", [])
                                    if p["in"] == "path"}
                        if method == "POST":
                            expected.update({k: v for k, v in payload.items()
                                             if k in {"order_no", "shipment_id", "member_id", "approval_no", "subscription_id", "sku"}})
                        if isinstance(data, dict) and any(k in data and data[k] != v for k, v in expected.items()):
                            raise ValueError("API returned a different target")
                    except (ValueError, TypeError):
                        return {**details, "status": "uncertain", "error": "invalid_response", "evidence": []}
                    return {**details, "status": "submitted" if method == "POST" else "ok", "data": data,
                            "evidence": [{"source_id": api_id, "locator": path,
                                          "retrieved_at": datetime.now(UTC).isoformat()}],
                            "message": "요청 접수이며 실제 상태·금액 변경 완료가 아닙니다." if method == "POST" else "조회 성공"}
                try:
                    error = response.json()
                    error = error if isinstance(error, dict) else {}
                except ValueError:
                    error = {}
                if method == "GET" and response.status_code == 404:
                    return {**details, "status": "failed", "http_status": 404, "not_found": True,
                            "error": error.get("code", "http_error"),
                            "evidence": [{"source_id": api_id, "locator": path,
                                          "http_status": 404, "not_found": True,
                                          "retrieved_at": datetime.now(UTC).isoformat()}],
                            "message": "조회한 대상이 없습니다. 다른 대상·혜택의 부재나 접수 가능 여부의 근거가 아닙니다."}
                exhausted = error.get("message") == "누적 API 호출 한도를 초과했습니다."
                retry = attempt + 1 < max_attempts and (response.status_code in {500, 503} or
                                          response.status_code == 429 and not exhausted)
                if retry:
                    try:
                        delay = float(response.headers.get("Retry-After", "0.1"))
                    except ValueError:
                        delay = 0.1
                    if 0 <= delay <= 1:
                        await asyncio.sleep(delay)
                        continue
                return {**details, "status": "uncertain" if method == "POST" and response.status_code >= 500 else "failed",
                        "http_status": response.status_code, "error": error.get("code", "http_error"),
                        "limit_exhausted": exhausted, "retry_after": response.headers.get("Retry-After"),
                        "evidence": []}
        finally:
            if own_client:
                await client.aclose()
