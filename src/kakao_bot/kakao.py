"""One-shot Kakao callback transport. No automatic retry after an ambiguous send."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

CALLBACK_URL = "https://kapi.kakao.com/v1/bot/callback"
COMMANDS_URL = "https://kapi.kakao.com/v1/bot/commands"


@dataclass(frozen=True)
class DeliveryResult:
    status: str
    code: str
    counts: dict[str, Any]


def classify_response(response: httpx.Response) -> DeliveryResult:
    try:
        body = response.json()
    except ValueError:
        return DeliveryResult("unknown", "invalid_json", {})
    if not isinstance(body, dict):
        return DeliveryResult("unknown", "invalid_body", {})
    counts = {k: body.get(k) for k in ("totalCount", "sendCount", "errorCount") if k in body}
    if response.status_code == 200:
        if len(counts) != 3 or any(type(value) is not int or value < 0 for value in counts.values()):
            return DeliveryResult("unknown", "invalid_counts", counts)
        if counts == {"totalCount": 1, "sendCount": 1, "errorCount": 0}:
            return DeliveryResult("sent", "ok", counts)
        if body.get("sendCount") == 0 and body.get("errorCount", 0) > 0:
            return DeliveryResult("rejected", "callback_errors", counts)
        return DeliveryResult("unknown", "unexpected_counts", counts)
    if body.get("code") == -881:
        return DeliveryResult("unknown", "token_expired_or_used", counts)
    if 400 <= response.status_code < 500:
        return DeliveryResult("rejected", f"http_{response.status_code}", counts)
    return DeliveryResult("unknown", f"http_{response.status_code}", counts)


class KakaoClient:
    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None):
        self.token = token
        self._owned = client is None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(10, connect=3), follow_redirects=False)

    async def close(self) -> None:
        if self._owned:
            await self.client.aclose()

    async def send(self, callback_token: str, payload: dict[str, Any]) -> DeliveryResult:
        try:
            response = await self.client.post(
                CALLBACK_URL,
                headers={"Authorization": f"KakaoAK {self.token}", "X-Bot-Callback-Token": callback_token},
                json=payload,
            )
            return classify_response(response)
        except httpx.HTTPError:
            return DeliveryResult("unknown", "transport_error", {})

    async def read_check(self) -> dict[str, Any]:
        response = await self.client.get(COMMANDS_URL, headers={"Authorization": f"KakaoAK {self.token}"})
        return {"status_code": response.status_code, "ok": response.status_code == 200}
