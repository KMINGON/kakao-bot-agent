"""Readiness/authentication smoke check. Never queues a reply or calls a model."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def run(base_url: str, secret: str) -> dict:
    results = []
    header = {"Content-Type": "application/json", "X-Bot-Secret": secret}
    message = {
        "op": 0,
        "t": "MESSAGE_CREATE",
        "d": {
            "id": 558185822630215770,
            "content": "HTTP smoke: callback token intentionally omitted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "isChannelChatroom": True,
            "botGroupKey": "smoke_no_reply",
            "botUserKey": "smoke_no_reply",
        },
    }
    cases = [
        ("ready", "/health/ready", None, {}, 200),
        ("missing_auth", "/webhooks/kakao", b"{}", {}, 401),
        ("wrong_auth", "/webhooks/kakao", b"{}", {**header, "X-Bot-Secret": "invalid"}, 401),
        ("invalid_json", "/webhooks/kakao", b"{", header, 400),
        ("unsupported_event", "/webhooks/kakao", json.dumps({"op": 0, "t": "UNSUPPORTED", "d": {}}).encode(), header, 204),
        ("no_callback_token", "/webhooks/kakao", json.dumps(message).encode(), header, 204),
    ]
    for name, path, data, headers, expected in cases:
        started = time.monotonic()
        request = Request(base_url.rstrip("/") + path, data=data, headers={"User-Agent": "kakao-bot-agent-smoke/0.1", **headers})
        try:
            with urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        results.append({
            "case": name,
            "expected_status": expected,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "body_bytes": len(body),
            "passed": status == expected and (expected != 204 or len(body) == 0),
        })
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "scope": "HTTP/TLS/auth and ignored events only; no model or callback",
        "passed": all(row["passed"] for row in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--secret-file", default=".local/secret/webhook_secret")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args.base_url, Path(args.secret_file).read_text().strip())
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized)
    print(serialized)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

