"""Observe actual uptime without model calls or Kakao messages.

The monitor shares the host's failure domain. Missing samples are not successes;
this does not claim availability while the PC or its Internet is unavailable.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch(url: str, token: str | None = None) -> dict:
    started = time.monotonic()
    headers = {"User-Agent": "kakao-bot-agent-monitor/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=8) as response:
            body = response.read(64_000)
            result = {"ok": response.status == 200, "status": response.status}
            try:
                result["body"] = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                result["ok"] = False
                result["error"] = "non_json_response"
    except HTTPError as error:
        result = {"ok": False, "status": error.code}
    except (URLError, TimeoutError, OSError) as error:
        result = {"ok": False, "error": type(error).__name__}
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result


def main() -> None:
    directory = Path(os.environ.get("MONITOR_OUTPUT_DIR", ".local/implementation/v1/evidence/monitor"))
    directory.mkdir(parents=True, exist_ok=True)
    interval = max(10, int(os.environ.get("MONITOR_INTERVAL_SECONDS", "60")))
    duration = float(os.environ.get("MONITOR_DURATION_HOURS", "48"))
    state_file = directory / "run.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        started = utc_now()
        state = {
            "started_at": started.isoformat(),
            "ends_at": (started + timedelta(hours=duration)).isoformat(),
            "duration_seconds": duration * 3600,
            "interval_seconds": interval,
            "scope": "local monitor; no paid model calls or Kakao messages",
        }
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    ends_at = datetime.fromisoformat(state["ends_at"])
    internal = os.environ.get("MONITOR_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")
    public = os.environ.get("MONITOR_PUBLIC_URL", "https://kakao.mingon.dev").rstrip("/")
    token_path = os.environ.get("OPS_TOKEN_FILE")
    token = Path(token_path).read_text().strip() if token_path else None
    if utc_now() >= ends_at:
        print("Observation window already ended; create a new output directory to start another run.", flush=True)
        return
    with (directory / "observations.jsonl").open("a", buffering=1) as output:
        while True:
            tick = time.monotonic()
            observed_at = utc_now()
            if "first_observed_at" not in state:
                state["first_observed_at"] = observed_at.isoformat()
                ends_at = observed_at + timedelta(seconds=state.get("duration_seconds", duration * 3600))
                state["ends_at"] = ends_at.isoformat()
                state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            row = {
                "observed_at": observed_at.isoformat(),
                "internal": fetch(internal + "/health/ready"),
                "public": fetch(public + "/health/ready"),
            }
            if token:
                row["ops"] = fetch(internal + "/ops/status", token)
                status = row["ops"].get("body", {})
                age = status.get("worker_heartbeat_age_seconds")
                row["ops"]["worker_healthy"] = status.get("db") == "ok" and isinstance(age, (int, float)) and age <= 30
                row["ops"]["ok"] = row["ops"]["ok"] and row["ops"]["worker_healthy"]
            row["ok"] = all(value.get("ok", False) for value in row.values() if isinstance(value, dict))
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"observed_at": row["observed_at"], "ok": row["ok"]}), flush=True)
            if observed_at >= ends_at:
                break
            time.sleep(max(0, min(interval - (time.monotonic() - tick), (ends_at - utc_now()).total_seconds())))
    (directory / "finished.json").write_text(json.dumps({"finished_at": utc_now().isoformat()}) + "\n")


if __name__ == "__main__":
    main()

