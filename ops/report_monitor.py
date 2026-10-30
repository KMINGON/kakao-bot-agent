"""Summarize recorded observations without filling gaps with invented successes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def summarize(directory: Path) -> dict:
    state = json.loads((directory / "run.json").read_text())
    rows = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines() if line.strip()]
    interval = state["interval_seconds"]
    stamps = [datetime.fromisoformat(row["observed_at"]) for row in rows]
    gaps = []
    for previous, current in zip(stamps, stamps[1:]):
        seconds = (current - previous).total_seconds()
        if seconds > interval * 1.5:
            gaps.append({"from": previous.isoformat(), "to": current.isoformat(), "seconds": seconds})
    observed_seconds = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) > 1 else 0
    report = {
        "started_at": state["started_at"],
        "ends_at": state["ends_at"],
        "first_observed_at": stamps[0].isoformat() if stamps else None,
        "last_observed_at": stamps[-1].isoformat() if stamps else None,
        "observed_hours": round(observed_seconds / 3600, 3),
        "observations": len(rows),
        "failed_observations": sum(not row["ok"] for row in rows),
        "gaps": gaps,
        "forty_eight_hours_observed": observed_seconds >= 48 * 3600,
        "continuous_observation_passed": observed_seconds >= 48 * 3600 and not gaps and all(row["ok"] for row in rows),
        "scope": "periodic observations; local PC outage also stops this monitor",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".local/implementation/v1/evidence/monitor")
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.directory)), ensure_ascii=False, indent=2))
