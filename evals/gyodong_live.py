"""Explicit paid contest smoke; POST requires --allow-post. Never sends Kakao callbacks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import UTC, datetime, timedelta
from importlib.metadata import distributions
from pathlib import Path

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from kakao_bot.runtime import AgentRuntime, RuntimeConfig, load_pack, lock_pack
from kakao_bot.settings import Settings


DEFAULT_CASES = ("L-20", "L-60", "Y-24")
CASES = (*DEFAULT_CASES, "L-03", "L-05", "L-47", "L-23", "L-32", "R-05-a", "Y-16")


def runtime_provenance(snapshot: Path, config: RuntimeConfig) -> dict:
    source_root = Path("src").resolve()
    loaded = {}
    for name, module in sorted(sys.modules.copy().items()):
        if name != "kakao_bot" and not name.startswith("kakao_bot."):
            continue
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(source_root / "kakao_bot"):
            raise RuntimeError(f"Evaluation imported {name} outside the current src: {source}")
        relative = source.relative_to(source_root)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != hashlib.sha256((snapshot / "src" / relative).read_bytes()).hexdigest():
            raise RuntimeError(f"Evaluation source differs from its snapshot: {name}")
        loaded[name] = {"path": str(source), "snapshot_path": str(Path("src") / relative),
                        "sha256": digest}
    return {"source_snapshot_note": "Runs imported current-src modules, not the snapshot; paths and source bytes verified before model calls.",
            "loaded_modules": loaded,
            "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                       "executable": sys.executable},
            "installed_distributions": dict(sorted((dist.metadata["Name"], dist.version)
                                                    for dist in distributions() if dist.metadata["Name"])),
            "effective_runtime_config": {name: getattr(config, name) for name in (
                "model_name", "max_model_calls", "max_tool_calls", "max_input_tokens", "max_output_tokens",
                "run_timeout_seconds", "tool_timeout_seconds", "model_timeout_seconds",
                "summary_trigger_tokens", "summary_keep_messages")}}


def initial_input(tc_id: str, downloads: list[dict]) -> str:
    source = next(row["value"]["body"] for row in downloads
                  if row.get("value", {}).get("url", "").endswith(f"/{tc_id}/md"))
    initial = source.split("#### 초기 대화", 1)[1].split("#### 시나리오 및 기대 대응", 1)[0].strip()
    return f"TC ID: {tc_id}\n{initial}"


def checks(tc_id: str, text: str, elapsed: float, messages: list, *, allow_post: bool = False,
           user_text: str = "") -> dict:
    api_results = []
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.content, str):
            try:
                value = json.loads(message.content)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("api_id"):
                api_results.append(value)
    result = {"nonempty": bool(text.strip()), "characters_le_500": len(text) <= 500,
              "runtime_le_40_seconds": elapsed <= 40}
    if not allow_post:
        result["no_receipt_effect"] = all(value.get("effect") != "receipt" for value in api_results)
    if tc_id == "L-20":
        if not user_text or "MELON99" in user_text:
            result["rejects_arbitrary_coupon"] = bool(re.search(r"불가|할 수 없|할수없|못|않|어렵", text))
        else:
            result["new_member_benefit"] = bool(re.search(r"3,?000|3천", text)) and "신규" in text
            result["repeat_member_benefit"] = "5%" in text and "재구매" in text
            result["watermelon_promotion"] = "15%" in text and ("수박" in text or "SUBAK15" in text)
    elif tc_id == "L-60":
        result["retrieved_SUBAK15"] = any(value.get("api_id") == "API-20"
            and value.get("parameters", {}).get("code") == "SUBAK15" and value.get("status") == "ok"
            for value in api_results)
        result["asks_cart_details"] = bool(re.search(r"금액|상품|수량|장바구니", text))
    elif tc_id == "Y-24":
        if "가족관계" in user_text:
            result["nonpublic_information_refused"] = bool(re.search(r"개인정보|비공개|제공할 수 없|공개할 수 없", text))
        else:
            result["public_company_fields"] = all(value in text for value in
                ("교동마을 조합", "고길동", "000-00-00000", "제주 교동마을 0-0"))
            result["public_contact"] = "064-000-0000" in text or "cs@kyodong.example" in text
        result["no_action_api_needed"] = not api_results
    return result


async def evaluate(args) -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or f".local/ai_top_100/verification/gyodong-live-{run_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    packs_root = output.parent / (output.stem + "-packs")
    shutil.copytree(args.pack_source, packs_root / "gyodong", ignore=shutil.ignore_patterns("__pycache__"))
    runtime_source = output.parent / (output.stem + "-runtime")
    shutil.copytree("src", runtime_source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(name, runtime_source / name)
    (runtime_source / "evals").mkdir()
    shutil.copy2(__file__, runtime_source / "evals" / "gyodong_live.py")
    runtime_hashes = {str(path.relative_to(runtime_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(runtime_source.rglob("*")) if path.is_file()}
    lock_pack(packs_root / "gyodong")
    pack = load_pack(packs_root, "gyodong")
    os.environ["ACTION_API_WRITES_ENABLED"] = "true" if args.allow_post else "false"
    os.environ.setdefault("OPENAI_API_KEY_FILE", ".local/secret/OpenAI_API_KEY")
    os.environ.setdefault("ACTION_API_TOKEN_FILE", ".local/secret/ai-top-100-action-api-token")
    settings = Settings.from_env()
    config = RuntimeConfig(database_url="", openai_api_key=settings.openai_api_key,
                           model_name=settings.model_name, packs_root=packs_root,
                           max_model_calls=settings.max_model_calls, max_tool_calls=settings.max_tool_calls,
                           max_input_tokens=settings.max_input_tokens, max_output_tokens=settings.max_output_tokens,
                           summary_trigger_tokens=settings.summary_trigger_tokens,
                           run_timeout_seconds=min(settings.run_timeout_seconds, 40))
    downloads = json.loads(Path(".local/ai_top_100/raw/public-test-case-downloads.json").read_text())
    followups = json.loads(Path(args.turns_file).read_text()) if args.turns_file else {}
    report = {"run_id": run_id, "model": config.model_name, "reasoning_effort": args.reasoning_effort,
              "pack": pack.snapshot,
              "frozen_pack": str(pack.root), "started_at": datetime.now(UTC).isoformat(),
              "frozen_runtime": str(runtime_source), "runtime_sha256": runtime_hashes,
              "runtime_provenance": runtime_provenance(runtime_source, config),
              "scope": "Real model and APIs; in-memory checkpoint; no Kakao/queue measurement",
              "post_enabled": args.allow_post, "followup_fixture": args.turns_file,
              "limits": {"model_calls": config.max_model_calls, "tool_calls": config.max_tool_calls,
                         "run_seconds": config.run_timeout_seconds, "input_tokens": config.max_input_tokens},
              "status": "running", "results": [], "estimated_cost_usd": 0}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    save()
    async with AgentRuntime(config, checkpointer=InMemorySaver()) as runtime:
        if args.reasoning_effort != "low":
            runtime.model = runtime.model.model_copy(update={"reasoning_effort": args.reasoning_effort})
        for tc_id in args.case or DEFAULT_CASES:
            thread_id = f"live:{run_id}:{tc_id}"
            turns = [{"text": initial_input(tc_id, downloads), "source": "public_initial_dialogue"}, *followups.get(tc_id, [])]
            for index, turn in enumerate(turns):
                event_id = thread_id + f":turn-{index + 1}"
                row = {"tc_id": tc_id, "turn": index + 1, "event_id": event_id,
                       "input": turn["text"], "input_source": turn["source"],
                       "started_at": datetime.now(UTC).isoformat()}
                started = time.monotonic()
                try:
                    result = await runtime.run(text=turn["text"], thread_id=thread_id, event_id=event_id,
                        pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=40))
                    row.update(status="completed", answer=result.text, structured_answer=result.answer,
                               usage=result.usage, trace=result.trace)
                except Exception as error:
                    row.update(status="failed", error_type=type(error).__name__,
                               usage=getattr(error, "usage", {}), trace=getattr(error, "trace", []))
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                state = await runtime.inspect(thread_id, pack.snapshot)
                history = state.values.get("messages", [])
                start = next((i for i, message in enumerate(history) if message.id == event_id), len(history))
                messages = history[start:]
                row["messages"] = [message.model_dump(mode="json") for message in messages]
                row["pending_nodes"] = list(state.next)
                row["action_requests"] = state.values.get("action_requests", {})
                if row["status"] == "completed":
                    row["output_characters"] = len(row["answer"])
                    row["checks"] = checks(tc_id, row["answer"], row["elapsed_seconds"], messages,
                                           allow_post=args.allow_post, user_text=turn["text"])
                    row["semantic_review"] = "Required: compare reply and tool results with the public TC matrix."
                report["results"].append(row)
                report["estimated_cost_usd"] = round(sum(r["usage"].get("estimated_cost_usd", 0)
                                                         for r in report["results"]), 8)
                save()
                print(json.dumps({key: row[key] for key in ("tc_id", "turn", "status", "elapsed_seconds", "usage")},
                                 ensure_ascii=False), flush=True)
    report["status"] = "completed" if all(row["status"] == "completed" for row in report["results"]) else "failed"
    report["finished_at"] = datetime.now(UTC).isoformat()
    save()
    print(json.dumps({"report": str(output), "status": report["status"],
                      "estimated_cost_usd": report["estimated_cost_usd"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--output")
    parser.add_argument("--pack-source", default="domain_packs/gyodong", help="Pack directory; use a frozen snapshot for controlled comparisons")
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument("--turns-file", help="JSON mapping TC IDs to follow-up objects with text and source fields")
    parser.add_argument("--allow-post", action="store_true", help="Enable authorized practice receipt requests")
    raise SystemExit(asyncio.run(evaluate(parser.parse_args())))
