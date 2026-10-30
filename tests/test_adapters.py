"""Reviewed pack adapters use the installed SDK's middleware lifecycle."""
import json
import shutil
import sys
import typing
from datetime import UTC, datetime, timedelta

import pytest
import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from kakao_bot.runtime import AgentRuntime, load_pack, lock_pack
from kakao_bot.runtime.agent import RESERVED_MIDDLEWARE_NAMES, pack_middleware
from test_runtime import ScriptedModel, answer, config, tool_call


def adapter_pack(tmp_path, source, relative="adapters.py"):
    path = tmp_path / "equipment_demo"
    shutil.copytree("domain_packs/equipment_demo", path)
    module = path / relative
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(source)
    manifest = yaml.safe_load((path / "pack.yaml").read_text())
    manifest["adapters_module"] = relative
    (path / "pack.yaml").write_text(yaml.safe_dump(manifest))
    lock_pack(path)
    return load_pack(tmp_path, "equipment_demo")


def test_adapter_is_optional_and_custom_locked_path_is_supported(tmp_path):
    assert load_pack("domain_packs", "equipment_demo").adapter_module() is None
    pack = adapter_pack(tmp_path, "value = 42\n", "extensions/hooks.py")
    assert pack.adapter_module().value == 42
    assert pack_middleware(pack) == []
    (pack.root / "extensions/hooks.py").write_text("value = 99\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        pack.adapter_module()


def test_adapter_dataclass_annotations_and_failed_import_cleanup(tmp_path):
    source = """from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Options:
    enabled: bool = True
"""
    pack = adapter_pack(tmp_path, source)
    module = pack.adapter_module()
    assert module.Options().enabled is True
    assert typing.get_type_hints(module.Options) == {"enabled": bool}
    assert sys.modules[module.__name__] is module
    (pack.root / "adapters.py").write_text(source + '\nraise RuntimeError("broken adapter")\n')
    lock_pack(pack.root)
    broken = load_pack(tmp_path, "equipment_demo")
    name = f"domain_pack_{broken.digest}_adapters"
    with pytest.raises(RuntimeError, match="broken adapter"):
        broken.adapter_module()
    assert name not in sys.modules


@pytest.mark.parametrize("relative", [None, 42, "", "/tmp/adapters.py", "../adapters.py",
                                     "./adapters.py", "adapters.json", "missing.py"])
def test_invalid_adapter_reference_is_rejected(tmp_path, relative):
    pack = adapter_pack(tmp_path, "pass\n")
    manifest = dict(pack.manifest, adapters_module=relative)
    (pack.root / "pack.yaml").write_text(yaml.safe_dump(manifest))
    lock_pack(pack.root)
    with pytest.raises(ValueError, match="Pack module"):
        load_pack(tmp_path, "equipment_demo")


@pytest.mark.parametrize("source", [
    "build_middleware = None\n",
    "async def build_middleware(pack):\n    return []\n",
    "def build_middleware(pack):\n    return None\n",
    "def build_middleware(pack):\n    return ()\n",
    "def build_middleware(pack):\n    return [object()]\n",
])
def test_invalid_middleware_factory_is_rejected(tmp_path, source):
    with pytest.raises(ValueError, match="build_middleware"):
        pack_middleware(adapter_pack(tmp_path, source))


@pytest.mark.parametrize("name", sorted(RESERVED_MIDDLEWARE_NAMES))
def test_core_middleware_cannot_be_replaced(tmp_path, name):
    source = f"""
from langchain.agents.middleware import AgentMiddleware
class Hook(AgentMiddleware):
    name = {name!r}
def build_middleware(pack):
    return [Hook()]
"""
    with pytest.raises(ValueError, match="Reserved"):
        pack_middleware(adapter_pack(tmp_path, source))


def test_duplicate_names_and_middleware_tool_injection_are_rejected(tmp_path):
    source = """
from langchain.agents.middleware import AgentMiddleware
class Hook(AgentMiddleware):
    pass
def build_middleware(pack):
    return [Hook(), Hook()]
"""
    pack = adapter_pack(tmp_path, source)
    with pytest.raises(ValueError, match="duplicate"):
        pack_middleware(pack)
    (pack.root / "adapters.py").write_text("""
from langchain.agents.middleware import AgentMiddleware
from kakao_bot.runtime.tools import calculate
class Hook(AgentMiddleware):
    tools = [calculate]
def build_middleware(pack):
    return [Hook()]
""")
    lock_pack(pack.root)
    with pytest.raises(ValueError, match="Middleware tools are forbidden"):
        pack_middleware(load_pack(tmp_path, "equipment_demo"))


LIFECYCLE_ADAPTER = """
import json
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

def build_middleware(pack):
    events = pack.data.setdefault("hook_events", [])
    class ContextAndToolHook(AgentMiddleware):
        def before_model(self, state, runtime):
            events.append("before_model")
            return {"messages": [SystemMessage(content="adapter context marker", id="adapter-context")]}
        def after_model(self, state, runtime):
            events.append("after_model")
        async def awrap_tool_call(self, request, handler):
            events.append("before_tool")
            result = await handler(request)
            data = json.loads(result.content)
            data["adapter_observed"] = True
            events.append("after_tool")
            return result.model_copy(update={"content": json.dumps(data)})
    return [ContextAndToolHook()]
"""


@pytest.mark.asyncio
async def test_context_and_tool_hooks_execute_in_real_sdk_graph(tmp_path):
    pack = adapter_pack(tmp_path, LIFECYCLE_ADAPTER)
    model = ScriptedModel(responses=[tool_call("calculate", {"expression": "2 + 3"}),
                                    AIMessage(content=json.dumps(answer("계산 결과는 5입니다.", sources=[])))])
    async with AgentRuntime(config(packs_root=tmp_path), checkpointer=InMemorySaver(), model=model) as runtime:
        result = await runtime.run(text="2+3?", thread_id="hooks", event_id="hook-event",
            pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=20))
        assert result.text == "계산 결과는 5입니다."
        assert runtime.packs[pack.digest].data["hook_events"] == [
            "before_model", "after_model", "before_tool", "after_tool", "before_model", "after_model"]
        assert all(any(m.content == "adapter context marker" for m in request) for request in model.requests)
        state = await runtime.graphs[pack.digest].aget_state({"configurable": {"thread_id": "hooks"}})
        message = next(m for m in state.values["messages"] if isinstance(m, ToolMessage))
        assert json.loads(message.content)["adapter_observed"] is True
        assert result.usage["tool_calls"] == 1
        assert set(runtime.tool_names[pack.digest]) == set(pack.manifest["allowed_tools"]) | {
            "ls", "read_file", "glob", "grep", "write_file", "edit_file"}


@pytest.mark.asyncio
async def test_tool_policy_timeout_still_wraps_custom_hook(tmp_path):
    pack = adapter_pack(tmp_path, """
import asyncio
from langchain.agents.middleware import AgentMiddleware
class SlowHook(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        await asyncio.sleep(1)
        return await handler(request)
def build_middleware(pack):
    return [SlowHook()]
""")
    model = ScriptedModel(responses=[tool_call("calculate", {"expression": "2+3"}),
                                    AIMessage(content=json.dumps(answer("조회 시간이 초과되었습니다.", sources=[])))])
    async with AgentRuntime(config(packs_root=tmp_path, tool_timeout_seconds=.02),
                            checkpointer=InMemorySaver(), model=model) as runtime:
        result = await runtime.run(text="2+3?", thread_id="timeout", event_id="timeout-event",
            pack_ref=pack.snapshot, deadline=datetime.now(UTC) + timedelta(seconds=10))
        assert result.usage["tool_calls"] == 1
        assert any(e["kind"] == "tool_error" and e["error_type"] == "TimeoutError" for e in result.trace)
        observed = [m for m in model.requests[-1] if isinstance(m, ToolMessage)]
        assert observed[0].status == "error"
        assert json.loads(observed[0].content)["status"] == "failed"
