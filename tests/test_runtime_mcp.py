import asyncio
import json
import shutil
import socket
import subprocess
import sys
from contextlib import AsyncExitStack

import pytest
import yaml
from fastmcp.client.transports import StdioTransport

from kakao_bot.runtime import load_pack, lock_pack
from kakao_bot.runtime.mcp import load_mcp_tools


def mcp_pack(tmp_path, mcp, adapters_source=None):
    path = tmp_path / "equipment_demo"
    shutil.copytree("domain_packs/equipment_demo", path)
    shutil.copyfile("tests/fixtures/mcp_server.py", path / "mcp_server.py")
    manifest = yaml.safe_load((path / "pack.yaml").read_text())
    manifest["mcp"] = mcp if isinstance(mcp, list) else [mcp]
    if adapters_source is not None:
        (path / "adapters.py").write_text(adapters_source)
        manifest["adapters_module"] = "adapters.py"
    (path / "pack.yaml").write_text(yaml.safe_dump(manifest))
    lock_pack(path)
    return load_pack(tmp_path, "equipment_demo")


CUSTOM_STDIO_ADAPTER = """
import sys
from fastmcp.client.transports import StdioTransport

def build_mcp_transport(spec, pack):
    return StdioTransport(
        sys.executable, [str(pack.root / spec["script"])],
        env={}, cwd=str(pack.root), keep_alive=True,
    )
"""


@pytest.mark.asyncio
async def test_stdio_discovery_allowlist_call_and_secret_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-secret")
    pack = mcp_pack(tmp_path, {"transport": "stdio", "script": "mcp_server.py", "allow_tools": ["multiply"]})
    async with AsyncExitStack() as stack:
        tools = await load_mcp_tools(pack, stack)
        assert [tool.name for tool in tools] == ["multiply"]
        result = await tools[0].ainvoke({"a": 6, "b": 7})
        text = json.dumps(result)
        assert '42' in text
        assert 'true' not in text.lower()


@pytest.mark.asyncio
async def test_http_mcp_discovery_and_call(tmp_path):
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    process = subprocess.Popen([sys.executable, "tests/fixtures/mcp_server.py", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(.05)
        else:
            raise AssertionError("MCP fixture failed to start")
        pack = mcp_pack(tmp_path, {"transport": "streamable_http", "url": f"http://127.0.0.1:{port}/mcp", "allow_tools": ["multiply"]})
        async with AsyncExitStack() as stack:
            tools = await load_mcp_tools(pack, stack)
            result = await tools[0].ainvoke({"a": 7, "b": 8})
            assert "56" in json.dumps(result)
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.asyncio
async def test_mcp_config_rejects_unreviewed_transports_and_missing_tools(tmp_path):
    pack = mcp_pack(tmp_path, {"transport": "stdio", "script": "../outside.py", "allow_tools": ["multiply"]})
    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match="inside"):
            await load_mcp_tools(pack, stack)


@pytest.mark.asyncio
async def test_custom_transport_discovery_call_and_secret_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-secret")
    closed = []
    close = StdioTransport.close

    async def track_close(transport):
        await close(transport)
        closed.append(transport)

    monkeypatch.setattr(StdioTransport, "close", track_close)
    pack = mcp_pack(tmp_path, {
        "transport": "custom", "script": "mcp_server.py", "allow_tools": ["multiply"],
    }, CUSTOM_STDIO_ADAPTER)
    async with AsyncExitStack() as stack:
        tools = await load_mcp_tools(pack, stack)
        assert [tool.name for tool in tools] == ["multiply"]
        result = json.dumps(await tools[0].ainvoke({"a": 8, "b": 9}))
        assert "72" in result
        assert "true" not in result.lower()
    assert len(closed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("source,error,match", [
    (None, ValueError, "requires build_mcp_transport"),
    ("build_mcp_transport = None", ValueError, "requires build_mcp_transport"),
    ("def build_mcp_transport(spec, pack):\n    return 'stdio'", TypeError, "ClientTransport"),
])
async def test_custom_transport_requires_factory_and_transport(tmp_path, source, error, match):
    pack = mcp_pack(tmp_path, {"transport": "custom", "allow_tools": ["multiply"]}, source)
    async with AsyncExitStack() as stack:
        with pytest.raises(error, match=match):
            await load_mcp_tools(pack, stack)


@pytest.mark.asyncio
@pytest.mark.parametrize("allow,match", [([], "explicit tool allowlist"), (["absent"], "tools missing")])
async def test_custom_transport_preserves_tool_allowlist(tmp_path, allow, match):
    pack = mcp_pack(tmp_path, {
        "transport": "custom", "script": "mcp_server.py", "allow_tools": allow,
    }, CUSTOM_STDIO_ADAPTER)
    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match=match):
            await load_mcp_tools(pack, stack)


@pytest.mark.asyncio
async def test_custom_and_builtin_transports_reject_duplicate_tools(tmp_path):
    pack = mcp_pack(tmp_path, [
        {"transport": "stdio", "script": "mcp_server.py", "allow_tools": ["multiply"]},
        {"transport": "custom", "script": "mcp_server.py", "allow_tools": ["multiply"]},
    ], CUSTOM_STDIO_ADAPTER)
    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match="Duplicate MCP tool names"):
            await load_mcp_tools(pack, stack)
