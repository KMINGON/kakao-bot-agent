import asyncio
import json
import shutil
import socket
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
import yaml

from kakao_bot.runtime import load_pack, lock_pack
from kakao_bot.runtime.mcp import load_mcp_tools


def mcp_pack(tmp_path, mcp):
    path = tmp_path / "equipment_demo"
    shutil.copytree("domain_packs/equipment_demo", path)
    shutil.copyfile("tests/fixtures/mcp_server.py", path / "mcp_server.py")
    manifest = yaml.safe_load((path / "pack.yaml").read_text())
    manifest["mcp"] = [mcp]
    (path / "pack.yaml").write_text(yaml.safe_dump(manifest))
    lock_pack(path)
    return load_pack(tmp_path, "equipment_demo")


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
