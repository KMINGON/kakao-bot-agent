"""MCP connections are selected by the deployed pack, never by chat input."""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import urlparse

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain.mcp import MCPAdapter


async def load_mcp_tools(pack, stack: AsyncExitStack):
    tools = []
    for spec in pack.manifest.get("mcp", []):
        allow = set(spec.get("allow_tools", []))
        if not allow:
            raise ValueError("MCP requires an explicit tool allowlist")
        if spec["transport"] == "stdio":
            script = (pack.root / spec["script"]).resolve()
            if not script.is_relative_to(pack.root) or not script.is_file():
                raise ValueError("MCP script must be inside the locked pack")
            env = {name: os.environ[name] for name in spec.get("env_allowlist", [])}
            # A declared script receives only explicitly selected credentials.
            target = StdioTransport(sys.executable, [str(script), *spec.get("args", [])],
                                    env=env, cwd=str(pack.root), keep_alive=True)
        elif spec["transport"] == "streamable_http":
            url = spec["url"]
            parsed = urlparse(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Invalid MCP URL")
            if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Remote MCP requires HTTPS")
            headers = {header: os.environ[name] for header, name in spec.get("headers_env", {}).items()}
            target = StreamableHttpTransport(url, headers=headers)
        else:
            raise ValueError("Unsupported MCP transport")
        adapter = await stack.enter_async_context(MCPAdapter(target))
        discovered = await adapter.list_tools()
        names = {tool.name for tool in discovered}
        if not allow <= names:
            raise ValueError(f"MCP allowlisted tools missing: {sorted(allow - names)}")
        tools.extend(tool for tool in discovered if tool.name in allow)
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("Duplicate MCP tool names")
    return tools
