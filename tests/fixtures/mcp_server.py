"""Local fixture, never an external service or paid model."""
import argparse
import os

from fastmcp import FastMCP

server = FastMCP("fixture")


@server.tool
def multiply(a: int, b: int) -> dict:
    """Multiply two integers."""
    return {"product": a * b, "model_secret_inherited": bool(os.getenv("OPENAI_API_KEY"))}


@server.tool
def forbidden_extra() -> str:
    """Not permitted by the pack allowlist."""
    return "must not be visible"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.port:
        server.run(transport="http", host="127.0.0.1", port=args.port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)
