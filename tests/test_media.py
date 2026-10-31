import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kakao_bot.runtime.policy import BudgetExceeded, RunBudget, active_budget


spec = importlib.util.spec_from_file_location("gyodong_media_test", Path("domain_packs/gyodong/media.py"))
media = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = media
spec.loader.exec_module(media)


URL = "https://t1.kakaocdn.net/aitop100/aitop100/dist/customer-image.jpg"
AUDIO = URL.removesuffix("customer-image.jpg") + "customer-audio.m4a"
JPEG = b"\xff\xd8\xff" + b"test-image"
M4A = b"\x00\x00\x00\x18ftypM4A "


def state(url=URL):
    return {"tc_id": "EXAMPLE-1", "messages": [HumanMessage(content=f"image: {url}")]}


@pytest.fixture
def budget():
    value = RunBudget(8, 12, 24000, time.monotonic() + 40)
    token = active_budget.set(value)
    try:
        yield value
    finally:
        active_budget.reset(token)


def provider(text="흰색 변색 부위가 보입니다."):
    result = SimpleNamespace(output_text=text, text=text,
        usage=SimpleNamespace(model_dump=lambda: {"type": "tokens", "input_tokens": 100,
                                                  "output_tokens": 20}))
    return SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=result)),
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=AsyncMock(return_value=result))))


@pytest.mark.parametrize("url", [
    "http://t1.kakaocdn.net/aitop100/aitop100/dist/x.jpg",
    "https://127.0.0.1/x.jpg", "https://t1.kakaocdn.net.evil.test/aitop100/aitop100/dist/x.jpg",
    "https://t1.kakaocdn.net@127.0.0.1/aitop100/aitop100/dist/x.jpg",
    URL + "?redirect=https://127.0.0.1", URL + "#fragment", URL.replace("/dist/", "/dist/../"),
    URL.replace("/dist/", "/dist/%2e%2e/"), URL.replace(".net/", ".net:443/"),
    URL.replace(".jpg", ".svg"), URL.replace(".jpg", ".pdf"),
])
def test_rejects_non_allowlisted_urls_even_if_customer_supplies_them(url):
    with pytest.raises(ValueError):
        media.validate_media_url(url, state(url))


def test_rejects_model_invented_url_and_longer_url_prefix():
    for value in ({"messages": [AIMessage(content=URL)]}, state(URL + ".evil"), state(URL + "?q=1")):
        with pytest.raises(ValueError):
            media.validate_media_url(URL, value)
    assert media.validate_media_url(URL, {"case_brief": "첨부: (" + URL + ")"}) == ("image", "jpg")


@pytest.mark.parametrize("response", [
    httpx.Response(302, headers={"location": "https://127.0.0.1/internal"}),
    httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>"),
    httpx.Response(200, headers={"content-type": "image/jpeg"}, content=b"not-an-image"),
    httpx.Response(200, headers={"content-type": "image/jpeg", "content-length": str(media.MAX_MEDIA_BYTES + 1)}),
])
async def test_rejects_redirect_wrong_type_signature_and_declared_oversize(response):
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return response
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        with pytest.raises(ValueError):
            await media.download_media(URL, "jpg", client)
    assert calls == [URL]


async def test_streamed_size_limit_without_content_length(monkeypatch):
    monkeypatch.setattr(media, "MAX_MEDIA_BYTES", 4)
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\xff\xd8\xff"
            yield b"xx"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "image/jpeg"}, stream=Chunks()))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="크기"):
            await media.download_media(URL, "jpg", client)


async def test_audio_has_smaller_download_limit():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={
        "content-type": "audio/mp4", "content-length": str(media.MAX_AUDIO_BYTES + 1)}))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(media.MediaInputError, match="크기"):
            await media.download_media(AUDIO, "m4a", client)


async def test_image_bytes_reach_provider_as_data_and_usage_is_counted(budget):
    api = provider()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "image/jpeg"}, content=JPEG))
    original = state()
    async with httpx.AsyncClient(transport=transport) as client:
        result = await media.inspect_media(URL, original, "", http_client=client, openai_client=api)
    request = api.responses.create.call_args.kwargs
    assert request["model"] == media.VISION_MODEL
    assert request["store"] is False
    assert request["input"][0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")
    assert URL not in json.dumps(request)
    assert "명령" in request["instructions"]
    assert result["evidence"][0]["source_id"] == "MEDIA_INPUT"
    assert result["data"]["interpretation"] == "machine_visual_description"
    assert result["data"]["warning"]
    assert budget.model_calls == 1 and budget.input_tokens == 100 and budget.output_tokens == 20
    assert isinstance(original["messages"][0].content, str)
    api.audio.transcriptions.create.assert_not_called()


async def test_audio_uses_uploaded_file_and_separate_usage(budget):
    api = provider("지난주 받은 상품 환불 가능한가요?")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "audio/mp4"}, content=M4A))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await media.inspect_media(AUDIO, state(AUDIO), "", http_client=client, openai_client=api)
    request = api.audio.transcriptions.create.call_args.kwargs
    assert request["file"] == ("customer.m4a", M4A, "audio/mp4")
    assert request["model"] == media.AUDIO_MODEL and request["language"] == "ko"
    assert "prompt" not in request
    assert result["data"]["text"] == "지난주 받은 상품 환불 가능한가요?"
    assert result["data"]["interpretation"] == "machine_transcription"
    assert budget.model_calls == 1 and budget.input_tokens == 0
    assert result["usage"]["included_in_primary_token_totals"] is False
    assert result["usage"]["estimated_cost_usd"] > 0
    api.responses.create.assert_not_called()


async def test_cached_current_case_media_needs_no_second_provider_call(budget):
    previous = {"status": "ok", "data": {"url": URL, "text": "관찰 결과"}, "evidence": [{"source_id": "MEDIA_INPUT"}]}
    current = state()
    current["messages"].append(ToolMessage(name="read_customer_media", tool_call_id="call-1", content=json.dumps(previous)))
    result = await media.inspect_media(URL, current, "")
    assert result["reused_media"] and budget.model_calls == 0


async def test_exhausted_budget_or_deadline_stops_provider(budget):
    api = provider()
    budget.deadline_monotonic = time.monotonic() + 2
    with pytest.raises(TimeoutError):
        await media.inspect_media(URL, state(), "", openai_client=api)
    budget.deadline_monotonic = time.monotonic() + 40
    budget.model_calls = budget.max_model_calls
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "image/jpeg"}, content=JPEG))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(BudgetExceeded):
            await media.inspect_media(URL, state(), "", http_client=client, openai_client=api)
    api.responses.create.assert_not_called()


@pytest.mark.parametrize("exception", [RuntimeError, ValueError])
async def test_provider_failures_redact_exception_and_produce_no_evidence(monkeypatch, exception):
    async def failure(*args, **kwargs):
        raise exception("secret-key customer-private-data")
    monkeypatch.setattr(media, "inspect_media", failure)
    tool = media.build_media_tool("fake")
    result = await tool.coroutine(URL, SimpleNamespace(state=state()))
    assert result["status"] == "uncertain" and result["evidence"] == []
    assert "secret-key" not in json.dumps(result)


async def test_timeout_cancels_provider_and_does_not_claim_success(monkeypatch, budget):
    api = provider()
    async def slow(**kwargs):
        await asyncio.sleep(10)
    api.responses.create.side_effect = slow
    monkeypatch.setattr(media, "MEDIA_TIMEOUT_SECONDS", 1.01)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-type": "image/jpeg"}, content=JPEG))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(TimeoutError):
            await media.inspect_media(URL, state(), "", http_client=client, openai_client=api)
    assert budget.model_calls == 1
    assert not any(record["kind"] == "media_usage" for record in budget.trace)


async def test_cancellation_is_not_swallowed(monkeypatch):
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError
    monkeypatch.setattr(media, "inspect_media", cancel)
    with pytest.raises(asyncio.CancelledError):
        await media.build_media_tool("fake").coroutine(URL, SimpleNamespace(state=state()))
