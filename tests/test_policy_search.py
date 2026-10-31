import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PACK = Path("domain_packs/gyodong")


@pytest.fixture
def search_factory(tmp_path):
    spec = importlib.util.spec_from_file_location("gyodong_policy_search_test", PACK / "tools.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def build(sources=None):
        if sources is not None:
            (tmp_path / "data").mkdir(exist_ok=True)
            shutil.copyfile(PACK / "data/action-openapi.json", tmp_path / "data/action-openapi.json")
            for helper in ("media.py", "access.py", "order_context.py"):
                shutil.copyfile(PACK / helper, tmp_path / helper)
            (tmp_path / "data/policies.json").write_text(json.dumps({"sources": sources}))
            module.ROOT = tmp_path
        tool = next(tool for tool in module.build_tools({}) if tool.name == "search_policy")

        def search(query):
            command = tool.func(query, SimpleNamespace(tool_call_id="search-test"))
            return json.loads(command.update["messages"][0].content)

        return search

    return build


@pytest.mark.parametrize("query,first,required", [
    ("곰팡이 섭취", "D008", {"D008"}),
    ("수박 곰팡이 섭취", "D008", {"D008"}),
    ("곰팡이 부패 수박", "D008", {"D008"}),
    ("가격 인하 차액", "D003", {"D003", "D025"}),
    ("자연감량", "D027", {"D027"}),
    ("조합장 승인 소개", "D038", {"D038"}),
    ("직판 위탁 고객 창구", "D038", {"D038", "D030"}),
    ("기사 연락처 간선", "D038", {"D038"}),
    ("라이브 배송비 발언", "D039", {"D039", "D033", "D019"}),
    ("MANGO10 GAMJA10 마감 게시물", "D039", {"D039", "D025"}),
])
def test_general_policy_questions_retrieve_applicable_sources(search_factory, query, first, required):
    result = search_factory()(query)
    hits = result["data"]
    ids = [hit["source_id"] for hit in hits]
    assert ids[0] == first
    assert required <= set(ids)
    assert len(ids) == len(set(ids)) <= 3
    sources = {s["id"]: s for s in json.loads((PACK / "data/policies.json").read_text())["sources"]}
    for hit in hits:
        assert hit["scope"] == sources[hit["source_id"]]["scope"][:600]
        assert hit["scope_truncated"] is (len(sources[hit["source_id"]]["scope"]) > 600)
    assert {item["source_id"] for item in result["evidence"]} == set(ids)


def test_repeating_query_word_does_not_change_ranking(search_factory):
    search = search_factory()
    assert search("수박 수박 수박 곰팡이 섭취")["data"] == search("수박 곰팡이 섭취")["data"]


def test_common_product_repetition_cannot_displace_relevant_policy_and_scope_is_bounded(search_factory):
    search = search_factory([
        {"id": "CATALOG", "title": "상품표", "scope": "상품 목록", "text": "수박 " * 1000},
        {"id": "POLICY", "title": "하자 정책", "scope": "적용 조건 " * 150,
         "text": "곰팡이 발견 시 섭취 여부와 증상을 확인한다."},
    ])
    hits = search("수박 곰팡이 섭취")["data"]
    assert hits[0]["source_id"] == "POLICY"
    assert hits[0]["scope_truncated"] and len(hits[0]["scope"]) == 600
    assert all(len(hit["text"]) <= 1400 for hit in hits)


@pytest.mark.parametrize("query", [" ", "가" * 201])
def test_invalid_search_is_not_a_policy_read(search_factory, query):
    result = search_factory()(query)
    assert result["status"] == "needs_input" and result["evidence"] == []
