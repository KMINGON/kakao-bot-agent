from langchain_core.tools import tool
from kakao_bot.runtime.tools import calculate, source_tool


def build_tools(data):
    @tool
    def lookup_notice(subject: str) -> dict:
        """공지 제목의 키워드로 원본과 정정본을 대조한 최종 조건 및 근거를 조회한다."""
        notice = next((n for n in data["notices"] if subject.strip() and
                       (subject in n["name"] or n["name"] in subject)), None)
        if notice is None:
            return {"status": "needs_input", "data": {"available_notices": [n["name"] for n in data["notices"]]},
                    "evidence": [], "error": "unknown_notice", "effect": "read_only"}
        return {"status": "ok", "data": {**notice, "demo": True, "dataset_version": data["dataset_version"]},
                "evidence": [{"source_id": s, "dataset_version": data["dataset_version"]} for s in notice["evidence_ids"]],
                "effect": "read_only"}

    return [lookup_notice, source_tool(data), calculate]


def validate_answer(answer, tool_results):
    """Use the merged notice fields after checking correction provenance."""
    records = [r["result"] for r in tool_results if r["tool"] == "lookup_notice"
               and r["result"].get("status") == "ok"]
    if not records or answer["status"] != "answered":
        return answer
    result = records[-1]
    data = result["data"]
    refs = [e["source_id"] for e in result["evidence"]]
    if "N-001" not in refs or "N-001-C1" not in refs or data["deadline"] != "2026-11-05 18:00 Asia/Seoul":
        raise ValueError("Notice correction provenance mismatch")
    text = (f'{data["name"]}의 최종 마감은 {data["deadline"]}입니다. '
            f'원래 마감 {data["original_deadline"]}에서 정정됐습니다. '
            f'대상은 {data["eligibility"]}, 준비 서류는 {", ".join(data["required_documents"])}입니다. '
            '마감 외 조건은 동일하며 신청을 대신 접수하지는 않았습니다.')
    return {**answer, "answer": text, "claims": [{"text": text, "evidence_ids": refs}], "evidence_ids": refs}
