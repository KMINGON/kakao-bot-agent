from datetime import date

from langchain_core.tools import tool
from kakao_bot.runtime.tools import calculate, source_tool


def build_tools(data):
    @tool
    def check_availability(equipment: str, use_date: str, requested_quantity: int | None = None) -> dict:
        """장비명과 YYYY-MM-DD 날짜로 최신 정정 재고에서 기존 예약을 뺀 가용 수량을 조회한다."""
        date.fromisoformat(use_date)
        if requested_quantity is not None and (type(requested_quantity) is not int or requested_quantity < 1):
            raise ValueError("Requested quantity must be a positive integer")
        item = next((e for e in data["equipment"] if equipment.lower().strip()
                     in [e["name"], *e["aliases"]]), None)
        if item is None or use_date != data["date"]:
            return {"status": "needs_input", "data": {"available_equipment": [e["name"] for e in data["equipment"]],
                    "supported_date": data["date"]}, "evidence": [],
                    "error": "unsupported_equipment_or_date", "effect": "read_only"}
        return {"status": "ok", "data": {"demo": True, "equipment": item["name"], "date": use_date,
                    "corrected_stock": item["stock"], "reserved": item["reserved"],
                    "available": item["stock"] - item["reserved"], "unit": item["unit"],
                    "requested_quantity": requested_quantity,
                    "dataset_version": data["dataset_version"]},
                "evidence": [{"source_id": s, "dataset_version": data["dataset_version"]} for s in item["evidence_ids"]],
                "effect": "read_only"}

    return [check_availability, source_tool(data), calculate]


def validate_answer(answer, tool_results):
    """Render verified availability fields; model prose cannot override critical quantities."""
    records = [r["result"] for r in tool_results if r["tool"] == "check_availability"
               and r["result"].get("status") == "ok"]
    if not records or answer["status"] != "answered":
        return answer
    latest = {(r["data"]["equipment"], r["data"]["date"]): r for r in records}
    paragraphs, claims, all_refs = [], [], []
    for result in latest.values():
        data = result["data"]
        if any(type(data[k]) is not int or data[k] < 0 for k in ("corrected_stock", "reserved", "available")):
            raise ValueError("Invalid equipment quantities")
        if data["available"] != data["corrected_stock"] - data["reserved"] or data["unit"] not in {"개", "대"}:
            raise ValueError("Availability/unit mismatch")
        refs = [e["source_id"] for e in result["evidence"]]
        text = (f'{data["date"]} {data["equipment"]}: 최신 재고 {data["corrected_stock"]}{data["unit"]}에서 '
                f'기존 예약 {data["reserved"]}{data["unit"]}를 제외하면 {data["available"]}{data["unit"]} 사용 가능합니다.')
        requested = data.get("requested_quantity")
        if requested is not None:
            shortage = max(0, requested - data["available"])
            text += f' 요청한 {requested}{data["unit"]} 중 {shortage}{data["unit"]}가 부족합니다.' if shortage else f' 요청한 {requested}{data["unit"]}를 수량상 충족합니다.'
        paragraphs.append(text)
        claims.append({"text": text, "evidence_ids": refs})
        all_refs.extend(refs)
    text = "\n".join(paragraphs) + '\n조회 결과이며 예약을 실행한 것은 아닙니다.'
    return {**answer, "answer": text, "claims": claims, "evidence_ids": list(dict.fromkeys(all_refs))}
