# 문제가 정해지면 붙이는 방법

기본 루프·저장·카카오 연결은 유지한다. 업무 기능은 팩의 도구로, 실행 흐름 조정은 LangChain의 `AgentMiddleware`로 추가한다. 별도의 플러그인 프레임워크는 없다.

| 필요한 변화 | 사용할 접점 | 준비 상태 |
|---|---|---|
| API·DB·파일 처리 등 새 기능 | `tools.py`의 `build_tools(data)` → `BaseTool` | 기존 제공 |
| 작업 방법·판단 순서 | `skills/<이름>/SKILL.md` | 기존 제공 |
| 항상 적용할 업무 지침 | `instructions/AGENTS.md` | 기존 제공 |
| 일반 MCP 서버 | `pack.yaml`의 stdio / Streamable HTTP | 기존 제공 |
| 다른 MCP 연결·인증 방식 | `adapters.py`의 `build_mcp_transport(spec, pack)` | 확장 접점 제공 |
| 문맥 선택·결과 변환·실행 전후 처리 | `adapters.py`의 `build_middleware(pack)` | SDK 훅 연결 제공 |
| 최종 업무 규칙 확인 | `tools.py`의 `validate_answer(answer, tool_results)` | 기존 제공 |

## 설정은 필요한 경우에만 추가한다

기존 팩은 수정하지 않아도 동작한다. 훅이나 커스텀 MCP 연결이 필요할 때 다음 한 줄을 팩에 추가한다.

```yaml
adapters_module: adapters.py
```

`adapters.py`의 두 팩토리는 모두 선택 사항이며 일반 동기 함수다. 팩토리가 반환하는 미들웨어의 실제 처리는 async 훅으로 작성할 수 있다. 팩 파일을 수정한 뒤에는 `lock_pack`으로 해시를 갱신한다.

```python
from langchain.agents.middleware import AgentMiddleware

class TopicHooks(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        # 필요한 도구의 결과만 공통 응답 형태로 변환한다.
        return result

def build_middleware(pack):
    return [TopicHooks()]
```

이 예제는 연결 형태만 보여 준다. 실제 변환 규칙은 문제를 받은 뒤 해당 훅에 작성한다. 도구는 `build_tools`와 `allowed_tools`에 등록하며, 미들웨어의 `tools` 속성으로 우회 등록하지 않는다.

## 근거와 컨텍스트도 기존 접점으로 처리한다

```mermaid
flowchart LR
    X[문제에 필요한 외부 기능] --> T[Python 도구 또는 MCP]
    T --> H[필요할 때 도구 결과 변환 훅]
    H --> E[데이터 + 등록된 출처 ID]
    E --> C[필요한 문맥 선택]
    C --> A[기존 에이전트 루프]
    A --> V[기존 근거 검사 + 업무 검증 훅]
```

근거로 쓸 도구 결과는 다음 형태의 **JSON object 문자열**로 모델에 전달하면 기존 검증과 연결된다. 도구는 dict를 반환할 수 있다. MCP가 자체 텍스트 블록이나 다른 구조를 반환하면 도구 래퍼 또는 `awrap_tool_call`에서 변환한다.

```json
{
  "status": "ok",
  "data": {"value": 7},
  "evidence": [{"source_id": "INVENTORY_API", "locator": "item:chair"}],
  "effect": "read_only"
}
```

`INVENTORY_API` 같은 출처 ID는 팩의 `data/records.json` 내 `sources`에 등록한다. 데이터가 갱신되는 API는 출처 ID와 별개로 조회 시각·레코드 위치·버전을 결과에 남긴다. 현재 공통 검사는 출처 ID의 등록·조회 여부를 확인한다. 위치와 시각의 업무상 유효성은 해당 도구와 검증 훅에서 확인한다.

컨텍스트를 조정할 때는 아래 중 필요한 곳 하나를 먼저 선택한다. [SDK 훅 계약](https://docs.langchain.com/oss/python/langchain/middleware/custom)을 그대로 따른다.

| 상황 | 구현 위치 |
|---|---|
| 모델에 보낼 문맥만 잠시 선택·추가 | `wrap_model_call` / `awrap_model_call` |
| 다음 턴에도 보존할 대화 상태 갱신 | `before_model` / `after_model`의 state 업데이트 |
| 외부 결과 정리·형식 통일 | `wrap_tool_call` / `awrap_tool_call` |
| 실행 전체의 시작·종료 처리 | `before_agent` / `after_agent` |

요청별 값은 SDK state 또는 `runtime.context`를 사용한다. 미들웨어 인스턴스의 변경 가능한 속성에 사용자별 기억을 쌓지 않는다. 지침·업무 원자료·대화 요약의 역할은 유지하고, 자료에 포함된 문장을 시스템 지침으로 승격하지 않는다.

## MCP 기본 연결로 부족할 때

대부분은 기존 stdio 또는 HTTP 설정으로 연결한다. 그 외에는 팩의 팩토리에서 [FastMCP `ClientTransport`](https://gofastmcp.com/clients/transports)를 반환한다.

```yaml
mcp:
  - transport: custom
    url: https://example.com/mcp
    allow_tools: [lookup]
```

```python
from fastmcp.client.transports import StreamableHttpTransport

def build_mcp_transport(spec, pack):
    return StreamableHttpTransport(spec["url"])
```

위 코드는 접점 예시다. 일반 HTTP라면 기존 `streamable_http` 설정이면 충분하다. 특수 인증이나 다른 transport가 필요할 때 팩토리 내부만 바꾼다. 실행 파일·자격증명·추가 패키지는 해당 연결을 선택할 때 준비한다. 접점이 있다고 모든 서버의 인증까지 자동 해결되는 것은 아니다.

## 문제를 받은 뒤 결정할 것

| 문제 특성 | 먼저 할 일 | 예외가 생기면 |
|---|---|---|
| 작은 정적 자료 | 검토한 데이터를 팩에 넣고 조회 도구 연결 | 원문·단위·정정 관계가 애매하면 확인 질문 |
| 실시간 API·DB | 조회 도구나 MCP 연결, 시각·출처 포함 | 실패를 명시하고 오래된 값을 새 값으로 답하지 않음 |
| 문서·이미지 등 별도 해석 필요 | 적합한 처리기 하나를 도구/MCP로 감싸기 | 불확실한 추출 결과는 검토 대상으로 표시 |
| 컨텍스트보다 큰 자료 | 검색 도구로 필요한 구간만 반환 | 범위를 좁히고 원본 위치를 유지 |
| 오래 걸리는 처리 | 사전 처리 또는 별도 작업으로 분리 | 현재 도구 15초·전체 180초·카카오 기한 확인 |
| 예약·결제·수정 같은 쓰기 | 실행 조건·중복 방지·성공 확인을 먼저 설계 | 결과 불명 시 재실행하지 않고 상태 조회 |

이 경로들은 연결 방식에 대한 준비다. 실제 품질과 예외 처리는 선택한 데이터·서비스로 확인해야 한다. 특정 처리기, 검색 저장소, 쓰기 워크플로는 미리 설치하지 않았다.

기본 자료 계약은 도구가 반환하는 텍스트·JSON이다. 이미지·음성을 모델에 직접 넣는 경로를 선택하면 해당 모델의 입력 형식과 현재 토큰 예산 검사도 함께 확인한다.

기본 시스템 지침은 조회 전용이다. 쓰기 기능이 필요한 문제라면 해당 도구뿐 아니라 지침과 실행 조건도 함께 변경한다.

## 바꾼 팩을 확인하는 순서

1. 도구의 정상·오류 결과를 확인한다.
2. 실제 SDK 그래프에 팩을 넣어 도구·훅·근거가 이어지는지 확인한다.
3. 문맥 유지·다른 사용자 격리·시간 제한을 확인한다.
4. 필요한 실제 모델 질문만 실행하고 API와 워커를 함께 반영한다.

재사용할 검사 예제는 `tests/test_adapters.py`, `tests/test_runtime_mcp.py`, `tests/test_runtime.py`에 있다. 전체 회귀 검사는 `./ops/test.sh -q`다. 어댑터 Python 코드는 검토한 애플리케이션 코드이며 별도의 실행 샌드박스는 아니다.
