# 로컬 운영

2026-11-01에 대회용 서비스를 종료했다. 컨테이너·프로젝트 이미지·네트워크와 Cloudflare 터널·DNS, 카카오 웹훅을 제거했다. DB 볼륨은 보존했으며 복구용 dump·로그·정리 근거는 `.local/cleanup/20261101/`에 있다. 로컬 `.env`는 `BOT_MODE=echo`, `ACTION_API_WRITES_ENABLED=false`로 바꿨다. 기존 터널 토큰은 보관 폴더로 옮겼으므로, 다시 운영할 때 새 터널 토큰과 DNS·웹훅 등록이 필요하다.

프로젝트 루트에서 실행한다. `.local/secret/`의 키는 이미지나 Git에 포함되지 않는다. 공개 URL은 `https://kakao.mingon.dev/webhooks/kakao`, Tunnel 내부 대상은 `http://api:8000`이다. 공유기 포트 포워딩은 사용하지 않는다.

```bash
# .env의 BOT_MODE=echo로 연결 검증 후 agent로 바꾼다.
docker compose build api
docker compose --profile tunnel up -d

# 로컬·공개 HTTPS 수신 경로 검사. 모델과 카카오 답장을 호출하지 않는다.
python3 ops/smoke_http.py
python3 ops/smoke_http.py --base-url https://kakao.mingon.dev

# 격리된 PostgreSQL로 일반 테스트. 종료 시 테스트 DB만 제거한다.
./ops/test.sh

# 최종 배포본에서 관측 시작. 모델·카카오 메시지는 호출하지 않는다.
docker compose --profile monitor up -d monitor
python3 ops/report_monitor.py

docker compose ps
docker compose logs --tail 100 api worker cloudflared
```

PostgreSQL은 `kakao-bot-agent_postgres_data` 볼륨을 사용한다. 운영 데이터를 유지하려면 `docker compose down --volumes`를 사용하지 않는다. `docker compose restart api worker`는 저장된 대화와 작업을 지우지 않는다.

모니터 기록은 `.local/implementation/v1/evidence/monitor/`에 남는다. `run.json`의 시작·종료 시각을 재시작 후에도 유지하며, 새 관측을 시작하려면 기존 디렉터리를 다른 이름으로 보관한 뒤 빈 디렉터리를 만든다. 실제 관측 시간과 실패·누락을 보고하며, 이 프로세스를 설치했다고 48시간 검증이 완료되는 것은 아니다. 모니터도 같은 PC에 있으므로 PC가 꺼진 구간은 복구 후 기록 간격으로만 확인할 수 있다.

Windows의 AC 절전은 운영 준비 과정에서 15분에서 사용 안 함으로 변경했고, 2026-11-01 서버 정리 때 15분으로 복구했다. 아래는 복구 명령이다. DC 전원·업데이트·재부팅 설정은 변경하지 않았다.

```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 900
powercfg /setactive SCHEME_CURRENT
```

Docker Desktop의 로그인 시 자동 시작은 활성화돼 있다. Windows 로그인 전 무인 복구와 실제 재부팅은 검증하지 않았다.
