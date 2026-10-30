# 로컬 운영

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

Windows의 AC 절전은 운영 준비 과정에서 15분에서 사용 안 함으로 변경했다. 원복은 PowerShell에서 다음을 실행한다. DC 전원·업데이트·재부팅 설정은 변경하지 않았다.

```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 900
powercfg /setactive SCHEME_CURRENT
```

Docker Desktop의 로그인 시 자동 시작은 활성화돼 있다. Windows 로그인 전 무인 복구와 실제 재부팅은 검증하지 않았다.
