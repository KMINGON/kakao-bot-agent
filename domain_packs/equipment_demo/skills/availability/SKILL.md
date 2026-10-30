---
name: availability
description: 장비 재고와 예약의 정정 이력을 확인하고 가용 수량을 계산한다.
---

장비 가용 수량 질문을 처리할 때 이 절차를 사용한다.
1. 장비명과 사용 날짜를 확인한다. 날짜가 없으면 묻는다.
2. check_availability로 해당 날짜의 최신 재고와 기존 예약을 조회한다.
3. available = corrected_stock - reserved 관계를 확인한다.
4. 최신 정정 출처와 예약 출처 ID를 답변에 포함한다.
5. 조회는 예약 실행이 아니므로 완료를 주장하지 않는다.
