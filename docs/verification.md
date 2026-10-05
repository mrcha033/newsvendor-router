# 검증 기록

2026-10-05, macOS arm64, Python 3.12.12에서 실행했다.

- 정적 검사·format 검사 통과, pytest **27개 통과**.
- 서로 다른 원본 업무 묶음 30개에서 수치·순차 요청 smoke 통과.
- full: 120개 원본 묶음/600개 episode, train/dev/cal/test = 360/60/60/120.
- 공개 pretrained encoder를 실제 내려받아 6개 PyTorch head를 학습했다. 3-fold 표적 생성의 적합/평가 source 겹침은 0이다.
- test 120개에 2,280개 trajectory를 실행했다. checkpoint를 불러온 재평가의 요약은 동일했다.
- TAT-QA 200건, ShARC 200건, FreshRetail 50개 시계열/4,500 daily row 준비 및 분리/hash/시간 검사 통과.
- 외부 API 형식, 호출 상한/cache, 공통 후보 decoder, 학습·정책 교차 연결을 로컬 서버와 명시된 test fixture로 검사했다. 실제 Jev/SGLang/agent 결과는 없다.

## 결과 해석

현재 full 통제 실행의 학습형 구성+정책 total loss는 89.994, 같은 구성의 reference planner는 87.729다. 학습형의 우위를 입증한 결과가 아니다. Pilot은 작은 학습 자료로 인해 false handoff가 발생했다. 자동 주석을 사람이 검토한 수는 0이며, 실제 업무 자료에 대한 효과는 별도로 검증해야 한다.

원시 trajectory와 checkpoint는 로컬 results에 보존하고 요약·학습 기록을 이 폴더에 저장했다. [full 보고서](evidence/full-report.md), [측정값](evidence/full-metrics.json), [학습 기록](evidence/full-training.json)을 함께 확인한다. 실행 명령은 README에 있다. GitHub Actions는 이후 push한 commit에서 검사한다.
