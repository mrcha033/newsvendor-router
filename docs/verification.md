# 검증 기록

2026-10-06, Linux x86_64, Python 3.12.14 / CPU PyTorch에서 수정 내용을 검증했다. 측정 원본은 Git에서 제외한 `results/`에 보존한다.

- 기존 회귀는 현재 실험에 필요한 검증 10개로 정리하고, 새 영어 workload의 통합검증 3개를 유지한다. PyTorch 자체 동작, 이전 구조화 진단 및 반복적인 내부 특성 검사는 제거했다. Ruff 정적/format 검사와 diff 검사 통과.
- 기존 수치 회귀와 수정 workload의 서로 다른 30개 source 순차 smoke 통과.
- 기본 full은 600건/120개 참모수 조합이다. Train/dev/cal/test = 360/60/60/120이며 Test의 Train 참모수 중복은 0이다. 최적 수량이 경계에 있는 사례와 scenario 이름을 포함한 SKU도 0건이다.
- 고정 MiniLM encoder와 8개 PyTorch head를 실제 학습했다. 3-fold의 적합/표적 source 겹침 및 train/dev/cal/test 묶음 겹침은 0이다.
- Test 120건에서 5,640개 경로를 평가했다. 학습형의 false handoff는 0건, hold 비율은 24.17%다. Checkpoint를 다시 불러오는 재평가는 동일 요약을 확인한다.
- 일반 문장 네 문서의 span/단위/계산 식, 원문 typed·agent의 공유 후보/자기 상태/호출 cache·상한, 계약 충돌·미선택 선호·검열 판매·마감·지원 밖 solver 조건을 protocol fixture로 검사했다.
- Raw 사례 14건/13묶음/공통 template 하나의 수치 fixture 6개 및 응답 11개를 검사했다. 실제 조직 자료와 사람 검토는 0이다.
- 상보적 영어 workload 1,802건을 실제 수집·변환했다. CUAD 450, ContractNLI 360, OR-ShARC 240, ABCD 462, TAT-QA 240, FreshRetail 50개 시계열이다. Train/Dev/Test는 1,074/362/366건이고 계약/page/tree/대화/매장/상품 연결, 입력·문서 template 검사에서 분할 중복은 0이다. 신규 사람이 검토한 사례와 실제 모델 실행 수는 0이다.
- 원래 공개 주석에서 CUAD 근거 있음/미기재 184/266건, NLI 함의/모순/미기재 164/34/162건, OR-ShARC 추가 질문 70건, ABCD 도구 선택 143건을 포함한다. 도구 인자 143건 중 입력에서 관측 가능한 126건만 인자 정확도로 채점한다. FreshRetail은 매장 50개/상품 43개이며 후속 350일 중 품절 관측 127일이다. 이를 잠재수요 정답으로 사용하지 않는다.
- ABCD의 source `turn_count`가 비연속인 실제 대화를 확인해 위치/speaker 기준으로 원문을 정렬했다. 새 테스트는 입력 누출, source 분할, 독립 채점의 통합검증 3개로 유지한다. 숫자 부호·scale, 잘못된 근거, 누락 예측·forecast coverage와 미래 정보 누출도 이 통합검증에서 확인한다. 실제 Test 366건은 정답 파일을 사용한 오프라인 채점기 검증을 통과했으며 모델 실행으로 세지 않는다. 원천의 실제 input·label·수집 hash와 준비 snapshot은 `cases/complementary/manifest.json`에 보존한다.

## 결과 해석

수정 full의 학습형 구성+정책 loss는 **178.223**, 같은 구성의 own-state planner는 **175.934**다. 차이는 2.289, 24개 source 묶음의 paired bootstrap 95% CI는 [-0.821, 7.013]이다. 학습형의 우위는 확인되지 않았다. 이 숫자는 새 경제·문장·응답 조건의 통제 실행이며 이전 조건의 loss와 직접 비교하지 않는다.

기존 main의 원시 기록을 기록된 응답으로 재채점하면 계약 충돌 20건의 미승인 전달을 모두 오류로 잡는다. 이전 coverage-only false handoff 0건은 승인 안전성의 결과가 아니다. 경제 손실 89.99355와 과거 coverage를 보존하며 승인 오류만 추가했다. 이전 문법의 수정 회귀 실행은 같은 충돌 20건의 승인 오류 0건, loss 39.33185다. 원본 corpus hash는 이전 저장본과 동일함을 확인했다.

현재 revision의 원시 학습/측정/재평가는 `results/revised/`, 이전 수정 회귀는 `results/repair/`, 사례 검사는 `results/workload-review/`에 보존한다. `docs/evidence/`의 27-test 당시 자료는 역사적 실행이며 새 승인 채점과 새 workload 결과로 읽지 않는다.

실제 Jev/SGLang/agent endpoint/model 설정이 없어 외부 모델 측정은 실행하지 않았다. 원문 typed/checklist와 agent 실행 경로는 구현됐지만 학습형 router의 raw 학습/평가 연결, 실제 조직 자료와 비교 결과는 후속 실험이 필요하다. Protocol fixture 결과를 실제 모델 성능으로 세지 않는다. 계획과 readiness는 기존 `protocol.md`와 `configs/workload-v2.json`에 유지한다.
