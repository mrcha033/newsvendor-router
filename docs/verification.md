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
- 2026-10-07: 주요 비교 계약에서 규칙·모형 기반 planner를 제거했다. 기존 13개 테스트, 수치 smoke, source 분할 검사가 통과했다. 기존 checkpoint의 Test 120건을 `results/baseline-controls/full/`에 별도로 재평가해 기본 실행에 planner/reference/one-step/oracle이 없고 paired 비교가 같은 구성의 정책 진단임을 확인했다. 원문 학습형 adapter와 새 suite의 실제 모델 비교는 여전히 미완료다.

## 결과 해석

2026-10-07, 구현 기준 `6d3acf3`: native-router-v3의 5개 seed × 12개 head를 CPU에서 실제 학습하고 Test 366건을 채점했다. 고정 MiniLM 위의 언어 head 8개와 수요 종류·모수 head 4개이며 학습 파라미터는 209,824개다. Train/Dev의 원래 주석을 사용하며 경제적 행동 가치와 사실/선호 type는 native 학습 표적으로 만들지 않았다. 초기 원시 결과는 `results/native-initial/`, 발화/도구 분리 실행은 `results/native-action-state/`, 현재 측정은 `results/native/`에 보존한다. 5개 seed의 가중치·예측·원시 측정·hash는 `models/native/`에도 포함한다. 수요 분포 예측의 종류·모수·F·period·orders는 저장 가중치 재생과 동일함을 확인했다.

7일 총수요 rolling 평가는 Train 810 / Dev 216 / Test 324기간이고 비품절 exact는 105 / 24 / 28기간이다. 5개 seed의 기간 평균 NLL 0.4093, CRPS 7.1025, 부족:잉여 3:1 실제 출력 발주 손실 17.4319, 80% 구간 포함률 21.4%를 기록했다. 모든 seed가 Test에서 로그정규를 선택했다. 분포·발주 형식의 유효성은 각 seed에서 100%이며, 품절 기간은 survival NLL과 손실 하한을 별도 기록한다. 이 결과는 현재 28차원 특성 기반 분포 head의 측정이다.

연결된 τ² retail 114건을 수집하고 고객/주문 묶음 53개를 61/28/25로 분할했다. 공식 분할의 고객 중복 22명과 참고 행동 경고 15개 과제를 보존한다. 잘못된 identity를 정정하는 실패 조회는 원래 evaluator와 같이 계속 재생한다. 참고 변경만 실패한 `orders-105`는 그대로 유지하고 주석 모호성을 표시한다. 전체 참고 DB 재생, source 분할, 원천 도구 함수의 AST 일치, 격리 DB, 미승인 변경 거절과 독립 최종 상태 채점을 검사했다. Typed·agent의 확인→격리 변경 및 공통 원문 조회 흐름도 protocol fixture로 검사했으며 실제 모델 성능으로 세지 않는다.

현재 테스트는 기존 13개와 종류·모수·F·실제 발주량의 일관성을 확인하는 통합검증 1개로 합계 14개다. 이번 문서 수정 후 기존 14개 테스트·Ruff/format·source 분할·30개 수치 smoke·snapshot/가중치 GPU preflight가 통과했다. DOCX 렌더링과 provenance hash를 확인했고 원본 OMML 수식 208개·drawing 13개를 보존했다. `scripts/run_gpu.py`에 남은 실행은 현재 Qwen 기반 typed/agent·기존 helper·주문 대화 진단이다. 이 runner가 다음 ModernBERT 모델을 구현하거나 학습하지는 않는다.

이번 문서 revision 10은 ModernBERT-base 149M / large 395M, token-level 공유 연구 head와 내부 도구 인자 추출, 256차원 2층 결합, 2층 GRU hidden 128 및 152–155M 설계 예산을 반영한다. Train 확대·masked 주석 학습, 종류별 cross-fit 예상 손실 점수와 1일 보조/7일 총수요 likelihood, 실제 rollout 행동 가치, base/large/value 제거 비교는 다음 구현·GPU 학습 계획이다. 현재 결과와 새 모델 결과를 구분하며 기존 `proposal.docx`·README·`protocol.md`·`sources.md`를 수정한다.

수정 full의 학습형 구성+정책 loss는 **178.223**, 같은 구성의 own-state planner는 **175.934**다. 차이는 2.289, 24개 source 묶음의 paired bootstrap 95% CI는 [-0.821, 7.013]이다. 이 과거 비교는 통제 rollout 표적의 근사도를 보는 진단이며, 주요 베이스라인과의 모델 우열을 측정하지 않는다. 2026-10-07부터 planner/reference/one-step/oracle은 기본 평가에서 제외하고 규칙 planner를 주요 비교 계약에서 제거했다. 이전 경제·문장·응답 조건의 loss와 직접 비교하지 않는다.

기존 main의 원시 기록을 기록된 응답으로 재채점하면 계약 충돌 20건의 미승인 전달을 모두 오류로 잡는다. 이전 coverage-only false handoff 0건은 승인 안전성의 결과가 아니다. 경제 손실 89.99355와 과거 coverage를 보존하며 승인 오류만 추가했다. 이전 문법의 수정 회귀 실행은 같은 충돌 20건의 승인 오류 0건, loss 39.33185다. 원본 corpus hash는 이전 저장본과 동일함을 확인했다.

현재 revision의 원시 학습/측정/재평가는 `results/revised/`, 이전 수정 회귀는 `results/repair/`, 사례 검사는 `results/workload-review/`에 보존한다. `docs/evidence/`의 27-test 당시 자료는 역사적 실행이며 새 승인 채점과 새 workload 결과로 읽지 않는다.

실제 Jev/SGLang/agent endpoint/model 설정이 없어 API 모델 측정은 실행하지 않았다. 영어 native head의 학습/평가와 같은 pinned Qwen을 쓰는 GPU 비교 경로는 구현했다. 실제 조직 발주 모수·권한·시간·비용을 연결한 end-to-end 경제적 가치 평가는 여전히 해당 자료가 필요하다. 공개 모의 주문의 전이 진단과 protocol fixture를 그 성능으로 세지 않는다. 계획과 readiness는 기존 `protocol.md`와 `configs/workload-v2.json`에 유지한다.
