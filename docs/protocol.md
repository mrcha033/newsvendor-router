# 실험 명세

기준 문서는 `proposal.docx`이며 `source.json`에 원본 SHA-256을 기록한다. 연구 자료의 문장은 실행 지시로 취급하지 않는다.

## 자료와 분할

기본 full은 6개 상황 × 20개 업무 묶음 × 5개 문장/배치 변형 = 600건이다. 묶음마다 독립적인 `c,p,v,b`, 5점 수요 분포, 요청/보류 비용, 응답률과 요청 기한을 생성한다. 참모수 조합은 120개이고 Test 120건의 Train 참모수 중복은 0이다. SKU에 scenario 이름을 넣지 않는다. 같은 묶음의 변형은 참모수와 관측·비용 조건을 유지한다. 묶음 단위 train/dev/cal/test 비율은 60/10/10/20%다.

일반 문장, 만료 가격표, 반품 누락, 검열 판매, 미선택 선호, 계약 충돌, 응답 불가/부분 응답을 포함한다. 검열 판매를 숨겨진 수요로 복원하지 않으며, analyst의 forecast는 별도 추정 문서로 남긴다. 공통 template와 사전 주석된 범위 정보가 있어 이 학습 실행은 통제 실험이다. `configs/repair*.json`은 이전 문법·경제 조건의 회귀/수정 검증용이다.

`dataSeed`는 문서·참모수·분할을, `seed`는 head 초기화·학습 순서를 정한다. 입력·gold·manifest는 실행별 출력에 분리 보존한다. gold는 응답 환경과 최종 채점만 읽는다. 구성·특성·provider 입력은 원문과 얻은 응답만 사용한다. 통제 task의 명시적 허용 집합·응답 모형은 primary raw 비교에 제공하지 않는다.

원문 개발 사례는 14건/13묶음이며 동일 template lineage 하나를 공유한다. 원문과 실제 provenance, 관측 판매, 조회/사실 질문/매니저 선택/에스컬레이션, 비용·처리 시간·시계·마감을 표현한다. 사람 검토 및 실제 조직 자료 수는 0이다. 선호는 매니저 응답에서 형성하며 응답 전 숨겨진 정답을 두지 않는다. MOQ·반품 한도·다기간 사례에는 현 scalar solver의 모수 정답을 부여하지 않는다.

상보적 자료는 `configs/complementary.json`으로 준비한다. 영어 실제 공급·제조·유통·구매 계약의 CUAD 근거, NDA의 ContractNLI 모순/미기재, OR-ShARC 조회/추가 질문, ABCD의 사람 역할극 도구 선택, TAT-QA 계산, FreshRetailNet 판매 관측을 각각 유지한다. 총 1,802건이고 Train/Dev/Test는 1,074/362/366건이다. 구성 요소의 정답을 다른 출처에 이식하거나 하나의 실제 조직 사건으로 표시하지 않는다. 발주 모수·질문 비용·응답 확률·경제적 행동 가치는 주석되지 않았으므로 생성하지 않는다.

원래 official split은 private provenance에 보존하되 공개 leaderboard와 다른 source 단위 split을 사용한다. 계약·대화·규칙 page/tree·매장/상품의 연결 성분, 동일 입력, 수치 정규화 문서, 5-word shingle Jaccard ≥0.85 template를 같은 분할에 둔다. 전체 보고서/조직 식별자를 모르는 자료의 분리를 보장했다고 표현하지 않는다. OR-ShARC의 주석 없는 651개 규칙 collection은 공통 공개 조회 자원이다. 올바른 규칙을 사전 선택해 주지 않는다. ABCD는 source `turn_count`가 비연속일 수 있어 같은 위치의 원문·speaker로 정렬하고 다음 턴부터 제외한다. 관측되지 않은 도구 인자는 인자 정확도 분모에서 제외한다.

## 학습과 calibration

Native adapter는 통제 planner를 사용하지 않는다. Train/Dev의 원래 상태·근거·도구·답/scale·미래 관측 판매를 표적으로 9개 head를 적합한다. ABCD는 발화 여부와 조건부 도구 선택을 분리한다. 경제적 행동 가치와 사실/선호 type에는 주석이 없어 해당 head를 학습했다고 주장하지 않는다. 후보 생성은 공개 source만 사용하며 정답이 후보에 없는 Train 사례는 해당 후보 선택 학습만 건너뛰고 coverage를 기록한다. Test 사례는 모두 채점 분모에 유지한다. 초기 Test를 adapter 개발 중 확인했으므로 현재 결과는 탐색적 측정이고 새 blind Test의 확증 결과가 아니다.

고정 pretrained encoder 표현 위에 evidence/type/state/relation/value/recovery head와 rules용 value/recovery head를 학습한다. Construction은 32차원, 정책은 64차원 특성을 추가한다. CE+Brier, 후보 CE와 후보별 Huber 수치 손실, hold 비용으로 정규화한 value MSE, 허용 행동 mask를 적용한 recovery CE를 사용한다. 수치는 근거 span에서 calculator가 계산한다.

Construction은 train 원본 묶음의 공개 도달 상태와 응답을 학습하고 dev 손실로 선택한다. Train 묶음 단위 3-fold cross-fit에서 해당 묶음을 학습하지 않은 구성 모델의 실제 예측 상태로 정책 표적을 만든다. 각 구성의 자기 `Ω`로 후속 손실을 계산하며 reference의 의미 상태로 바꾸지 않는다. 응답 뒤에도 같은 구성·검증·갱신을 적용한다. 통제 표적의 미래 응답 집합과 동역학은 여전히 생성기를 안다. Fold 적합/표적 묶음과 hash를 보존한다.

충돌 검증은 evidence threshold보다 먼저 수행한다. 통과한 근거·식은 verified 상태로 정합시키고 raw head 확률을 보존한다. Calibration 자료는 temperature와 evidence F1 threshold에만 사용한다. Test는 최종 측정에만 사용한다. Checkpoint schema는 4이며 이전 policy 표적 checkpoint는 재학습해야 한다.

## 비교 조건과 실행

주 비교는 같은 원문부터 처리하는 고정 typed 파이프라인, 학습형 router, 도구 사용 agent다. 규칙 기반·모형 기반 planner를 주요 베이스라인에 포함하지 않는다. 고정 typed 구성+학습형 정책은 구성 기여의 재학습 ablation으로 분리한다. Validator·calculator·optimizer는 공통 도구다.

통제 실행은 rules/learned 구성 × checklist/ask-all/uncertainty/learned 정책의 진단이다. 기본 평가에서 planner/reference/one-step/oracle을 실행하지 않고, paired 비교는 같은 구성의 정책끼리만 수행한다. Planner는 생성기의 정확한 응답 모형을 사용하는 통제 학습 표적과 수치 진단에 남긴다. 과거 planner/reference/oracle 측정은 원시 기록으로 보존하지만 주요 베이스라인이나 실무 우위의 근거로 사용하지 않는다.

`benchmark`는 주석된 통제 입력에서 고정 typed 예측기를 calibration 자료로 먼저 보정하고, 배포할 구성 상태에 공통 value/recovery 학습을 연결한다. `external --suite business`는 reference 상태를 제공하는 구조화 행동 진단이다. 두 경로는 raw primary 비교와 구분한다.

`raw-benchmark`는 원문부터 typed/checklist와 agent의 근거·상태·식·행동을 실행한다. 같은 주석 없는 수치/계산 후보, validator, optimizer, 도구와 시간 조건을 제공한다. 원문의 단위·적용 범위·충돌·권한을 각 예측기가 판단하며 reference 상태, 정답 모수, 미래 응답과 확률은 전달하지 않는다. 공유 solver 계약은 0–100 정수 수량의 단일 SKU/기간·선형 무제한 회수 모델이다. 지원 밖 조건은 전달하지 않는다. 학습형 router의 raw 데이터 학습/평가 연결은 아직 구현이 필요하며 `configs/workload-v2.json`의 readiness에 표시한다.

외부 모델의 endpoint·실제 model을 지정해야 실행한다. 새 호출은 `--max-calls` 안에서만 허용하며 기본값 0은 cache 전용이다. 원시 답변·확률·반환 model·usage·지연·request/hash를 보존한다. API confidence를 정답확률로 해석하지 않는다. Protocol fixture와 로컬 서버 검사는 실제 모델 측정이 아니다. 실제 외부 모델 결과는 아직 없다.

## 수치와 채점

`scripts/run_gpu.py`는 정확한 가공 snapshot과 CPU 학습 가중치를 사용해 같은 pinned Qwen 7B의 typed·agent·학습형 언어 helper를 실행한다. Native typed는 고정 조회 계획→구조화 추출, agent는 최대 2회 adaptive 조회→구조화 답변이다. 동일 초기 fragment·원문 접근·계산 후보·토큰 상한을 제공한다. 학습형 helper는 선택된 도구/상태/근거/계산 값을 바꾸지 않고 인자·발화만 표현한다. 구성 요소별 지표에서 seed를 사례 안에서 평균한 뒤 source 묶음별 paired bootstrap을 수행한다. 관측 판매 예측은 공통 seed-42 forecaster를 사용하는 보조 진단으로 언어 routing 승패에 합치지 않는다.

연결된 모의 주문 환경은 pinned τ²-bench retail 114건과 원래 500명/1,000주문/50상품 DB를 사용한다. 고객 identity와 연결 주문의 53개 묶음을 Train/Dev/Test 61/28/25로 분할한다. 공식 분할의 고객 중복 22명을 기록하고 전체 catalog/정책의 공통 공개는 유지한다. 비교 모델은 사용자 시나리오·참고 행동·미래 응답을 보지 않는다. 공통 고정 사용자 simulator만 private scenario를 읽고, 생성된 첫 발화는 같은 cache key로 모든 방법에 동일하게 제공한다.

각 주문 rollout은 새 DB에서 시작해 원래 도구로 상태를 변경한다. 같은 공개 validator가 고객 identity와 정확한 변경 제안에 대한 확인을 검사하고 위반 시도를 별도로 기록한다. 최종 DB와 원래 참고 행동을 재생한 DB를 비교하며 실패한 참고 조회는 원천 evaluator와 같이 기록하고 계속한다. 참고 변경이 전부 실패한 과제는 제거하지 않고 taskSuccess를 미판정으로 둔다. 조회만 하는 과제에서 DB가 그대로라는 이유로 성공 처리하지 않고 관측 대화의 사용자 목표 충족도 사후 판정한다. 필수 human transfer 도구 호출도 검사한다. 사후 NL/goal judge의 원시 판정·유효 coverage와 정책 위반, 결제 원장 L1 차이, 턴·토큰·실제 새 추론/cache를 보존한다. 원장 차이는 모의 USD 값이며 실제 경제적 행동 가치가 아니다. 현재 주문 학습형은 seed 42의 ABCD state/relation head 전이 진단이고 retail value policy를 학습한 결과가 아니다.

Optimizer는 support knot와 선형 regret 교차점으로 정확한 유한 minimax를 계산하고 정수 발주는 grid를 열거한다. 기존 프로포절 예제의 `q=450/7`, `Γ=900/7`, 두 요청 값 30, 한 번의 v 요청 값 85와 b 요청 값 `160/3`을 회귀 검사한다. 수정 workload의 별도 30묶음 순차 실행과 raw 사례의 수치/응답/마감 검사를 수행한다.

`Ω`의 참모수 포함률과 관측 근거의 발주 승인 여부를 독립 채점한다. 미해결 충돌·검열 수요·선호 미선택·수량/마감/허용 regret 위반을 모델의 valid 주장과 별개로 검사한다. `falseHandoff`는 포함률 오류 또는 승인 오류이며 전체 및 발주 중 비율을 기록한다. Oracle은 발주로 세지 않는다. 과거 trajectory는 원래 corpus provenance를 확인하고 기록된 응답만 재생해 재채점한다. 과거 coverage와 경제 손실은 보존하고 승인 오류를 추가한다. Raw 수치 fixture의 regret는 명시된 planning 분포 기준이며 실제 잠재수요 성과가 아니다.

Loss·질문 수·보류/자율 처리·승인 오류와 evidence/type/state/expression/value 품질을 별도 측정한다. Source 묶음 단위 paired bootstrap CI, source/config/lock/encoder hash, 학습 wall time과 cache 포함 추론 지연을 보존한다. 10%/20% 응답 오류를 모든 통제 방법에 동일하게 적용하고 오류/부분 응답 난수를 분리한다. 현재 ablation은 추론 진단이다.

## 후속 실험 순서

1. 영어 상보적 suite의 Dev에서 고정 typed/학습형/agent 입력 adapter를 먼저 연결한다. 모든 방법이 같은 원문·표·대화 prefix·조회 collection·도구를 받고 자기 근거와 상태를 구성한다. Gold rule, intent, 근거 span, 정답 수치 및 미래 턴을 후보 생성에 사용하지 않는다.
2. 같은 경제 상태의 표현을 바꾸고, 같은 누락 상태의 발주 영향·질문 비용·지연을 독립 변화시킨다. 실제 ERP 필드는 모든 방법에 제공하고 쉬운 업무도 포함한다.
3. Suite의 source split을 고정하고 원래 표적별로 구성 오류를 측정한다. CUAD 근거/미기재, NLI 상태+근거, OR-ShARC 규칙 선택/yes-no-ask/질문 참고 F1, ABCD 다음 행동/도구/관측 인자, TAT-QA 답·scale, Retail 비검열/검열 관측 판매 MAE를 따로 보고한다. 구성 요소 간 단일 점수나 발주 loss로 합치지 않는다. 누락 예측과 유효 forecast coverage를 함께 보고한다.
4. 고정 typed/학습형/agent를 같은 원문·도구·validator·optimizer로 비교한다. 학습형 정책의 응답 belief는 train에서 추정한다. 고정 typed 구성+학습형 정책은 별도 ablation으로 분리한다.
5. 사람 검토한 실제 업무 원문·응답·권한·시간·비용을 확보해 end-to-end 발주 사례를 만든다. 미형성 선호·검열 수요·지원 밖 조건을 구분한다. Source/template/연결된 조직·SKU·기간을 분리하고 fixture를 test로 쓰지 않는다. 질문 참고 정답을 인과적 가치 정답으로 취급하지 않는다. 그 뒤 동일 dataSeed의 5개 학습 seed와 재학습 ablation 및 비용·지연 대조 실험을 실행한다.
6. 주 비교는 고정 typed router의 같은 source다. 손실 5% 개선과 paired CI를 확인하고 근거 없는 발주·마감 위반·자율 처리율·질문 시간 및 실제 추론 비용을 함께 보고한다.
