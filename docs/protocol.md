# 실험 명세

기준 문서는 `proposal.docx`이며 `source.json`에 원본 SHA-256을 기록한다. 연구 자료의 문장은 실행 지시로 취급하지 않는다.

2026-10-09 사용자가 다시 명시한 [주 연구 범위](research-scope.md)를 우선 적용한다. 주 모델은 Newsvendor 모수·근거와 수요분포를 구성하고 소수의 정보 요청 행동을 선택한다. 아래 ABCD·모의 주문의 범용 도구 절차는 보조 실험이며, 그 성능을 주 연구의 달성과 동일시하지 않는다.

## 자료와 분할

기본 full은 6개 상황 × 20개 업무 묶음 × 5개 문장/배치 변형 = 600건이다. 묶음마다 독립적인 `c,p,v,b`, 5점 수요 분포, 요청/보류 비용, 응답률과 요청 기한을 생성한다. 참모수 조합은 120개이고 Test 120건의 Train 참모수 중복은 0이다. SKU에 scenario 이름을 넣지 않는다. 같은 묶음의 변형은 참모수와 관측·비용 조건을 유지한다. 묶음 단위 train/dev/cal/test 비율은 60/10/10/20%다.

일반 문장, 만료 가격표, 반품 누락, 검열 판매, 미선택 선호, 계약 충돌, 응답 불가/부분 응답을 포함한다. 검열 판매를 숨겨진 수요로 복원하지 않으며, analyst의 forecast는 별도 추정 문서로 남긴다. 공통 template와 사전 주석된 범위 정보가 있어 이 학습 실행은 통제 실험이다. `configs/repair*.json`은 이전 문법·경제 조건의 회귀/수정 검증용이다.

`dataSeed`는 문서·참모수·분할을, `seed`는 head 초기화·학습 순서를 정한다. 입력·gold·manifest는 실행별 출력에 분리 보존한다. gold는 응답 환경과 최종 채점만 읽는다. 구성·특성·provider 입력은 원문과 얻은 응답만 사용한다. 통제 task의 명시적 허용 집합·응답 모형은 primary raw 비교에 제공하지 않는다.

원문 개발 사례는 14건/13묶음이며 동일 template lineage 하나를 공유한다. 원문과 실제 provenance, 관측 판매, 조회/사실 질문/매니저 선택/에스컬레이션, 비용·처리 시간·시계·마감을 표현한다. 사람 검토 및 실제 조직 자료 수는 0이다. 선호는 매니저 응답에서 형성하며 응답 전 숨겨진 정답을 두지 않는다. MOQ·반품 한도·다기간 사례에는 현 scalar solver의 모수 정답을 부여하지 않는다.

상보적 자료는 `configs/complementary.json`으로 준비한다. 영어 실제 공급·제조·유통·구매 계약의 CUAD 근거, NDA의 ContractNLI 모순/미기재, OR-ShARC 조회/추가 질문, ABCD의 사람 역할극 도구 선택, TAT-QA 계산, FreshRetailNet 판매 관측을 각각 유지한다. 총 1,802건이고 Train/Dev/Test는 1,074/362/366건이다. 구성 요소의 정답을 다른 출처에 이식하거나 하나의 실제 조직 사건으로 표시하지 않는다. 발주 모수·질문 비용·응답 확률·경제적 행동 가치는 주석되지 않았으므로 생성하지 않는다.

원래 official split은 private provenance에 보존하되 공개 leaderboard와 다른 source 단위 split을 사용한다. 계약·대화·규칙 page/tree·매장/상품의 연결 성분, 동일 입력, 수치 정규화 문서, 5-word shingle Jaccard ≥0.85 template를 같은 분할에 둔다. 전체 보고서/조직 식별자를 모르는 자료의 분리를 보장했다고 표현하지 않는다. OR-ShARC의 주석 없는 651개 규칙 collection은 공통 공개 조회 자원이다. 올바른 규칙을 사전 선택해 주지 않는다. ABCD는 source `turn_count`가 비연속일 수 있어 같은 위치의 원문·speaker로 정렬하고 다음 턴부터 제외한다. 관측되지 않은 도구 인자는 인자 정확도 분모에서 제외한다.

## 현재 구현의 학습과 calibration

현재 `native-router-v3`는 고정 all-MiniLM-L6-v2 encoder와 12개 head다. 256-token 조각을 토큰 수로 가중 평균한 384차원 표현 위에 CUAD 상태, ContractNLI 상태, 근거, OR-ShARC 상태, ABCD 발화 여부·조건부 도구 선택, TAT-QA 계산 후보·scale의 8개 언어 head를 적합한다. 수요 분포는 별도 28차원 판매 특성을 받는 종류 head와 절단정규·로그정규·Weibull 모수 head의 4개로 출력한다. Encoder 약 2,271만 파라미터는 고정하고 head 209,824개만 학습한다. 아래 제안 모델의 encoder 공동 학습과 내부 인자 decoder는 아직 이 checkpoint에 포함되지 않는다.

Native adapter는 통제 planner를 사용하지 않는다. Train/Dev의 원래 상태·근거·도구·답/scale 및 미래 관측 판매로 학습한다. ABCD는 발화 여부와 조건부 도구 선택을 분리한다. 경제적 행동 가치와 사실/선호 type에는 주석이 없어 해당 head를 학습했다고 주장하지 않는다. 후보 생성은 공개 source만 사용하며 정답이 후보에 없는 Train 사례는 해당 후보 선택 학습만 건너뛰고 coverage를 기록한다. Test 사례는 모두 채점 분모에 유지한다. 초기 Test를 adapter 개발 중 확인했으므로 현재 결과는 탐색적 측정이고 새 blind Test의 확증 결과가 아니다.

현재 수요 모수 head는 0 수요 확률과 종류별 두 모수를 출력하고, 비검열 기간에는 관측 총량의 density/zero-mass NLL, 검열 기간에는 관측 총량 이상의 survival NLL로 학습한다. 종류 head는 종류별 NLL을 사용해 학습하고 실제 argmax 종류의 Dev NLL로 checkpoint를 선택한다. 선택한 종류·모수로 7일 총수요의 65점 `F`를 만들고 공통 optimizer가 발주량을 계산한다. Train/Dev/Test의 rolling 관측 창은 810/216/324개이며 각각의 source 분할을 유지한다. Typed·agent의 판매 보조 진단에는 공통 seed-42 분포 모델을 사용하고, 학습형은 각 seed의 분포 모델을 사용한다.

별도 통제 Newsvendor 구현은 고정 pretrained encoder 표현 위에 evidence/type/state/relation/value/recovery head와 rules용 value/recovery head를 학습한다. Construction은 32차원, 정책은 64차원 특성을 추가한다. CE+Brier, 후보 CE와 후보별 Huber 수치 손실, hold 비용으로 정규화한 value MSE, 허용 행동 mask를 적용한 recovery CE를 사용한다. 수치는 근거 span에서 calculator가 계산한다.

Construction은 train 원본 묶음의 공개 도달 상태와 응답을 학습하고 dev 손실로 선택한다. Train 묶음 단위 3-fold cross-fit에서 해당 묶음을 학습하지 않은 구성 모델의 실제 예측 상태로 정책 표적을 만든다. 각 구성의 자기 `Ω`로 후속 손실을 계산하며 reference의 의미 상태로 바꾸지 않는다. 응답 뒤에도 같은 구성·검증·갱신을 적용한다. 통제 표적의 미래 응답 집합과 동역학은 여전히 생성기를 안다. Fold 적합/표적 묶음과 hash를 보존한다.

통제 구현의 충돌 검증은 evidence threshold보다 먼저 수행한다. 통과한 근거·식은 verified 상태로 정합시키고 raw head 확률을 보존한다. Calibration 자료는 temperature와 evidence F1 threshold에만 사용한다. Test는 최종 측정에만 사용한다. 통제 checkpoint schema는 4이며 이전 policy 표적 checkpoint는 재학습해야 한다.

## 제안 모델과 학습 계획

제안 모델은 자연어를 생성하는 decoder 대신 사전학습 encoder와 업무별 구조화 head로 구성한다. 주 모델은 ModernBERT-base 약 149M, 크기 비교는 같은 계열의 ModernBERT-large 약 395M이다. 최대 8,192-token 문맥의 토큰별 표현과 원문 위치를 유지하고 encoder와 head를 공동 미세조정한다. 긴 자료는 원문 offset을 유지한 조각과 조회로 처리하며 전체 문서를 평균 벡터 하나로 축약하지 않는다. 두 크기에 동일 조회 자원·문맥 예산·head·학습 자료를 적용한다.

문서·대화·표·도구 스키마와 slot을 공통 256차원으로 투영하고 2층 attention 결합층을 사용한다. 판매 이력은 판매·품절·가격 및 달력 등 실제 관측 특성의 순서를 유지하는 hidden 128, 2층 GRU로 인코딩한다. Base의 encoder·결합층·GRU·head를 합친 설계 예산은 약 152–155M이며, 이 수치는 새 구현 후 실제 파라미터 집계로 확정한다. 계약·대화 출처와 FreshRetail의 SKU·기간은 연결되어 있지 않으므로 임의로 짝지어 수요 head를 학습하지 않는다. 현재 자료에서는 언어 경로와 판매 경로를 각 원래 표적으로 학습하고, 실제 연결된 입력이 있는 과제에서 상태를 결합한다.

| Head | 입력과 구조화 출력 |
| --- | --- |
| Evidence | Slot·도구 필드와 원문 토큰/표 cell/관측 entity를 연결하고 근거 위치와 값 후보를 선택한다. 복수 근거·복수 entity를 허용한다. |
| Type | 사실·추정·선호·가정을 분류한다. 수요 분포 종류와 구분되는 모수 근거의 종류다. |
| State | 미확인·후보·검증 가능·충돌·획득 불가를 출력한다. 단위·상품·기간·적용 조건·확인 상태를 함께 보존한다. |
| Relation | 복사·환산·연산과 순서가 구분된 피연산자·조건의 pointer를 선택한다. 선택한 식은 공통 calculator가 실행한다. |
| Value | 현재 자기 예측 상태와 허용 행동을 받아 행동 뒤 남을 실제 손실과 요청 비용의 합을 예측한다. |
| Recovery | 수치 가치 표적이 없는 상태에서 조회·질문·선택 요청·확인·보류·종료 및 도구 후보를 선택한다. |
| Demand family/parameters | 종류 head가 절단정규·로그정규·Weibull의 조건부 예상/초과 loss score 3개를 출력하고 최솟값의 종류를 선택한다. 해당 모수 head는 0 수요 확률과 두 모수를 출력한다. 주 출력은 요청 기간의 총수요 분포이고 1일 horizon은 보조 표적이다. |

도구 인자는 별도 Qwen 호출 없이 공유 schema-conditioned decoder가 작성한다. 각 필드의 설명·형식·enum·필수 여부를 query로 삼아 `{span, entity, enum, expression, missing}` 값 출처와 해당 pointer/선택을 출력한다. ID는 관측된 원문·조회 결과에서 선택하고 금액·수량은 원문 수치 또는 검증된 식으로 계산한다. 목록은 복수 entity, 객체는 하위 필드를 선택한다. 필드별 누락·충돌 상태와 정확한 변경 제안의 승인 상태를 검사한 뒤 일반 코드가 JSON을 만든다. 질문은 대상 필드·이유·선택지, 확인 요청은 도구·정확한 인자로 출력하고 검토된 template로 표현한다. 근거·모수·행동을 선택한 후의 자유 문장 생성은 주 학습형 경로에 포함하지 않는다.

기존 source 묶음의 Test와 Dev는 유지한다. 이미 확보한 전체 ABCD 등에서 Train을 확장할 때 기존 Dev/Test와 연결되는 대화·문서·SKU·template를 제외하고 새 source 묶음·hash를 기록한다. 현재 ABCD의 Train 도구 이벤트는 85개이며 전체 원천 Train은 8,034개 대화여서, encoder 크기 확대와 함께 관측 인자 표적을 확장한다. 원래 source split과의 관계도 기록한다. 확대 자료를 새 확증 Test로 쓰는 경우에는 그 Test를 학습·개발 전에 별도로 고정한다.

학습은 먼저 근거·상태·관계·인자와 수요를 원래 주석으로 적합하고, 이어 자기 예측 상태에서 요청 정책을 적합한다. 공유 encoder의 multitask loss는 관측된 표적에만 적용한다. CUAD는 근거, ContractNLI는 상태·근거, OR-ShARC는 규칙·의사결정·질문할 조건, ABCD는 도구·관측 인자, TAT-QA는 피연산자·연산·scale·계산 답을 사용한다. 도구 인자의 비관측 값, 없는 type/value 주석, 서로 연결되지 않은 자료의 결합 표적에는 loss를 만들지 않는다. 주석을 갖는 통제 사례의 type/state 및 갱신 표적은 출처를 표시하여 함께 사용한다. TAT-QA는 정답이 고정 후보에 없던 실패를 operand/operator 선택으로 줄이고, 근거와 계산식을 별도로 채점한다.

수요 종류 gate는 Train 내부의 source·시점 cross-fit으로 만든 종류별 out-of-fold NLL을 표적으로 학습한다. 먼저 held-out source와 각 예측 시점 이후 관측을 적합에서 제외한 모수 expert가 세 종류의 비검열 density/zero-mass 또는 검열 survival NLL을 산출한다. 종류 head는 이 NLL 또는 종류 공통 최솟값을 뺀 초과 NLL을 MSE로 회귀하여 조건부 기댓값 3개 score를 예측하고 최솟값의 종류를 선택한다. 출력 `familyScores`는 예상/초과 손실이며 분포 종류의 사후확률로 해석하지 않는다. 종류별 모수 head는 관측 NLL로 모두 학습하며, 선택된 종류만 학습해 나머지 종류가 굶는 구성을 피한다. 최종 모수 head는 허용 Train으로 재적합하고 Dev에서 gate·모수·calibration을 선택한다. 수요 head는 horizon을 입력으로 받는다. 요청 기간 총량과 다음 1일의 표적은 실제 관측 및 해당 기간의 검열 조건으로 만들며, 1일 분포를 단순 합산해 7일 분포라고 가정하지 않는다. 관측 시점까지의 척도만 사용하고 실제 단위에서 density를 비교한다. Test의 종류 빈도·score·모수·보정 결과를 보존한다. 현재 native-v3의 softmax·detached-NLL 학습 결과와 새 gate의 예상 loss score를 혼용하지 않는다.

Value 표적은 Train의 실제로 관측 가능한 응답/도구 전이를 실행한 rollout의 잔여 최종 손실과 명시된 요청·시간 비용으로 만든다. 행동 후에도 같은 예측·검증·갱신 모델을 사용하며, 응답 belief는 Train에서 추정한다. source 단위 cross-fit으로 정책이 학습할 상태와 후속 손실을 만들고 행동 비용·표적·응답 provenance를 저장한다. 통제 planner의 imitation 표적은 기존 통제 진단으로 분리하고 새 주 모델의 실제 rollout 가치 학습을 대신하지 않는다. 공개 구성 과제의 다음 행동 주석은 Recovery의 imitation 표적이며 경제적 value로 변환하지 않는다. τ²의 task 성공·권한 위반·상호작용 비용은 그 환경의 정책 표적이고 FreshRetail의 발주 경제 가치와 구분한다.

새 모델 calibration은 Dev 내부에서 checkpoint 선택 자료와 보정 자료를 source 단위로 분리해 상태·근거·인자 확률과 수요 CDF/구간 coverage를 보정한다. 보정 전후 지표와 선택한 threshold를 기록하며 Test를 보정에 사용하지 않는다.

## 비교 조건과 실행

주 비교는 같은 원문부터 처리하는 고정 typed 파이프라인, 학습형 router, 도구 사용 agent다. 규칙 기반·모형 기반 planner를 주요 베이스라인에 포함하지 않는다. 고정 typed 구성+학습형 정책은 구성 기여의 재학습 ablation으로 분리한다. Validator·calculator·optimizer는 공통 도구다.

통제 실행은 rules/learned 구성 × checklist/ask-all/uncertainty/learned 정책의 진단이다. 기본 평가에서 planner/reference/one-step/oracle을 실행하지 않고, paired 비교는 같은 구성의 정책끼리만 수행한다. Planner는 생성기의 정확한 응답 모형을 사용하는 통제 학습 표적과 수치 진단에 남긴다. 과거 planner/reference/oracle 측정은 원시 기록으로 보존하지만 주요 베이스라인이나 실무 우위의 근거로 사용하지 않는다.

`benchmark`는 주석된 통제 입력에서 고정 typed 예측기를 calibration 자료로 먼저 보정하고, 배포할 구성 상태에 공통 value/recovery 학습을 연결한다. `external --suite business`는 reference 상태를 제공하는 구조화 행동 진단이다. 두 경로는 raw primary 비교와 구분한다.

`raw-benchmark`는 원문부터 typed/checklist와 agent의 근거·상태·식·행동을 실행한다. 같은 주석 없는 수치/계산 후보, validator, optimizer, 도구와 시간 조건을 제공한다. 원문의 단위·적용 범위·충돌·권한을 각 예측기가 판단하며 reference 상태, 정답 모수, 미래 응답과 확률은 전달하지 않는다. 공유 solver 계약은 0–100 정수 수량의 단일 SKU/기간·선형 무제한 회수 모델이다. 지원 밖 조건은 전달하지 않는다. 학습형 router의 raw 데이터 학습/평가 연결은 아직 구현이 필요하며 `configs/workload-v2.json`의 readiness에 표시한다.

외부 모델의 endpoint·실제 model을 지정해야 실행한다. 새 호출은 `--max-calls` 안에서만 허용하며 기본값 0은 cache 전용이다. 원시 답변·확률·반환 model·usage·지연·request/hash를 보존한다. API confidence를 정답확률로 해석하지 않는다. Protocol fixture와 로컬 서버 검사는 실제 모델 측정이 아니다. 실제 외부 모델 결과는 아직 없다.

## 수치와 채점

현재 `scripts/run_gpu.py`는 정확한 가공 snapshot과 CPU 학습 가중치를 사용해 같은 pinned Qwen 7B의 typed·agent·학습형 helper를 실행하는 기존 구현의 진단이다. 새 ModernBERT 모델을 학습하거나 평가하는 runner가 아니다. Native typed는 고정 조회 계획→구조화 추출, agent는 최대 2회 adaptive 조회→구조화 답변이다. 동일 초기 fragment·원문 접근·계산 후보·토큰 상한을 제공한다. 현재 학습형 helper는 선택된 도구/상태/근거/계산 값은 유지하지만 인자를 작성하며, 주문에서는 인자 누락에 따른 질문과 변경 확인 요청도 판단한다. 따라서 이 실행의 learned 결과는 내부 head만의 결과로 해석하지 않는다. 구성 요소별 지표에서 seed를 사례 안에서 평균한 뒤 source 묶음별 paired bootstrap을 수행한다. 수요 분포·발주 진단은 언어 routing 승패와 분리하고 실제 출력 발주량을 채점한다.

연결된 모의 주문 환경은 pinned τ²-bench retail 114건과 원래 500명/1,000주문/50상품 DB를 사용한다. 고객 identity와 연결 주문의 53개 묶음을 Train/Dev/Test 61/28/25로 분할한다. 공식 분할의 고객 중복 22명을 기록하고 전체 catalog/정책의 공통 공개는 유지한다. 비교 모델은 사용자 시나리오·참고 행동·미래 응답을 보지 않는다. 공통 고정 사용자 simulator만 private scenario를 읽고, 생성된 첫 발화는 같은 cache key로 모든 방법에 동일하게 제공한다.

각 주문 rollout은 새 DB에서 시작해 원래 도구로 상태를 변경한다. 같은 공개 validator가 고객 identity와 정확한 변경 제안에 대한 확인을 검사하고 위반 시도를 별도로 기록한다. 최종 DB와 원래 참고 행동을 재생한 DB를 비교하며 실패한 참고 조회는 원천 evaluator와 같이 기록하고 계속한다. 참고 변경이 전부 실패한 과제는 제거하지 않고 taskSuccess를 미판정으로 둔다. 조회만 하는 과제에서 DB가 그대로라는 이유로 성공 처리하지 않고 관측 대화의 사용자 목표 충족도 사후 판정한다. 필수 human transfer 도구 호출도 검사한다. 사후 NL/goal judge의 원시 판정·유효 coverage와 정책 위반, 결제 원장 L1 차이, 턴·토큰·실제 새 추론/cache를 보존한다. 원장 차이는 모의 USD 값이며 실제 경제적 행동 가치가 아니다. 현재 주문 학습형은 seed 42의 ABCD state/relation head 전이 진단이고 retail value policy를 학습한 결과가 아니다.

Optimizer는 support knot와 선형 regret 교차점으로 정확한 유한 minimax를 계산하고 정수 발주는 grid를 열거한다. 기존 프로포절 예제의 `q=450/7`, `Γ=900/7`, 두 요청 값 30, 한 번의 v 요청 값 85와 b 요청 값 `160/3`을 회귀 검사한다. 수정 workload의 별도 30묶음 순차 실행과 raw 사례의 수치/응답/마감 검사를 수행한다.

`Ω`의 참모수 포함률과 관측 근거의 발주 승인 여부를 독립 채점한다. 미해결 충돌·검열 수요·선호 미선택·수량/마감/허용 regret 위반을 모델의 valid 주장과 별개로 검사한다. `falseHandoff`는 포함률 오류 또는 승인 오류이며 전체 및 발주 중 비율을 기록한다. Oracle은 발주로 세지 않는다. 과거 trajectory는 원래 corpus provenance를 확인하고 기록된 응답만 재생해 재채점한다. 과거 coverage와 경제 손실은 보존하고 승인 오류를 추가한다. Raw 수치 fixture의 regret는 명시된 planning 분포 기준이며 실제 잠재수요 성과가 아니다.

Loss·질문 수·보류/자율 처리·승인 오류와 evidence/type/state/expression/value 품질을 별도 측정한다. Source 묶음 단위 paired bootstrap CI, source/config/lock/encoder hash, 학습 wall time과 cache 포함 추론 지연을 보존한다. 10%/20% 응답 오류를 모든 통제 방법에 동일하게 적용하고 오류/부분 응답 난수를 분리한다. 현재 ablation은 추론 진단이다.

## 후속 실험 순서

1. 기존 Dev/Test source를 보호하며 확보한 Train을 확장하고, 공통 토큰/표/entity와 도구 스키마 adapter를 만든다. 원문·표·대화 prefix·조회 collection·도구를 동일하게 제공한다. Gold rule, intent, 근거 span, 정답 수치와 미래 턴을 입력·후보 생성에 사용하지 않는다. 구조화 인자와 질문 template를 먼저 연결해 학습형 추론 경로에서 Qwen helper를 제거한다.
2. Base encoder·공유 인자 decoder·core head·GRU·수요 head를 공동 학습한다. Source·시점 cross-fit의 수요 gate와 자기 예측 상태의 rollout value를 적합하고, Dev 내부에서 checkpoint와 보정을 선택한다. 현재 GPU runner의 저장 가중치를 새 모델 학습 결과로 사용하지 않는다.
3. 주 비교는 고정 typed 파이프라인, 새 학습형 base, 도구 사용 agent의 세 방법이다. 같은 원문과 등록된 도구·validator·calculator·optimizer, 조회/턴 예산 및 응답 환경을 사용한다. Typed와 agent는 pinned Qwen을 사용하는 구조화 추출·도구 경로이고, 학습형은 내부 head가 인자와 질문·확인 결정을 전부 출력한다. 모든 방법에 미래 응답·참고 상태·정답 모수를 주지 않는다. 실제 ERP 필드가 있는 사례에서는 그 필드도 공통 입력으로 제공한다.
4. 추가 비교는 같은 구조의 large와 base의 `no_value` 두 조건만 먼저 실행한다. Large는 같은 head·학습 자료·문맥·정책 조건에서 encoder 크기만 바꾼다. `no_value`는 value 학습과 예상 손실 선택을 제거하고 같은 구성·인자 decoder와 Recovery의 행동 선택을 재학습해, 가치 학습의 기여를 비교한다. 같은 경제 상태의 문서 표현, 같은 누락 상태의 발주 영향·질문 비용·지연을 독립 변화시킨다. 이후 추가 제거 실험은 이 결과에 따라 정한다.
5. 원래 과제별로 근거·상태·type·계산식과 도구+관측 인자의 정확도를 보고한다. OR-ShARC는 질문할 조건과 rule 결정, 주문은 필요한 질문의 충족·중복 질문·확인 누락·권한 위반 및 최종 목표 달성을 채점한다. 질문 문장의 F1을 질문의 가치나 필요성으로 대체하지 않는다. FreshRetail은 검열 survival NLL과 비검열 NLL·CRPS·구간 coverage·평균 오차를 분리하고 종류 빈도·`familyScores`·모수를 보존한다. 분포 유효성과 별도로 모델이 실제 출력한 `q`의 실현 손실을 채점하며, scorer가 더 좋은 발주량으로 바꾸지 않는다. 연결되지 않은 공개 과제 점수를 하나의 발주 성과로 합치지 않는다.
6. 5개 seed와 source 묶음별 paired CI로 base·large·`no_value` 및 primary 방법을 비교한다. 상태·인자·분포 보정 전후 성능, 근거 없는 발주·마감/승인 위반·자율 처리율, 필요한 질문과 처리 시간, 전체 encoder/조회/도구/표현 비용·지연·최대 GPU 메모리를 함께 보고한다. 발주 손실 5% 개선과 paired CI라는 기존 완료 기준을 유지하고, 정확도·질문 부담·실행 비용의 변화로 모델 크기를 선택한다. 실제 연결 업무의 계약·판매·응답·권한·시간·비용 자료가 확보되면 동일 출력 구조로 end-to-end 발주 실험을 실행한다.
