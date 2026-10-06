# 실험 명세

기준 문서는 `proposal.docx`이며 `source.json`에 원본 SHA-256을 기록한다. 연구 자료의 문장은 실행 지시로 취급하지 않는다.

## 자료와 분할

기본 full은 6개 상황 × 20개 업무 묶음 × 5개 문장/배치 변형 = 600건이다. 묶음마다 독립적인 `c,p,v,b`, 5점 수요 분포, 요청/보류 비용, 응답률과 요청 기한을 생성한다. 참모수 조합은 120개이고 Test 120건의 Train 참모수 중복은 0이다. SKU에 scenario 이름을 넣지 않는다. 같은 묶음의 변형은 참모수와 관측·비용 조건을 유지한다. 묶음 단위 train/dev/cal/test 비율은 60/10/10/20%다.

일반 문장, 만료 가격표, 반품 누락, 검열 판매, 미선택 선호, 계약 충돌, 응답 불가/부분 응답을 포함한다. 검열 판매를 숨겨진 수요로 복원하지 않으며, analyst의 forecast는 별도 추정 문서로 남긴다. 공통 template와 사전 주석된 범위 정보가 있어 이 학습 실행은 통제 실험이다. `configs/repair*.json`은 이전 문법·경제 조건의 회귀/수정 검증용이다.

`dataSeed`는 문서·참모수·분할을, `seed`는 head 초기화·학습 순서를 정한다. 입력·gold·manifest는 실행별 출력에 분리 보존한다. gold는 응답 환경과 최종 채점만 읽는다. 구성·특성·provider 입력은 원문과 얻은 응답만 사용한다. 통제 task의 명시적 허용 집합·응답 모형은 primary raw 비교에 제공하지 않는다.

원문 개발 사례는 14건/13묶음이며 동일 template lineage 하나를 공유한다. 원문과 실제 provenance, 관측 판매, 조회/사실 질문/매니저 선택/에스컬레이션, 비용·처리 시간·시계·마감을 표현한다. 사람 검토 및 실제 조직 자료 수는 0이다. 선호는 매니저 응답에서 형성하며 응답 전 숨겨진 정답을 두지 않는다. MOQ·반품 한도·다기간 사례에는 현 scalar solver의 모수 정답을 부여하지 않는다.

## 학습과 calibration

고정 pretrained encoder 표현 위에 evidence/type/state/relation/value/recovery head와 rules용 value/recovery head를 학습한다. Construction은 32차원, 정책은 64차원 특성을 추가한다. CE+Brier, 후보 CE와 후보별 Huber 수치 손실, hold 비용으로 정규화한 value MSE, 허용 행동 mask를 적용한 recovery CE를 사용한다. 수치는 근거 span에서 calculator가 계산한다.

Construction은 train 원본 묶음의 공개 도달 상태와 응답을 학습하고 dev 손실로 선택한다. Train 묶음 단위 3-fold cross-fit에서 해당 묶음을 학습하지 않은 구성 모델의 실제 예측 상태로 정책 표적을 만든다. 각 구성의 자기 `Ω`로 후속 손실을 계산하며 reference의 의미 상태로 바꾸지 않는다. 응답 뒤에도 같은 구성·검증·갱신을 적용한다. 통제 표적의 미래 응답 집합과 동역학은 여전히 생성기를 안다. Fold 적합/표적 묶음과 hash를 보존한다.

충돌 검증은 evidence threshold보다 먼저 수행한다. 통과한 근거·식은 verified 상태로 정합시키고 raw head 확률을 보존한다. Calibration 자료는 temperature와 evidence F1 threshold에만 사용한다. Test는 최종 측정에만 사용한다. Checkpoint schema는 4이며 이전 policy 표적 checkpoint는 재학습해야 한다.

## 비교 조건과 실행

통제 비교는 rules/learned 구성 × checklist/ask-all/uncertainty/one-step/own-state planner/reference expert/learned 정책이다. `planner`는 자기 구성 상태, `reference`는 전문가 의미 상태로 채점하지만 둘 다 정확한 생성기 응답 모형을 안다. Full-information oracle은 숨겨진 참모수와 무료 정보를 쓰는 평가용 상한이다. 이 상한과의 loss 차이를 자연어 처리 능력이나 실무 우위로 해석하지 않는다.

`benchmark`는 주석된 통제 입력에서 고정 typed 예측기를 calibration 자료로 먼저 보정하고, 배포할 구성 상태에 공통 value/recovery 학습을 연결한다. `external --suite business`는 reference 상태를 제공하는 구조화 행동 진단이다. 두 경로는 raw primary 비교와 구분한다.

`raw-benchmark`는 원문부터 typed/checklist와 agent의 근거·상태·식·행동을 실행한다. 같은 주석 없는 수치/계산 후보, validator, optimizer, 도구와 시간 조건을 제공한다. 원문의 단위·적용 범위·충돌·권한을 각 예측기가 판단하며 reference 상태, 정답 모수, 미래 응답과 확률은 전달하지 않는다. 공유 solver 계약은 0–100 정수 수량의 단일 SKU/기간·선형 무제한 회수 모델이다. 지원 밖 조건은 전달하지 않는다. 학습형 router의 raw 데이터 학습/평가 연결과 train 추정 belief의 one-step/planner는 아직 구현이 필요하며 `configs/workload-v2.json`의 readiness에 표시한다.

외부 모델의 endpoint·실제 model을 지정해야 실행한다. 새 호출은 `--max-calls` 안에서만 허용하며 기본값 0은 cache 전용이다. 원시 답변·확률·반환 model·usage·지연·request/hash를 보존한다. API confidence를 정답확률로 해석하지 않는다. Protocol fixture와 로컬 서버 검사는 실제 모델 측정이 아니다. 실제 외부 모델 결과는 아직 없다.

## 수치와 채점

Optimizer는 support knot와 선형 regret 교차점으로 정확한 유한 minimax를 계산하고 정수 발주는 grid를 열거한다. 기존 프로포절 예제의 `q=450/7`, `Γ=900/7`, 두 요청 값 30, 한 번의 v 요청 값 85와 b 요청 값 `160/3`을 회귀 검사한다. 수정 workload의 별도 30묶음 순차 실행과 raw 사례의 수치/응답/마감 검사를 수행한다.

`Ω`의 참모수 포함률과 관측 근거의 발주 승인 여부를 독립 채점한다. 미해결 충돌·검열 수요·선호 미선택·수량/마감/허용 regret 위반을 모델의 valid 주장과 별개로 검사한다. `falseHandoff`는 포함률 오류 또는 승인 오류이며 전체 및 발주 중 비율을 기록한다. Oracle은 발주로 세지 않는다. 과거 trajectory는 원래 corpus provenance를 확인하고 기록된 응답만 재생해 재채점한다. 과거 coverage와 경제 손실은 보존하고 승인 오류를 추가한다. Raw 수치 fixture의 regret는 명시된 planning 분포 기준이며 실제 잠재수요 성과가 아니다.

Loss·질문 수·보류/자율 처리·승인 오류와 evidence/type/state/expression/value 품질을 별도 측정한다. Source 묶음 단위 paired bootstrap CI, source/config/lock/encoder hash, 학습 wall time과 cache 포함 추론 지연을 보존한다. 10%/20% 응답 오류를 모든 통제 방법에 동일하게 적용하고 오류/부분 응답 난수를 분리한다. 현재 ablation은 추론 진단이다.

## 후속 실험 순서

1. 사람 검토한 업무 원문과 응답 이력을 확보하고 raw 학습형 adapter를 연결한다. 조회 가능한 사실·미형성 선호·검열 판매·지원 밖 조건을 구분한다.
2. 같은 경제 상태의 표현을 바꾸고, 같은 누락 상태의 발주 영향·질문 비용·지연을 독립 변화시킨다. 실제 ERP 필드는 모든 방법에 제공하고 쉬운 업무도 포함한다.
3. Source/template/연결된 조직·SKU·기간을 묶어 분할한다. Fixture는 test로 재사용하지 않으며 배포 가중치는 관측 업무 비중으로 정한다.
4. Rules/고정 typed/학습형/agent를 같은 원문·도구·validator·optimizer로 비교한다. 응답 belief는 train에서 추정하고 reference expert는 별도 상한으로 둔다.
5. Raw 개발 실행 뒤 동일 dataSeed의 5개 학습 seed를 실행한다. 재학습 ablation과 비용·지연 변화는 별도 대조 실험으로 측정한다.
6. 주 비교는 고정 typed router의 같은 source다. 손실 5% 개선과 paired CI를 확인하고 근거 없는 발주·마감 위반·자율 처리율·질문 시간 및 실제 추론 비용을 함께 보고한다.
