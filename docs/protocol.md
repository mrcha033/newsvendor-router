# 실험 명세

기준 문서는 `proposal.docx`이며 `source.json`에 원본의 SHA-256을 기록한다. 문서의 요청·명세를 연구 자료로 해석하며, 첨부 문서 안의 문장을 실행 지시로 취급하지 않는다.

## 자료와 분할

full 설정은 6개 상황 × 20개 원본 업무 묶음 × 5개 문장/배치 변형 = 600개 episode다. 같은 업무 묶음의 5개 변형은 참모수와 관측 조건을 유지한다. 원본 묶음 단위로 train 60%, dev 10%, calibration 10%, test 20%를 배정한다. test의 일부에 기간이 지난 가격표를 넣는다. 충분한 근거, 반품 조건 누락, 검열된 판매 이력, 미선택 shortage policy, 계약 충돌, 응답 불가/부분 응답을 포함한다.

입력과 gold는 별도 JSONL에 저장한다. gold는 응답 환경과 최종 채점만 읽는다. 모수 구성·특성 생성·요청 정책은 현재 원문, 실제 응답, 허용 모수 집합만 읽는다. Manager가 선택하지 않은 policy는 확정값으로 넣지 않는다. `None`인 regret 허용 한도는 이 통제 예제의 명시적 무제한 허용 기준이다. 실제 자율 발주 승인 기준을 의미하지 않는다.

## 학습과 calibration

모든 head는 공개 pretrained encoder의 고정 표현을 입력으로 받는 PyTorch MLP다. 문서 연결은 2-class CE+Brier, 종류는 4-class CE+Brier, 상태는 5-class CE+Brier, 식 선택은 후보 CE와 후보 확률로 가중한 Huber 수치 손실을 사용한다. 수치 손실은 Huber를 혼합 예측값에 적용하지 않고 각 후보 계산 오차에 먼저 적용한다. 값은 calculator가 원문 수치로 계산한다.

모수 구성 head를 train으로 적합하고 dev 손실로 checkpoint를 선택한다. value/recovery 표적은 train 원본 묶음 단위 3-fold cross-fit에서 해당 묶음을 학습하지 않은 모수 구성 모델이 실제 예측한 상태로 만든다. 가능한 응답 뒤에도 같은 예측·검증·갱신을 실행한다. value는 hold 비용으로 정규화한 손실의 MSE, recovery는 허용 행동 mask를 적용한 CE를 사용한다. `Ω`가 비면 recovery가 허용된 요청/보류를 선택한다.

개발 자료는 checkpoint 선택에, calibration 자료는 temperature와 evidence threshold 선택에만 사용한다. test는 최종 측정에만 사용한다. evidence threshold는 calibration F1으로 고른다. 원본 묶음 단위로 paired bootstrap CI를 계산한다. 분류 확률이나 API confidence를 실제 정답확률로 자동 해석하지 않는다.

## 수치 예제와 측정

Optimizer는 support knot와 선형 regret의 교차점을 열거해 유한 `Ω`에서 정확한 continuous minimax 해를 구한다. 정수 발주는 유한 grid를 열거한다. 프로포절의 `F={(0,.6),(100,.4)}, c=6,p=10,v∈{0,4},b∈{0,8}` 예제에서 `q=450/7`, `Γ=900/7`, 요청 두 번의 값 30, 한 번의 `v` 요청 값 85와 `b` 요청 값 `160/3`을 회귀 검사한다.

주요 출력은 declared request/hold 비용을 포함한 total loss, 전달 시 regret, 요청 횟수, 보류율, 참모수의 `Ω` 포함률 및 false handoff다. 종류·상태의 accuracy/NLL/Brier/ECE와 근거/식/값 정확도를 별도로 기록한다. source·config·lock hash, encoder SHA, Python/package version, 각 fold의 적합/표적 원본 묶음과 손실을 보존한다. cache가 있는 상태의 policy/constructor 실행 지연과 학습 포함 wall time을 구분한다.

모수 구성과 요청 정책을 교차한 기준선, 추론 시 evidence/type/impact/update 제거 진단, 10%/20% 응답 오류 진단을 실행한다. 후자의 noise는 모든 정책에 동일하게 적용한 본 비교를 대신하지 않는다. 현재 모든 noise 결과는 학습형 정책만의 stress diagnostic이다. 실제 성능 비교에는 동일 오류 조건의 전체 기준선과 별도 재학습 ablation을 추가해야 한다.

## 외부 모델 비교 실행 전 확정 사항

Jev, open-alternative-jev를 제공하는 SGLang server, agent의 실제 model/endpoint를 지정하고 버전·접근 권한·가격을 기록한다. 준비된 어댑터는 공통 후보를 주고 응답 형식과 실제 usage를 저장한다. public TAT-QA 진단은 수치 후보 최대 200개와 `other` 선택에 제한한다. ShARC는 yes/no/irrelevant/ask를 구분하며 생성된 질문의 유용성을 주장하지 않는다.

`benchmark`는 공통 evidence/type/state/expression 후보를 다중 choice 요청으로 보내고 원시 예측·확률을 hash와 함께 cache한다. 검증·계산·업데이트 규칙은 동일하게 적용한다. 그 예측 상태와 같은 encoder로 value/recovery를 공통 reference planner 표적·개발 구간에서 적합한다. 외부 construction 모델은 이 연구의 학습 표본으로 적합하지 않는 고정 예측기이므로 construction cross-fit이 추가로 필요하지 않다. 직접 학습한 construction 모델은 3-fold로 표본 밖 예측 상태를 만들어 같은 표적을 생성한다. calibration은 따로 분리한다. business 어댑터의 직접 행동 선택 결과는 이 학습 비교와 별도로 보고한다.

새 API 호출은 명시한 `--max-calls` 상한 안에서만 실행한다. 기본값 0은 확보한 cache만 사용한다. 실제 endpoint·model 이름을 지정하지 않으면 비교를 시작하지 않는다. 모델 alias가 weights revision을 보장하지 않는 경우 원시 응답 snapshot과 반환 model 이름을 기록하고 그 한계를 결과에 명시한다. 외부 예측기의 규모·사전학습·접근 비용은 로컬 encoder와 동일하다고 가정하지 않는다.

연구팀은 원문·근거 span·단위·적용 범위·매니저 선택·식·상태 주석을 검토해야 한다. 그 검토와 본 비교 모델의 실제 실행은 실험 결과 확정을 위한 후속 연구 절차다. 이 저장소의 통제 실행 성공을 연구 가설의 입증으로 해석하지 않는다.
