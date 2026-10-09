# 범용 도구 확장과 이전 구조화 학습 이력

이 문서는 주 연구를 Newsvendor 모수 구성·정보 요청으로 다시 집중하기 전의 설계·실행 명령·미달 결과를 보존합니다. 아래의 “현재”, “다음”, 진행 중이라는 표현은 당시 기록입니다. 현행 주 모델과 완료 여부는 [연구 범위](research-scope.md)를 기준으로 판단합니다.

## 채택한 모델 설계와 구현 상태

다음 모델은 **ModernBERT encoder와 구조화 head를 함께 학습**하며, 도구 인자 작성도 모델 내부에 포함합니다. 출력은 근거·모수 상태·도구와 인자·분포 종류와 모수·다음 행동입니다. 질문과 확인 요청은 선택한 필드·선택지를 템플릿으로 표현합니다.

| 구분 | 기존 `native-router-v3` 저장 결과 | 새 `structured-router-v1` 구현 |
|---|---|---|
| 자연어 표현 | 고정 MiniLM, 256-token 조각의 평균 벡터 | 학습 가능한 ModernBERT, 겹치는 chunk와 원문 offset 보존 |
| 도구 인자 | CPU 정규식, GPU Qwen 보조 | 공유 schema query·span/enum/연산 선택·JSON 조립·누락 질문·확인 |
| 연구 head | 통제 연구 head와 공개 과제별 head가 별도 경로 | 근거·종류·상태·계산 관계·행동 가치·복구를 공유 모형으로 연결 |
| 수요 | 28개 통계 입력 → 종류·모수 4개 MLP | 2층 GRU hidden 128, 1일/7일 likelihood, Train cross-fit 손실 점수 |
| 실행 상태 | CPU 5개 seed 완료 | L40S 세 조건의 Train/Dev 실행 점검 완료; 전체 Train seed 42 비교 학습 진행 중 |

아래 CPU 결과와 실행 명령은 **현재 `native-router-v3` 구현**을 재현합니다. ModernBERT의 성능 결과로 읽지 않습니다.

새 경로는 [`scripts/run_structured.py`](../scripts/run_structured.py)와 [`configs/structured.json`](../configs/structured.json)을 사용합니다. encoder 학습률은 `2e-5`, 공유 층 학습률은 `3e-4`이며 기본 chunk는 1,024 tokens, overlap은 64입니다. `maxLength`를 8,192까지 설정할 수 있으며 base/large/no_value에는 같은 설정을 적용합니다. 토큰 평균으로 원문 표현을 대체하지 않습니다.

```sh
# 고정 원천 Train 전체에서 확장; 기존 Dev/Test는 그대로 보호
uv run --no-project scripts/run_structured.py --stage expand
uv run --no-project scripts/run_structured.py --stage check

# 고정 pretrained base로 encoder·결합층·GRU gradient와 수치 확인
uv run --no-project scripts/run_structured.py --stage smoke --variant base

# 언어/구성 → 기간 조건부 수요 cross-fit → 자기 예측 상태의 실제 rollout
uv run --no-project scripts/run_structured.py --stage train --variant all --seed 42
# 5개 seed 전체는 위 명령의 --seed 42 대신 --all-seeds

# 저장된 모델의 Test 재평가와 과제 adapter 없는 구조화 추론
uv run --no-project scripts/run_structured.py --stage evaluate \
  --checkpoint results/structured/base/42/model.pt
uv run --no-project scripts/run_structured.py --stage predict \
  --checkpoint results/structured/base/42/model.pt --inputs observed.jsonl
# 주문 대화의 인자·질문·확인도 내부 모델로 측정; Qwen은 사용자와 judge 역할
uv run --no-project scripts/run_structured.py --stage orders \
  --checkpoint results/structured/base/42/model.pt
```

도구 역할·상태 복사·절차 검색·후보 재평가·자기 상태 학습은 [도구 학습 변경](tool-upgrade.md)에 정리했습니다. [v3 비교](tool-upgrade-results.md)에서 확인된 복사·호출 편향·공유 가중치 간섭을 수정하고 L40S 재학습·비교를 완료했습니다. 실행기 `scripts/run_tools.py`의 기본 설정은 `configs/l40s-tools-v4.json`입니다. [수정과 결과](tool-repair.md)에서 v3 대비 인자 완전 일치 60.9%→78.3%, 호출 precision 52.2%→80.0%, 생성 Test 평균 총손실 632.03→257.34를 확인할 수 있습니다. 도구 재현율은 92.3%→76.9%로 줄었고 경제 손실도 이전 base의 240.35보다 높아, 모든 성능 목표를 달성한 구성으로 보지는 않습니다.

현재 [성능 목표](performance-goal.md)는 도구·인자·호출 precision 각각 90% 이상과 생성 총손실 240.35 미만입니다. v5에서 대화와 후보의 공동 인코딩, 별도 호출 판단, Train 응답 이후 구성 학습을 추가했고, 초기 Dev에서 새 선택 경로의 낮은 정확도를 확인했습니다. 이를 기존 학습 점수의 보정 방식으로 바꾼 v6는 `scripts/run_tools.py --config configs/l40s-tools-v6.json --continuation results/l40s-tools-v5/base/42/language-progress.pt`로 이어갑니다. 이후 동일 실행 재개에는 `--continuation` 대신 `--resume`을 사용합니다. 언어·가치 학습과 Train/Dev 선택을 모두 마치고 가중치를 동결한 뒤 고정 Test를 평가합니다.

v7는 문맥 점수 직접 학습을, v8는 공개 정책의 실행 조건 보존을 추가했습니다. v9는 패딩 묶음을 최적화해 L40S 역전파 probe 처리량을 8.20→10.57건/초로 높였습니다. v10은 기본 점수와 재평가 점수를 각각 학습하도록 연결을 수정했습니다. v11은 과거 호출 횟수 대신 실제 Train 흐름의 다음 공개 도구 위치를 절차별로 학습합니다. v12는 원천 Train 슬롯에서 구별 가능한 역할 표적 16,587개를 기존 역할 head 학습에 추가합니다. 실행은 `scripts/run_tools.py --config configs/l40s-tools-v12.json --continuation results/l40s-tools-v11/base/42/language-progress.pt`입니다. 성능 목표는 아직 미달이며 변경별 가중치·optimizer 출처와 실패한 결과도 보존합니다.

v12의 최종 Test는 도구 76.9%, 도구+인자 78.3%, 호출 precision 71.4%, 생성 총손실 248.0525로 네 목표 모두 미달했습니다. v13은 학습률을 낮추고 Train 응답 이후 상태 130개를 추가했지만 Dev 9회 연속 개선이 없어, 74,752건의 복원 가능한 checkpoint를 보존하고 중단했습니다. v13의 가치 학습·Test는 실행하지 않았습니다.

현재 v14는 controller의 문맥별 후보 표현을 재평가에 연결하고 호출 여부와 조건부 도구 분포를 함께 학습합니다. 파라미터와 encoder 통과를 추가하지 않았으며, L40S 64건 probe의 정상 구간은 기존 9.67건/초·수정 9.58건/초였습니다. 실행은 `scripts/run_tools.py --config configs/l40s-tools-v14.json`이며 같은 실행 재개에는 `--resume`을 사용합니다. Dev는 4,096건마다 확인하고 새 단계 16,384건 이후 연속 3회 미개선이면 종료합니다. 초기 Dev는 부모와 같으며 추가 학습의 효과는 아직 미확인입니다.

최종 평가 실행기는 `scripts/finalize_tools.py --run results/l40s-tools-v14/base/42 --wait`입니다. 실제 학습 프로세스가 종료되고 Train/Dev 선택이 완료된 가중치를 고정한 뒤 평가합니다. Dev 목표 미달도 기록하며, 완료된 동일 Test 비교는 재사용합니다.

L40S 한 장에서 세 조건을 순차 실행하는 명령은 다음과 같습니다. CUDA 13.0 PyTorch 환경을 별도로 만들고 L40S UUID로 장치를 제한합니다. CPU 환경은 그대로 유지합니다.

```sh
uv run --no-project scripts/run_l40s.py --seed 42
# 중단 시 optimizer·RNG·순서·선택된 Dev checkpoint를 복원
uv run --no-project scripts/run_l40s.py --seed 42 --resume
```

[`configs/l40s-efficient.json`](../configs/l40s-efficient.json)은 256-token 원문 chunk를 질문·현재 상태로 검색하고, 처음에는 문서 1,024 tokens를 선택합니다. 추가 조회마다 문서 예산을 두 배로 늘려 최대 16,384 tokens까지 읽습니다. 대화 이력과 표 cell은 유지하고 선택된 자료를 1,024-token sequence에 묶으며 원문 위치를 보존합니다. 같은 forward 안에서 동일한 도구·필드 스키마는 한 번 인코딩하고 사례별 결합층에서 관측 자료와 연결합니다. encoder는 전부 미세조정하며, 이전 가중치의 표현을 재사용하지 않습니다. BF16 encoder 연산·FP32 손실, 길이별 encoder microbatch 16, 층별 `torch.compile`, fused AdamW와 CPU 준비 worker 2개를 사용합니다.

94,137개 언어 Train 사례를 3 epochs 학습하고 16사례마다 가중치를 갱신합니다. base/no_value는 사례 batch 16, large는 8을 사용하며 같은 유효 batch·입력·학습률·head로 비교합니다. 기본 입력에서 주석 근거가 보이지 않는 Train 사례에만 추가 조회 학습을 적용합니다. Dev에서는 모델이 선택한 조회로 검증하고, 최종 Test 결과로 checkpoint를 선택하지 않습니다. `no_value`는 base의 언어·수요 학습 checkpoint를 공유한 뒤 별도 rollout 정책을 학습합니다. 이전 입력 구성과 갱신 간격이 달라 두 backbone은 고정된 pretrained revision에서 시작합니다. 이전 29,000사례 checkpoint와 원시 기록은 `results/l40s-opt/`에 보존합니다.

실행 중 `results/l40s-efficient/<variant>/42/progress.json`, 시작 설정·하드웨어·출처는 `run.json`, 실행한 소스는 `source.tar.gz`에 저장합니다. 1,024사례마다 언어 학습을 재시작할 수 있고 완료한 언어·수요 단계도 재사용합니다. 데이터·설정·소스 hash가 바뀌면 재시작을 거부합니다. 사용자 서비스는 `newsvendor-l40s-efficient-42.service`입니다. 최종 `comparison-42.md`와 `comparison-42.json`에는 같은 Test의 과제별 정확도·근거, rolling 수요 calibration·발주 손실, 통제 rollout 손실·행동 수·초기 모수 유형/상태, 지연·GPU 메모리와 source family 단위 95% bootstrap 차이를 저장합니다. 전체 학습과 `--limit` 진단을 구분합니다. [`scripts/bench_l40s.py`](../scripts/bench_l40s.py)는 Train만 사용하고 config·source·자료·입력·checkpoint hash와 모든 시간 측정을 저장합니다. [`scripts/audit_inputs.py`](../scripts/audit_inputs.py)는 원문 토큰 비용과 조회 후 주석 근거·피연산자 접근 가능성을 검사합니다. 처리량 측정과 입력 검사 자체는 예측 성능의 근거가 아닙니다.

정책 단계의 망각을 수정한 실행은 다음과 같습니다.

```sh
uv run --no-project scripts/run_l40s.py --config configs/l40s-policy-v2.json \
  --common-root results/l40s-efficient --seed 42
```

완료된 언어·수요 단계는 원래 backbone별 `common.pt`에서 가져오며, 원래 config·source·자료·checkpoint·source archive hash를 `common-parent.json`에 기록합니다. 정책 update는 연구 사례 8개와 공개 Train 복습 8개를 섞습니다. 복습은 ABCD 호출·발화를 각각 2개, 나머지 공개 구성 과제를 각각 1개씩 source family 단위로 선택합니다. encoder도 계속 학습합니다. 고정된 초기 모형의 예측은 복습 손실의 표적으로만 사용하며 추론 표현을 재사용하지 않습니다. 정책 checkpoint는 발주 Dev 손실이 개선되고 과제별 공개 Dev 지표도 유지될 때 채택합니다. 통과한 후보가 없으면 초기 checkpoint를 유지하고 그 이유를 보고서에 남깁니다.

관측 요청에 없는 행동과 스키마에서 지원하지 않는 추출 모드는 차단합니다. 상태 주석을 사용하는 연구 모수는 `stateRequired`로 검증하고, 도구 인자는 추출값·누락/충돌·JSON 타입·확인 조건으로 판단합니다. 이전 Test와 원시 결과는 `results/l40s-efficient/`에 보존하며 수정 실행의 동일 Test 평가는 회귀 비교입니다. 새로운 조직 업무의 효과성 근거로 해석하지 않습니다.

새 발주 rollout은 추가 조회·재인코딩도 허용합니다. `encoder.retrievalCost=1.0`은 통제 실험의 가정 비용이며, 실제 조직의 측정 비용으로 해석하지 않습니다. 조회는 같은 관측 원천의 인코딩 범위를 확장하고 남은 행동 예산을 사용합니다. 비용과 최종 손실은 실제 실행한 trace에서 보존합니다. 정식 tool schema의 승인도 도구·인자 전체 hash에 묶으며, 인자가 달라지면 다시 확인합니다.

`observed.jsonl`은 snapshot과 같은 `id`·`input` 형식입니다. 관측 자료와 JSON tool schema만 넣고 평가 주석은 분리합니다. `predict`는 구조화 결과를 저장하며 외부 도구를 실행하지 않습니다. 실제로 연결된 문서·판매 자료는 Python `infer(..., linked=True, bounds=..., hold_cost=...)`로 명시하며, 현재 공개 과제는 두 경로를 별도로 학습합니다. 공개 과제에는 경제적 비용 주석이 없으므로 관측 행동 head를 사용하고, 발주 환경에서는 학습된 잔여 손실과 요청 비용으로 행동을 선택합니다.

확장 Train에는 기존 85개 도구 사건 외에 **92,733개 사례·27,781개 도구 사건**을 추가했습니다. ABCD의 slot 목록은 가능한 필드 목록이며 실제 주석은 0·1·3개 값의 순서 있는 목록입니다. 이 adapter는 위치별 공유 추출과 사용 여부를 학습하고, 정식 JSON schema는 필드별 필수 조건을 적용합니다. 원천 Train 8,034개 대화 중 7,764개가 추가 사례를 제공하며, 기존 사례 369개와 보호된 출처·prefix 135개 사건은 제외했습니다. 동일 정책은 한 번 저장하고 원문 hash로 참조합니다. 대화·prefix·동일 입력 연결을 검사하고 기존 Dev/Test 입력 hash를 보존합니다.

base의 실제 전체 크기는 **151,848,374 trainable parameters**입니다. smoke는 작은 관측 사례에서 실제 pretrained encoder와 결합층의 gradient, optimizer 갱신 및 1일/7일 수요 경로를 확인했습니다. 이 확인은 성능 실험이나 실제 조직 업무의 효과성 근거가 아닙니다. 전체 학습의 가중치·원시 rollout·분포별 OOF NLL·선택된 분포와 발주량·지연·GPU 메모리는 variant/seed별 `results/structured/`에 저장합니다.

### Encoder와 결합층

| 구성 | 채택 설정 | 처리 대상 |
|---|---|---|
| 자연어 encoder | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), 약 149M 파라미터, 8,192-token 문맥, 전체 미세조정 | 요청·대화·업무 문서·표·도구 스키마 |
| 판매 이력 encoder | 은닉 크기 128, 2층 GRU | 과거 판매·품절·할인·달력과 관측 mask |
| 결합층 | 256차원, 2층 attention | 스키마 필드 표현·근거 토큰·판매 이력 표현 |
| 전체 크기 예산 | 약 152–155M 파라미터 | base encoder에 결합층·이력 encoder·공유 head를 포함한 구현 목표 |
| 크기 비교 | [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), 약 395M, 같은 8,192-token 문맥 | 동일 head·결합 차원·분할에서 encoder 크기의 효과 확인 |

계획에 사용할 모델 revision과 라이선스는 [출처 문서](sources.md)에 고정합니다.

문서는 겹치는 조각으로 나누고 원문 ID·토큰 위치·문자 offset을 보존합니다. 요청과 대화를 조건으로 근거를 조회·재순위화한 뒤 선택된 문서 조각을 함께 인코딩합니다. 근거 head와 인자 head는 토큰 표현을 사용하며, 문서 전체를 하나의 평균 벡터로 바꾸지 않습니다. 조회에서 정답 근거가 확보됐는지와 확보된 근거에서 head가 올바르게 판단했는지를 각각 측정합니다.

문서와 판매 이력이 실제로 함께 주어진 사례에서 두 표현을 결합합니다. 개별 공개 과제는 존재하는 입력 경로와 주석만 사용하고 나머지는 mask합니다. 서로 다른 출처의 계약과 시계열을 임의로 붙여 결합 학습의 정답 사례로 만들지 않습니다. 위 파라미터 수는 설계 예산이며, 구현 후 실제 trainable 수·GPU 메모리·지연을 기록합니다.

### 공유 head와 구조화 출력

과제별 adapter는 원천 입력과 주석을 아래 공유 head에 연결합니다. 원천 과제의 정답과 상태 label을 보존하고, 대응되는 공유 head만 감독합니다.

| Head | 출력 |
|---|---|
| 근거 `evidence` | 필드별 원문 ID와 근거 span·표 cell |
| 종류 `type` | 사실·추정·선호·가정; 수요의 분포 종류와 별도 |
| 상태 `state` | 확인됨·후보·미확인·충돌·미확보와 대상 필드 |
| 계산 관계 `relation` | 연산 종류, 피연산자 span·cell·필드 연결과 단위·scale |
| 행동 가치 `value` | 상태·후보 행동별 예상 후속 손실과 행동 비용 |
| 복구 `recovery` | 추가 조회·질문·충돌 해소·확인·진행·보류할 대상과 행동 |
| 도구·인자 | 도구 선택, 스키마 필드별 값·근거·누락·충돌 |
| 수요 종류 | 절단정규·로그정규·Weibull의 조건부 손실 점수 `familyScores`, 최소 점수의 `family` |
| 수요 모수 | 종류별 0수요 확률과 위치·척도, 로그평균·로그표준편차 또는 형상·척도 |

도구 인자는 필드마다 별도 거대 모형을 두지 않고, 스키마를 조건으로 같은 head를 공유합니다. 원문 값은 span pointer, 닫힌 선택지는 분류, 숫자·날짜·단위는 추출 후 결정적 정규화로 처리합니다. 산술은 선택한 피연산자와 연산을 executor가 계산합니다. JSON 직렬화·스키마 검사·확인 절차 검사와 Newsvendor 최적화도 executor가 담당합니다.

다음 학습형 추론에는 Qwen 보조를 호출하지 않습니다. 모델이 종류·모수에서 만든 `F`를 solver에 전달하고, 필요한 질문은 필드와 선택지를 출력합니다. 비교군 typed·agent, 주문 과제의 사용자 simulator·judge는 별도 실행 역할로 유지합니다.

### 학습 순서와 수요 종류 선택

1. **근거·필드 학습:** 공개 원천 Train을 확장해 토큰 근거, 인자, 상태와 계산 관계를 먼저 학습합니다. ModernBERT도 함께 학습하며 encoder와 새 층의 학습률을 분리합니다. 없는 주석은 loss에서 mask하고, 알 수 없는 값을 임의의 상태 정답으로 만들지 않습니다.
2. **판매 이력·수요 학습:** GRU와 같은 기간 조건부 수요 head로 1일 관측을 보조 학습하고, 7일 총수요를 주 학습·평가 대상으로 둡니다. 7일 분포는 총수요에 직접 맞추며 일별 분포를 독립으로 가정해 합치지 않습니다. 품절 기간은 관측 총판매에 대한 survival likelihood를 사용합니다.
3. **행동 가치·복구 학습:** Train의 관측 상호작용과 통제 rollout에서 후속 손실·질문/조회 비용을 학습하고, 주석이 연결된 경로를 공동 미세조정합니다. 공유 head가 구성한 상태와 후보 행동으로 결정하며, 정답 상태를 추론 입력으로 공급하지 않습니다.

수요 종류 head의 다음 학습은 **Train 내부 출처·시간 단위 cross-fit**을 사용합니다. 시간 경계에서는 겹치는 미래 관측 기간도 분리합니다. 종류별 모수 모형이 자신이 학습하지 않은 구간에서 만든 density·survival NLL을 표적으로, 종류 head가 조건부 기대 손실 또는 초과 손실 3개를 예측합니다. 선택은 최소 예측 손실의 단일 종류이며, `familyScores`는 그 손실 점수입니다. 분포 종류 정답 label을 새로 만들거나 혼합분포 likelihood로 학습한 뒤 단일 종류로 바꿔 출력하지 않습니다.

최종 모수 모형은 Train 전체로 학습하고, Dev는 **실제로 선택된 종류와 해당 모수**의 NLL로 selector checkpoint를 결정합니다. 현재 구조화 경로에는 별도 사후 보정이 없습니다. Test는 선택·cross-fit에서 제외합니다. native-v3의 detached NLL 가중 목적식과 `familyProbabilities` 출력은 아래 구현 기록에 그대로 구분해 두었습니다.

### 원천 확장과 최소 비교

현재 snapshot의 ABCD는 대화 80개에서 462개 사례를 준비했고, Train의 도구 호출 사례는 85개입니다. 다음 인자 학습은 원천 [ABCD Train의 8,034개 대화](https://github.com/asappresearch/abcd)를 활용하도록 확장합니다. CUAD·ContractNLI·OR-ShARC·TAT-QA·FreshRetail도 과제별 Train의 근거·인자·관측 표적을 확대합니다. 기존 snapshot의 Dev/Test와 연결된 출처 묶음은 확장 Train에서 제외하고, 대화·계약·규칙·context·매장/상품·중복 입력 단위 분할을 유지합니다. 새로운 자연어·시계열 결합 사례는 실제 연결된 자료와 Train의 명시적 주석으로 추가합니다.

구현된 최소 학습형 비교는 **base 전체 모형, large 전체 모형, base에서 value를 제거한 `no_value`** 세 가지입니다. 같은 원문·스키마·도구·분할·행동 상한의 typed·agent를 주 베이스라인으로 둡니다. base/large는 같은 구조화 head와 내부 인자 출력을 사용합니다. `no_value`는 value 손실과 예상 손실에 따른 선택을 제거하고, 같은 구성·인자 모듈과 Recovery의 관측 행동 주석 학습으로 다음 행동을 선택하도록 재학습합니다. 근거·필드 상태·정확한 도구와 전체 인자·필요한 질문·최종 목표 달성·발주 손실·호출 비용·지연·메모리를 채점합니다. 자유 문장의 표현 F1을 학습형의 필수 출력 요건으로 두지 않습니다.
