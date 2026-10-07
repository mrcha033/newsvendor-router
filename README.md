# Newsvendor Decision Router

업무 문서의 **근거 연결·모수 상태·행동 가치**를 학습하는 Decision Router의 실험 저장소입니다. 연구 모형은 [프로포절](docs/proposal.docx)에 정리했습니다. 주 비교는 같은 원문과 도구를 사용하는 **고정 typed 파이프라인·학습형 router·agent**입니다.

## 다른 GPU 머신에서 바로 실행

Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), CUDA GPU를 사용합니다. GPU 메모리는 24GB 이상을 권장합니다.

```sh
git clone https://github.com/mrcha033/newsvendor-router.git
cd newsvendor-router

# 포함된 데이터와 5개 seed 가중치의 hash·분할 검사
uv run --no-project scripts/run_gpu.py --check

# 남은 구성 요소 비교와 주문 대화 평가를 모두 실행
uv run --no-project scripts/run_gpu.py --stage all
```

GPU runner는 데이터 snapshot을 자동 복원하고 저장된 CPU 학습 가중치를 읽습니다. 데이터 원천을 다시 수집하거나 head를 재학습할 필요 없이 실행할 수 있습니다. 첫 추론에서 고정 revision의 Qwen과 MiniLM을 다운로드합니다.

프로젝트의 `uv sync`는 Linux에서 CPU PyTorch를 설치합니다. GPU 명령의 `--no-project`는 script에 선언된 별도 의존성 환경을 사용하기 위한 옵션입니다.

## GitHub에 포함된 데이터와 가중치

[`cases/evaluation.tar.xz`](cases/evaluation.tar.xz)에 **상보적 영어 과제 1,802건과 연결된 주문 과제 114건**의 가공 데이터를 담았습니다. [`cases/evaluation.json`](cases/evaluation.json)에 압축파일 크기와 파일별 SHA-256·원천 revision·라이선스를 기록했습니다.

| 포함 항목 | 저장 위치 또는 복원 위치 | 구성 |
|---|---|---|
| 상보적 영어 과제 | `data/processed/complementary/` | 원문 입력·분리된 정답·공통 조회 collection·manifest |
| 연결된 주문 과제 | `data/processed/orders/` | 원문 입력·분리된 정답·고객/주문/상품 DB·manifest |
| CPU 학습 가중치 | [`models/native/{42..46}.npz`](models/native/) | 5개 seed의 12개 head, NumPy 가중치 |
| 학습 설정·CPU 원시 결과 | [`models/native/`](models/native/) | seed별 metadata·예측·trace·측정·지표, 전체 `runs.json` |
| 통제 Newsvendor 600건 | [`configs/full.json`](configs/full.json), [`corpus.py`](src/newsvendor/corpus.py) | `dataSeed=42`의 생성 코드·설정으로 재생성 |
| 원문 개발 사례 | [`cases/pilot/`](cases/pilot/) | 입력 14건과 주석·manifest |

복원되는 `data/processed/`와 실행 결과 `results/`는 Git에서 제외됩니다. 데이터 자체는 위 압축파일로 추적합니다. 통제 Newsvendor 600건은 아래의 `prepare` 명령으로 생성합니다.

### 상보적 영어 과제의 출처와 분할

| 원천 | 준비 규모 | 학습·채점 대상 |
|---|---:|---|
| [CUAD](https://github.com/The-Atticus-Project/cuad) | 450건 / 계약 75개 | 공급·제조·유통·구매 계약의 조건 근거와 미기재 |
| [ContractNLI](https://github.com/stanfordnlp/contract-nli) | 360건 / NDA 60개 | 함의·모순·미기재와 근거 |
| [OR-ShARC](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) | 240건 / 규칙 묶음 60개 | 651개 공통 규칙에서 조회 후 결정·추가 질문 |
| [ABCD](https://github.com/asappresearch/abcd) | 462건 / 대화 80개 | 전체 정책·관측 대화에서 다음 발화·도구·인자 |
| [TAT-QA](https://github.com/NExTplusplus/TAT-QA) | 240건 / context 60개 | 문장·표의 답·계산·단위·scale |
| [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) | 시계열 50개 | 후속 7일 총수요 분포 `F`와 발주 손실 |

Train/Dev/Test는 **1,074/362/366건**, 출처 묶음은 378개입니다. 같은 계약·규칙 page/tree·대화·매장·상품의 연결 묶음과 동일 입력·유사 template를 같은 분할에 둡니다. TAT-QA는 context 단위입니다. OR-ShARC의 전체 규칙과 ABCD의 전체 정책을 공통으로 제공하고, 각 모델이 사용할 근거와 다음 행동을 구성합니다.

공통 모델 입력은 `newsvendor.suite.public_input(row)`이며 request·documents·tables·관측 history/sales·도구 목록을 반환합니다. 정답·근거 주석·의도·정답 규칙 ID·미래 턴은 별도 파일에 둡니다. 자료의 고정 revision과 수집 hash는 [`configs/complementary.json`](configs/complementary.json)과 [manifest](cases/complementary/manifest.json)에 있습니다.

### 연결된 주문 과제의 출처와 분할

[τ²-bench retail](https://github.com/sierra-research/tau2-bench/tree/5bfa7e37b36656b37dc6d022156be6563c1007f3/data/tau2/domains/retail)의 모의 주문 과제 114건과 고객 500명·주문 1,000건·상품 종류 50개의 DB, 원래 정책·도구를 사용합니다. 고객과 연결 주문을 53개 묶음으로 나눴으며 Train/Dev/Test는 **61/28/25건**입니다. 공식 분할에서 겹치던 고객 22명은 재분할 내역에 기록했습니다.

사용자 simulator가 목표를 읽고 모든 방법에 같은 첫 발화를 제공합니다. 각 방법은 새 DB에서 시작해 고객 확인·변경 제안·승인·도구 실행을 진행합니다. 최종 상태와 목표 응답, 승인 위반, 결제 원장 차이와 상호작용 비용을 채점합니다. 출처·파일 hash는 [`configs/orders.json`](configs/orders.json)과 [manifest](cases/orders/manifest.json)에 있습니다.

## CPU 학습과 평가 재현

포함된 가중치를 사용하는 GPU 평가만 진행한다면 이 단계는 건너뛸 수 있습니다. CPU 환경에서 새로 학습하려면 다음 순서로 실행합니다.

```sh
uv sync --frozen

# snapshot 복원: 파일 hash와 두 워크로드의 분할도 함께 검사
uv run newsvendor restore-eval

# 각각의 검사 결과를 다시 확인할 때
uv run newsvendor check-suite
uv run newsvendor check-orders

# seed 42–46의 12개 head 학습 → Test 평가 → 가중치 export
uv run newsvendor train-native --config configs/native.json
```

`train-native`는 `results/native/`와 Git에 추적된 `models/native/`를 갱신합니다. 설정은 [`configs/native.json`](configs/native.json)에 고정돼 있습니다.

| 항목 | 설정 |
|---|---|
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; 가중치 고정 |
| 표현 | 384차원; 전체 텍스트를 256-token 조각으로 인코딩한 뒤 토큰 수로 가중 평균 |
| 학습 | seed 42, 43, 44, 45, 46; 일반 head 25 epochs, 수요 head 100 epochs; Dev loss로 checkpoint 선택 |
| 최적화 | AdamW, 학습률 0.003, weight decay 0.0001, batch 128, gradient clip 5 |
| 손실 | 분류 CE + 0.1 × Brier, 후보 선택 CE + 0.1 × 기대 Huber, 수요 분포의 비품절 density·품절 survival NLL |
| Head | 은닉층 32; CUAD/NLI/규칙 상태, 근거, ABCD 발화/도구, TAT-QA 계산/scale, 7일 총수요 분포 |
| 후보 | 초기 근거 조각 12개, 계산·행동 후보 최대 48개 |
| 채점 | 각 seed에서 Test 366건, 원래 과제별 지표와 원시 예측 보존 |

현재 커밋에 포함된 CPU 결과는 다음과 같습니다. 각 수치는 5개 seed 평균입니다.

| 지표 | 값 |
|---|---:|
| CUAD 상태·근거 결합 | 0.527 |
| ContractNLI 상태 정확도 / 상태·근거 결합 | 0.578 / 0.227 |
| OR-ShARC 의사결정 정확도 | 0.558 |
| ABCD 도구 선택 / 관측 인자까지 정확 | 0.246 / 0.000 |
| TAT-QA 답·scale 모두 정확 | 0.017 |
| FreshRetail 총수요 분포·발주 지표 | 아래 수요 분포 실험 참조 |

TAT-QA Train 144건 중 정답 계산이 후보에 포함된 사례는 50건입니다. 후보 생성 실패와 선택 오류를 분리해 분석합니다. ABCD는 발화 여부와 조건부 도구 선택을 분리했으며 GPU 단계에서 선택 뒤 인자·발화 구성의 효과를 측정합니다. CPU 수치는 adapter 개발 과정에서 얻은 결과입니다.

### 수요 분포와 발주 실험

[`demand.py`](src/newsvendor/demand.py)의 수요 모형은 **분포 종류 head와 종류별 모수 head**로 구성됩니다. 분포의 대상은 **다음 7일 총수요**입니다. 각 head는 `28 → 32 → 3` MLP이며 1,027개 파라미터, 수요 head 4개 합계는 **4,108개**입니다.

| Head | 출력 | 변환 |
|---|---|---|
| `demand_family` | 절단 정규·로그정규·Weibull의 종류 점수 3개 | softmax → argmax로 종류 선택 |
| `demand_truncated_normal` | 0수요 확률·정규 위치·정규 표준편차 | sigmoid·softplus; 정규분포를 0 이상으로 절단 |
| `demand_lognormal` | 0수요 확률·로그 평균·로그 표준편차 | sigmoid·로그 평균·softplus |
| `demand_weibull` | 0수요 확률·형상·척도 | sigmoid·softplus·softplus |

`prediction.distribution`에는 선택한 `family`, `familyProbabilities`, 해당 종류의 `parameters`, `normalizationScale`이 모두 포함됩니다. 모수는 `D / normalizationScale`의 분포를 나타내며 `F`는 원래 판매 단위로 환산합니다.

입력은 직전 14일 판매, 7/14/30/60일 평균·표준편차·품절 비율, 최근 할인과 판매 규모입니다. 기간 총판매를 `7 × 과거 일평균 판매`로 정규화해 학습하며 일평균의 하한은 0.1입니다. 비품절 0판매는 0수요 질량, 양의 비품절 판매는 density NLL, 품절 기간은 `−log P(D ≥ 관측 총판매)`로 학습합니다. 채점의 density NLL에는 원래 판매 단위로 돌아가는 Jacobian을 반영합니다.

세 모수 head는 전 학습 동안 같은 관측의 NLL을 동일 가중치로 학습합니다. 처음 10 epochs 이후 종류 head는 각 후보의 관측 NLL을 softmax로 가중한 값을 낮추도록 학습합니다. 이때 모수 head로 가는 해당 경로의 gradient는 끊어, 선택되지 않은 종류도 계속 같은 데이터에서 학습되게 합니다. **종류 선택과 모수 추정은 함께 학습**하고 checkpoint는 실제 argmax 종류와 해당 모수의 Dev NLL로 선택합니다.

선택한 분포를 64개 분위 구간의 조건부 평균으로 이산화합니다. 0수요를 포함한 **65개 `[수요, 확률]` 쌍이 `prediction.F`**이며, 이 과정은 선택한 분포의 평균과 전체 확률을 보존합니다. 기존 Newsvendor solver가 이 `F`를 받아 발주량을 계산합니다. 출력의 `orders`는 잉여 비용 1, 부족 비용 1·3·9를 적용한 세 공개 실험 조건입니다. 비용은 정규화 판매 단위에 적용합니다.

원래 매장·상품 묶음 분할에서 최소 28일의 과거를 조건으로 7일 rolling window를 만듭니다. **Train 810 / Dev 216 / Test 324개 기간**, Test는 출처 묶음 10개입니다. Test 324개 중 비품절 28개에서 CRPS·총수요 MAE·80% 구간 포함률·실측 발주 손실을 계산하고, 품절 296개에서는 survival NLL과 발주 손실 하한을 기록합니다. 겹치는 기간은 출처 묶음별 평균도 함께 보고합니다. 기존 suite의 마지막 7일 Test 12건은 `metrics.json`에, 전체 rolling 평가는 `{seed}.demand.json`에 있습니다.

5개 seed의 rolling Test 평균은 다음과 같습니다. 발주 손실은 **출력에 실제 기록된 발주량**을 사용하며, 누락·잘못된 비용 조건·`F`와 맞지 않는 예상 손실은 `orderValid`로 집계합니다.

| 지표 | 기간 평균 | 출처 묶음 평균 |
|---|---:|---:|
| 관측 likelihood NLL | 0.4093 | 0.3682 |
| 비품절 CRPS | 7.1025 | 7.4979 |
| 비품절 발주 손실, 부족:잉여 3:1 | 17.4319 | 19.6373 |
| 비품절 80% 구간 포함률 | 21.4% | 51.6% |

각 seed의 Test 324기간은 모두 로그정규를 선택했습니다. `familyCounts`와 각 기간의 종류 점수·모수를 원시 결과에 보존합니다.

```sh
# 수요 head를 포함한 CPU 학습·채점·가중치 export
uv run newsvendor train-native --config configs/native.json

# 저장된 수요 분포 평가와 원시 측정 확인
python -m json.tool models/native/42.demand.json
```

```python
from newsvendor.demand import predict, order
from newsvendor.io import lines
from newsvendor.native_model import load_portable

heads, _ = load_portable("models/native", 42)
row = next(r for r in lines("data/processed/complementary/inputs.jsonl")
           if r["component"] == "retail" and r["split"] == "test")
prediction = predict(row["input"], heads)
F = prediction["F"]
print(prediction["distribution"])
print(prediction["period"], len(F), sum(p for _, p in F))
# 실제 계약 모수의 부족 비용 p-c+b, 잉여 비용 c-v도 전달할 수 있습니다.
print(order(F, underage=3, overage=1))
```

## GPU 비교 실행

### 구성 요소 비교

```sh
uv run --no-project scripts/run_gpu.py --stage components
```

같은 Test 366건에서 고정 typed·agent와 5개 seed 학습형을 비교합니다. FreshRetail에서 typed·agent는 같은 seed-42 수요 분포 head를 공통 도구로 사용하고, 학습형은 각 seed의 head를 사용합니다. 수요 분포와 발주 평가는 CPU에서 완료하며 GPU에서는 원문 처리·조회·도구 선택을 비교합니다.

| 방법 | 처리 순서 |
|---|---|
| 고정 typed | 조회 계획 → 공통 도구로 최대 2회 조회 → 구조화 추출·답변 |
| 학습형 router | 고정 encoder·학습 head로 근거/상태/도구/계산 선택 → Qwen으로 인자·질문·발화 표현 |
| Agent | 관측 결과에 따라 최대 2회 조회 → 구조화 답변 |

세 방법에 같은 원문 접근·관측 이력·초기 근거 조각·계산 후보를 제공합니다. 학습형의 언어 보조는 선택된 도구·상태·근거·계산 값을 유지합니다. Qwen2.5-7B-Instruct revision은 `a09a35458c702b33eeacc393d103063234e8bc28`이고, 문맥 상한은 32,768 token, 새 출력 상한은 512 token입니다.

### 주문 대화 비교

```sh
uv run --no-project scripts/run_gpu.py --stage orders
```

주문 Test 25건에서 typed·learned·agent를 실행합니다. Learned는 seed-42 ABCD head를 전이하고 같은 Qwen으로 인자·발화를 구성합니다. 사용자 simulator와 사후 목표 judge도 같은 고정 모델을 사용합니다. 방법별 상한은 assistant 40턴·도구 호출 20회입니다.

고객 identity와 정확한 변경 제안에 대한 확인을 공통 validator로 검사합니다. 최종 DB는 참고 행동을 재생한 DB와 비교하고, 조회 과제는 목표 응답과 필수 이관을 확인합니다. `orders-105`의 실패한 참고 변경은 manifest와 원시 판정에 표시합니다.

### 나눠 실행하고 이어가기

```sh
# 모든 단계를 한 번에 실행
uv run --no-project scripts/run_gpu.py --stage all

# 이번 실행의 새 생성 수만 제한
uv run --no-project scripts/run_gpu.py --stage components --max-calls 100

# 같은 명령을 다시 실행: 기존 생성은 cache에서 읽고 다음 100회 진행
uv run --no-project scripts/run_gpu.py --stage components --max-calls 100
```

`--max-calls N`은 API 호출이 아닌 **새 모델 생성 횟수**이며 양의 정수입니다. 생략하면 전체 단계를 진행합니다. 한도에 도달하면 `Generation budget exhausted`로 중단하며 그때까지의 생성은 `.cache/native-generation/`에 남습니다. 다시 실행하면 처음부터 순회하면서 기존 생성을 재사용합니다. 단계 완료 후 지표와 paired 결과가 생성되므로, 재개할 때 `.cache/`와 `results/`를 보존합니다.

## 결과 파일과 비교 방법

| 결과 | 위치 | 확인할 내용 |
|---|---|---|
| 포함된 CPU 결과 | `models/native/{seed}.metrics.json`, `runs.json` | 5개 seed의 과제별 점수 |
| 수요 분포·발주 평가 | `models/native/{seed}.demand.json`, `{seed}.demand.measurements.jsonl` | 324기간의 분포·비용별 손실과 재현 가능한 혼합분포 모수 |
| 재학습 CPU 결과 | `results/native/{seed}/` | `model.pt`, `training.json`, `predictions.jsonl`, `measurements.jsonl`, `metrics.json`, `provenance.json` |
| 학습형 GPU 결과 | `results/native/{seed}/gpu/` | seed별 예측·trace·원시 측정·지표 |
| Typed·agent 구성 요소 결과 | `results/native/{typed,agent}/` | 원시 예측·조회·token·지연·지표 |
| 구성 요소 간 짝 비교 | `results/native/paired.json` | 과제별 learned−baseline 차이와 95% CI |
| 주문 대화 | `results/orders/{typed,learned,agent}/` | `episodes.jsonl`, `metrics.json`; 상태·목표·승인·턴·도구·token |
| 주문 대화 짝 비교 | `results/orders/paired.json` | 고객 묶음별 learned−baseline 차이와 95% CI |
| 전체 실행 정보 | `results/gpu-run.json` | 실행 단계·새 생성 수·고정 모델 |
| 원시 모델 생성 | `.cache/native-generation/` | 실제 입력·출력·모델 revision·token·지연·메모리 |

```sh
python -m json.tool results/native/paired.json
python -m json.tool results/orders/paired.json
```

공개 과제는 사례 안에서 학습형 5개 seed를 먼저 평균한 뒤 출처 묶음별 paired bootstrap을 수행합니다. 주문 대화는 고객 묶음 단위로 비교합니다. `learnedMinusBaseline`은 정확도·성공률에서는 양수, MAE·위반·질문·token 수에서는 음수일 때 개선입니다. 구성 요소 지표와 주문 목표 달성, 결제 원장 차이(모의 USD), 처리 비용을 각각 읽습니다.

## 통제 Newsvendor 실험

6개 상황 × 20개 독립 업무 묶음 × 5개 문장·배치 변형의 600건을 생성합니다. 독립 참모수 조합은 120개이며 Train/Dev/보정/Test는 360/60/60/120건입니다. 같은 `dataSeed=42`로 문서·모수·분할을 유지하고 학습 seed만 바꿉니다.

```sh
# 데이터만 생성: data/synthetic/{inputs,labels}.jsonl, manifest.json
uv run newsvendor prepare --config configs/full.json

# 30개 서로 다른 업무 묶음의 수치·순차 실행 검사
uv run newsvendor smoke --config configs/full.json

# 단일 seed 학습·평가
uv run newsvendor experiment --config configs/full.json

# 저장된 checkpoint 재평가
uv run newsvendor evaluate --config configs/full.json --checkpoint results/revised/full/model.pt

# 같은 데이터에서 seed 42–46을 재학습·평가
uv run newsvendor sweep --config configs/full.json
```

단일 seed의 출력은 `results/revised/full/`, sweep은 `results/seeds/{seed}/`입니다. 입력·정답·manifest, checkpoint, 학습·보정, trajectory, CSV와 묶음별 비교를 저장합니다. 구성 head는 근거·종류·상태·계산 관계를, value/recovery head는 후속 손실과 복구 행동을 학습합니다. 통제 planner는 자기 예측 상태의 rollout 학습 표적과 수치 진단에 사용합니다. 정책은 고정 목록·모든 누락 요청·불확실성 우선·학습형을 같은 구성 안에서 비교합니다.

## 원천 재수집과 추가 실험

새로 원천 자료를 수집·가공할 때는 별도 checkout에서 다음 명령을 사용합니다. 현재 snapshot 재현에는 `restore-eval`을 사용합니다.

```sh
uv run newsvendor prepare-suite --config configs/complementary.json
uv run newsvendor prepare-orders --config configs/orders.json
uv run newsvendor pack-eval
```

수집 설정과 원천 revision·라이선스는 `configs/`에, 실제 수집 hash와 분할은 각 manifest에 기록합니다. 원래 대용량 다운로드는 `data/raw/`에 보존합니다.

Jev·SGLang endpoint나 OpenAI-compatible agent를 별도로 비교하려면 `.env.example`의 URL·실제 model·key를 설정합니다. 외부 `public` 진단 자료는 `fetch`로 준비합니다.

```sh
uv run newsvendor fetch
uv run newsvendor check
uv run newsvendor external --provider sglang --suite public --limit 20
uv run newsvendor benchmark --provider sglang --config configs/pilot.json --max-calls 1000
uv run newsvendor raw-benchmark --provider agent --config configs/pilot.json --limit 14 --max-calls 100
```

이 경로의 `--max-calls`는 새 HTTP 호출 상한이며 기본값 0은 cache만 사용합니다. `benchmark`는 고정 typed 구성과 학습형 정책을 연결하는 통제 비교이고 `raw-benchmark`는 원문 개발 사례에서 직접 구성·행동을 실행합니다.

구현을 변경한 뒤 기존 검증을 실행하려면 `uv run pytest -q`를 사용합니다. 자료 분할은 `check-suite`·`check-orders`, 수치 계산은 `smoke`로 확인합니다.

[데이터·모델 출처](docs/sources.md) · [실험 명세](docs/protocol.md) · [프로포절](docs/proposal.docx)
