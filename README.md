# Newsvendor Decision Router

업무 문서의 **근거 연결·모수 상태·행동 가치**를 직접 학습하는 Decision Router의 실험 저장소입니다. 사용자가 수정한 [프로포절](docs/proposal.docx)을 기준으로 Python·PyTorch 환경을 구성했습니다.

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 사용합니다. 첫 실행에서 공개 pretrained encoder를 내려받습니다. 로컬 실험에는 API key가 필요하지 않습니다.

```sh
uv sync --frozen
uv run pytest -q
uv run newsvendor smoke
uv run newsvendor fetch
uv run newsvendor check
uv run newsvendor audit-workload --config configs/full.json
uv run newsvendor check-workload --config configs/workload-review.json
uv run newsvendor experiment --config configs/pilot.json
```

기본 full은 120개 독립 참모수 묶음의 문장/배치 변형 600건입니다. Train/Test 참모수 중복은 0이며, 구매/판매 비율·5점 수요 분포·질문 비용·응답률을 묶음별로 변화시킵니다. 일반 문장에서 숫자와 단위의 span을 추출합니다. 5개 학습 seed는 동일한 `dataSeed`와 분할을 사용합니다.

```sh
uv run newsvendor experiment --config configs/full.json
uv run newsvendor evaluate --config configs/full.json --checkpoint results/revised/full/model.pt
uv run newsvendor sweep
uv run newsvendor doctor
```

`results/<실행>/`에 checkpoint, 학습 손실, calibration, 원시 trajectory, CSV, source 단위 bootstrap 결과와 보고서를 저장합니다. 원본 데이터와 모델 cache는 Git에서 제외합니다. source·설정·의존성 lock·encoder revision의 hash를 실행마다 기록합니다.

## 구성

| 구성 요소 | 역할 | 구현 |
|---|---|---|
| Pretrained encoder | 문서·slot·행동을 384차원 표현으로 변환; 가중치 고정 | `encoder.py` |
| Construction heads | 근거 연결, 사실/추정/선호/가정, 확인 상태, 식 후보를 예측 | `construction.py`, `heads.py` |
| Validator / calculator | SKU·기간·단위·선택 권한을 검사하고 원문 수치로 계산 | `construction.py` |
| Newsvendor optimizer | 유한 공동 모수 집합에서 정확한 minimax regret 계산 | `optimizer.py` |
| 통제 planner | 통제 환경의 rollout 학습 표적과 수치 진단 | `policy.py` |
| Value / recovery heads | 행동별 후속 손실과 복구 행동을 학습 | `train.py` |
| Updater / action selector | 관측 응답을 반영해 상태를 다시 구성하고 허용 행동 선택 | `corpus.py`, `policy.py` |

주 비교 대상은 같은 원문부터 처리하는 **고정 typed 파이프라인과 도구 사용 agent**이며, 우리 학습형 router를 이들과 비교합니다. 규칙 기반·모형 기반 planner는 주요 베이스라인에서 제외합니다. 고정 typed 구성에 학습형 정책을 연결하는 조건은 구성 기여를 분리하는 ablation입니다. Validator·calculator·optimizer는 모든 방법이 공유하는 도구입니다.

통제 실행은 고정 목록, 모든 누락 요청, 불확실성 우선, 학습형 정책을 규칙/학습형 구성과 교차하는 진단입니다. Planner·reference expert·one-step·oracle은 기본 평가에서 실행하지 않습니다. 통제 planner는 생성기 응답 모형을 사용하는 학습 표적 및 수치 진단에 사용하며, 주 비교의 승패 기준으로 사용하지 않습니다.

참모수의 `Ω` 포함률과 발주 승인 가능 여부를 따로 채점합니다. 미해결 계약 충돌·검열 수요·마감/수량 위반이 있는 전달은 오류입니다. 과거 결과는 원래 설정과 인접 `metrics.json`을 확인하고 기록된 응답으로 다시 채점합니다: `newsvendor rescore --config <원래설정> --trajectories <원시기록>`.

## 외부 비교

`.env.example`을 `.env`로 복사해 endpoint와 실제 model 이름, 필요한 key를 설정합니다. 값은 Git에 올리지 않습니다. Jev와 SGLang은 `/v1/systemone` choice 형식, agent는 OpenAI-compatible chat 형식을 사용합니다.

```sh
uv run newsvendor external --provider sglang --suite public --limit 20
uv run newsvendor external --provider jev --suite business --config configs/pilot.json --limit 12
uv run newsvendor external --provider agent --suite business --config configs/pilot.json --limit 12
uv run newsvendor benchmark --provider sglang --config configs/pilot.json --max-calls 1000
uv run newsvendor raw-benchmark --provider agent --config configs/pilot.json --limit 14 --max-calls 100
```

`benchmark`는 주석된 통제 입력의 공통 후보에서 기존 예측기의 evidence/type/state/expression을 구성하고, 자기 상태의 planner 표적으로 value/recovery를 학습합니다. 네 정책을 규칙/기존 예측기 구성과 교차하는 진단이며, paired 비교는 같은 구성의 정책끼리 수행합니다. `raw-benchmark`는 원문·실제 출처·동일 도구부터 typed/checklist와 agent를 실행합니다. 후보는 공통의 주석 없는 calculator가 만들며 적용 범위와 상태는 각 예측기가 판단합니다. Reference 상태·미래 응답·응답 확률은 제공하지 않습니다.

원시 예측·호출 hash·usage·지연을 cache와 결과에 남깁니다. `--max-calls`는 새 HTTP 호출의 상한이고 기본값 0은 cache만 사용합니다. 예시의 상한은 완료 호출 수나 요금의 보장이 아닙니다.

외부 실행은 직접 명령을 내렸을 때만 API를 호출합니다. 모델·endpoint가 없으면 실행 전에 중단합니다. 실제 Jev/SGLang/agent 측정은 아직 없습니다. 로컬 서버와 명시된 protocol fixture 검사는 실제 모델 성능 측정에 포함하지 않습니다.

## 상보적 영어 워크로드

`configs/complementary.json`은 영어 원문을 보존하는 6개 구성 요소를 준비합니다. 실제 계약·재무 문서와 판매 관측을 사용하며, 질의·도구 선택에는 공개된 사람 역할극 및 crowd 대화를 사용합니다. 출처가 다른 자료를 한 기업의 사건으로 결합하거나 없는 발주 정답을 생성하지 않습니다.

| 원천 | 준비 건수 | 역할과 원래 채점 표적 |
|---|---:|---|
| CUAD 공급·제조·유통·구매 계약 | 450 / 계약 75개 | 기간·가격 제한·최소 구매·수량·보증 조건의 근거와 미기재 |
| ContractNLI NDA | 360 / 계약 60개 | 함의·모순·미기재와 근거; 공급 조건 자료는 아님 |
| OR-ShARC | 240 / 규칙 묶음 60개 | 공통 규칙 651개에서 조회한 뒤 yes/no 또는 추가 질문 |
| ABCD | 462 / 대화 80개 | 전체 정책과 관측 대화에서 발화/도구 선택; 실제 고객 로그가 아닌 사람 역할극 |
| TAT-QA | 240 / 문서 context 60개 | 표·문장 계산 및 단위·scale |
| FreshRetailNet | 50개 시계열 | 과거 60일 → 미래 7일 관측 판매; 품절·비품절 오차를 따로 표시 |

```sh
uv run newsvendor prepare-suite
uv run newsvendor check-suite
uv run newsvendor retrieve-suite --query "electricity supplier help guarantee credit" --limit 5
uv run newsvendor score-suite --predictions predictions.jsonl --split test
```

공통 입력은 `newsvendor.suite.public_input(row)`로 얻습니다. 영어 request, 원문 documents, tables, 관측 history·sales, 도구 목록만 반환합니다. 정답·의도·근거 주석·미래 대화·정답 규칙 ID는 제외합니다. OR-ShARC의 올바른 문서는 입력에 미리 넣지 않고 공통 조회 도구로 찾습니다. ABCD에는 현재 정답 workflow를 선택해서 주는 대신 전체 정책을 제공합니다.

예측 JSONL은 `{"id":"<case id>","prediction":{"action":"answer","answer":12,"scale":"million"}}` 형식입니다. 행동은 `answer/ask/speak/call_tool/abstain`이며 근거는 `evidence:[{"document":"contract","start":0,"end":10}]`, 도구는 `tool`과 순서 있는 `arguments`, 조회 문서는 `retrieved`로 기록합니다. 학습은 Train/Dev 원래 주석만 읽고, Test 정답은 독립 채점기만 읽습니다. `native_inputs.py`가 주석 없는 원문 fragment·계산 후보를 만들고, `native_model.py`가 기존 PyTorch head를 원래 과제의 근거·상태·행동·계산·판매 예측에 적합합니다.

가공 입력·정답·조회 collection·manifest는 `data/processed/complementary/`에, 원시 응답은 `data/raw/complementary/`에 저장합니다. 같은 계약·규칙 page/tree·대화·매장·상품을 연결하고, 동일 입력과 수치 정규화 문서 및 유사 template를 묶어 60/20/20%로 분할합니다. 이 실행은 Train/Dev/Test 1,074/362/366건, 378개 출처 묶음입니다. TAT-QA는 전체 보고서 식별자가 없어 context 수준 분할이고, OR-ShARC의 주석 없는 조회 collection은 모든 방법에 공통으로 공개됩니다.

각 구성 요소의 점수와 묶음별 평균을 따로 보존합니다. 누락 예측은 분류 분모에 남고, 불완전한 판매 예측은 비교 가능 결과로 인정하지 않습니다. 기록된 질문·발화의 문구 유사도는 참고 지표입니다. 인과적 행동 가치·발주 손실·잠재수요는 이 공개 자료의 정답이 아니므로 해당 학습 표적은 만들지 않습니다. 준비 snapshot은 [manifest](cases/complementary/manifest.json)에 기록합니다.

CPU에서 5개 seed의 9개 head 학습과 Test 366건 평가를 실행했습니다. [가중치와 원시 결과](models/native/)를 커밋에 포함합니다. 초기 Test를 개발 중 확인했으므로 확증 실험이 아닌 탐색적 adapter 결과입니다.

| 원래 과제 지표 | 5개 seed 평균 |
|---|---:|
| CUAD 상태와 근거 결합 점수 | 0.527 |
| ContractNLI 상태 정확도 / 상태와 근거 결합 점수 | 0.578 / 0.227 |
| OR-ShARC 의사결정 정확도 | 0.558 |
| ABCD 도구 선택 정확도 / 관측 인자까지 정확 | 0.246 / 0.000 |
| TAT-QA 답과 scale 모두 정확 | 0.017 |
| FreshRetail 비품절 관측 판매 MAE | 0.509 |

ABCD는 발화 여부와 조건부 도구 선택을 분리했습니다. CPU 인자 추출은 제한된 공개 문법만 처리합니다. TAT-QA 계산 후보에 Train 정답 144건 중 50건만 포함되어 후보 생성과 선택 양쪽의 제약이 남습니다. 이 결과로 모델 우위를 주장하지 않습니다. GPU 실행은 같은 Qwen 7B를 사용해 typed·agent와 학습형 선택 후 인자/질문을 표현하는 조건을 측정하며, 선택된 도구·상태·근거·계산 값은 언어 helper가 바꾸지 못합니다.

## 연결된 주문 사례와 남은 GPU 실행

[τ²-bench retail](https://github.com/sierra-research/tau2-bench/tree/5bfa7e37b36656b37dc6d022156be6563c1007f3/data/tau2/domains/retail)의 영어 주문 운영 사례 114건, 고객 500명·주문 1,000건·상품 종류 50개의 연결 DB와 원래 정책/도구를 확보했습니다. 모두 모의 업무 자료이며 실제 발주·고객 로그가 아닙니다. 공식 Train/Test에는 고객 22명이 겹쳐 고객 묶음 53개로 다시 분할했습니다: Train/Dev/Test 61/28/25건. 공통 상품 catalog는 모든 방법에 공개됩니다.

사용자 목표는 응답 환경만 읽고 비교 모델은 실제 생성된 첫 발화·이후 관측·정책·조회 도구만 받습니다. 각 방법은 새 격리 DB에서 시작합니다. 최종 DB 일치, 미승인/다른 고객 변경, 거래 원장 차이, 상호작용·추론 비용을 따로 채점합니다. 조회 과제는 DB 무변경만으로 성공 처리하지 않고 실제 목표 응답과 필수 이관도 확인합니다. 원장 차이는 모의 USD 차이이며 경제적 행동 가치나 실제 발주 손실이 아닙니다. 원래 evaluator처럼 실패한 참고 조회 뒤에도 재생을 계속하며 경고를 보존합니다. 참고 변경 자체가 실패하는 `orders-105`는 제거하지 않고 주석 모호성을 표시합니다.

```sh
# CPU 환경: 준비된 정확한 snapshot 복원과 재학습
uv run newsvendor restore-eval
uv run newsvendor train-native
uv run newsvendor check-orders

# CUDA 머신, Python 3.12: Git에 포함된 데이터와 가중치로 남은 비교 실행
uv run --no-project scripts/run_gpu.py --stage all
# 개별 실행: --stage components 또는 --stage orders
```

프로젝트의 `uv sync`는 Linux에서 CPU PyTorch를 설치하므로 GPU 실행에는 위의 별도 script 환경을 사용합니다. 24GB 이상 GPU를 권장합니다. Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`을 고정하고 typed·agent와 학습형 출력 helper에 동일 가중치를 사용합니다. CUDA가 없으면 모델을 내려받기 전에 중단합니다. 토큰·도구·턴 상한과 원시 응답/cache를 보존하고 입력을 조용히 자르지 않습니다. `--max-calls N`으로 새 생성 수를 제한하고 같은 명령으로 cache에서 이어갈 수 있습니다.

구성 요소 비교는 학습 seed를 사례 안에서 먼저 평균한 뒤 출처 묶음별 paired bootstrap을 수행합니다. 주문 비교는 seed 42의 ABCD head를 재학습 없이 전이한 진단입니다. Typed·agent 및 주문 대화의 실제 GPU 결과는 아직 실행하지 않았습니다. 결과는 `results/native/{seed}/gpu`, `results/native/{typed,agent}`와 `results/orders/`에 저장합니다. 별도 원시 다운로드 없이 실행할 수 있도록 [평가 snapshot](cases/evaluation.json)을 포함하며 각 원천의 라이선스를 유지합니다.

## 실험 범위

현재 실행은 제한된 영어 업무 문서와 유한 모수 후보를 사용하는 **통제 실험**입니다. `c,p,v,b`의 숫자 근거와 `F`의 관측 조건을 다루며, 임의의 한국어 문서·다기간 재고·잠재수요 복원을 구현한 결과는 아닙니다. 자동 주석의 사람 검토 수는 0으로 기록합니다.

TAT-QA 200건, ShARC 200건, FreshRetailNet 50개 시계열을 준비합니다. 외부 public 진단은 각각 제한된 수치 후보 선택과 추가 질문 필요성 분류입니다. business 진단은 공통 규칙 모수 구성 위에서 기존 예측기의 직접 행동 선택을 검사합니다. **기존 예측기로 construction을 구성한 뒤 동일 학습 표적으로 value head를 적합하는 비교**는 `benchmark`로 실행합니다. 외부 모델을 확정한 뒤 cache·예측·API 비용을 실제 결과로 채워야 합니다.

원문 14개 개발 사례는 조회/사실 질문/선호 선택, 계약 충돌, 검열 판매, 마감, MOQ·반품 한도·다기간 조건을 검사합니다. 구성 자료 13묶음에 공통 template가 하나이며 사람 검토는 0입니다. 학습형 router의 raw 학습/평가 연결과 실제 조직 자료의 source/template/기간 분할은 아직 필요합니다. 준비 상태는 `configs/workload-v2.json`에 명시합니다.

제거 실험은 현재 추론 시 진단이며 재학습한 ablation 결과가 아닙니다. 본 연구의 효과를 주장하기 전에 [실험 명세](docs/protocol.md)의 사람 검토와 비교 조건을 충족해야 합니다.

[검증 기록](docs/verification.md) · [데이터·모델 출처](docs/sources.md)
