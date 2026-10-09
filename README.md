# Newsvendor Decision Router

문서와 판매 이력에서 **발주 모수와 근거를 구성하고, 부족한 값은 매니저에게 질문하는 작은 모델**의 연구 저장소입니다. 연구 모형의 원안은 [프로포절](docs/proposal.docx)에 보존합니다.

2026-10-09 주 연구를 **ModernBERT + 작은 수요 GRU + 모수·근거·상태 head + 소수의 정보 요청 행동**으로 집중했습니다. 문서와 판매 이력에서 Newsvendor 입력을 구성하고, 부족한 모수를 매니저에게 물어 갱신한 뒤 공통 optimizer로 발주량을 계산합니다. 주 실행은 `scripts/run_research.py`이며 범용 ABCD 도구·업무 절차 모듈은 포함하지 않습니다. 최신 결과·평가 분모·남은 과제는 [연구 범위와 검증 기록](docs/research-scope.md)을 기준으로 읽습니다. 범용 도구 확장·ABCD 결과는 [보조 실험 이력](docs/structured-history.md)에 보존합니다.

## 주 모델과 현재 검증

모델은 문서에서 경제 모수 `c/p/v/b`와 근거를 구성하고, 판매 이력에서 수요분포 `F`를 추정합니다. 필요한 값이 없거나 충돌하거나 매니저의 선택이 필요하면 해당 항목을 질문합니다. 응답으로 상태를 갱신한 뒤 공통 optimizer가 발주량을 계산합니다. 문장 생성은 요구하지 않습니다.

| 구성 | 역할 |
| --- | --- |
| 학습 가능한 ModernBERT | 원문 토큰·표 cell에서 값과 적용 조건 추출 |
| 모수·근거·유형·상태 head | 사실·추정·선호를 구분하고 누락·충돌 및 출처 보존 |
| 2층 GRU, hidden 128 | 판매·품절·달력 이력에서 1일/7일 수요분포의 종류·모수 추정 |
| 정보 요청 head | 추가 근거 조회, 모수별 매니저 질문, 계산 진행 또는 보류 |
| 계산기·optimizer | 선택된 식을 실행하고 실제 전달된 `F/c/p/v/b`로 발주량 계산 |

행동 가치 head는 같은 구성기에서 단순 질문 정책 및 가치 학습을 제거한 `no_value`와 비교합니다. 판매 이력에 연결한 통제 Dev에서는 경제 가치 학습의 추가 이점이 확인되지 않았습니다. 범용 업무 절차·도구 후보 재평가·ABCD controller는 주 경로에 포함하지 않습니다.

동일 head·자료·학습 횟수를 사용한 L40S base/large 비교는 종료됐습니다. 전체 크기는 151.85M/397.68M이며, 통제 Dev의 최종 모수·근거 정확도는 모두 100%, 완전 관측 기간의 평균 총손실은 모두 44.3274였습니다. 불필요한 질문은 base 3건, large 0건이었습니다. 정확한 손실은 **5개 고유 판매 기간 × 7개 통제 조건**에 한정하며, 문서와 매니저 응답의 통제 생성 조건을 실제 조직 효과로 해석하지 않습니다. 공개 문서 숫자 추출과 수요분포 품질에는 개선이 남아 있습니다. [비교 요약·원시 파일 hash와 지연·메모리](docs/evidence/research-paired-results.json)

확장 문서 Train을 사용한 후속 base 학습도 완료됐습니다. 같은 TAT-QA Dev의 엄격 정답은 **0/48 → 11/48**이지만, ContractNLI 상태 정확도는 **65.28% → 62.50%**로 낮아졌습니다. [후속 문서 학습 결과](docs/evidence/research-documents-expanded-results.json)

확장 수요 자료의 seed 42·43·44 학습도 완료됐으나, 주간 NLL·CRPS가 모두 악화해 기존 GRU를 유지했습니다. 수요 분포 보정은 미해결입니다. [확장 수요 비교](docs/evidence/research-demand-expanded-results.json) · [원래 생성 Dev 결과](docs/evidence/research-generated-dev-results.json)

원래 생성 Train과 실제 매니저 응답 상태를 복구한 다음 학습에서는 TAT-QA 엄격 정답이 **16/48**, 생성 Dev 평균 총손실이 **147.6813**이었습니다. 생성 Dev 60건·12개 family의 결과이며 보호된 Test 목표 달성으로 해석하지 않습니다. 새 가치 학습 후보는 채택되지 않아 구성기 재학습 이후의 개선으로 기록합니다. [재학습 결과](docs/evidence/research-replay-results.json) 같은 구성기의 `no_value`와 고정 질문 정책은 생성 Dev에서 181.6758이었습니다. [정책 비교](docs/evidence/research-replay-no-value-results.json)

이후 가중치를 고정한 **생성 Test 120건의 총손실은 243.6274**로 목표 240.35 미만에 미달했습니다. `no_value`와 고정 질문은 263.2379였습니다. 판매 이력에 연결한 Test 504사례의 모수·근거는 100%, 불필요한 질문·잘못된 발주 진행은 0건이지만, 정확한 손실 비교는 고유 기간 9개에 한정됩니다. 수요 예측 품질도 여전히 미해결입니다. [최종 고정 Test 결과와 제한](docs/evidence/research-core-test-results.json)

후속 Train 감사에서 행동 head의 텍스트 입력이 잘리면서 현재 모수 일부가 전달되지 않는 결함을 확인했습니다. 수치 상태를 직접 전달하는 선택 경로와, 작은 수치 차이 및 학습·추론 계산을 보존하는 FP32 행동 head를 추가했습니다. 실제 L40S의 수치 검증은 통과했으나 동일 예산 정책 비교에서는 추가 Dev 개선이 없어 기존 선택 모델을 유지합니다. [입력 수정](docs/policy-state.md) · [정밀도 수정과 세 조건 비교](docs/action-precision.md)

학습 표적도 점검했습니다. 추가 응답 잡음을 기본 평가와 맞추는 조건과, 행동별 8회 실제 응답 경로의 평균 손실을 학습하는 조건을 L40S에서 비교했습니다. 두 조건 모두 추가 Dev 개선이 없어 기존 모델을 유지합니다. Train 응답 seed·원시 손실·후보별 결과와 제한은 [응답 표본과 가치 학습](docs/response-supervision.md)에 정리했습니다.

## 새 clone에서 실행

Python 3.12와 uv를 사용합니다. 아래 경로는 저장소에 포함된 자료와 고정 revision의 공개 ModernBERT로 **새 수요 GRU부터 학습**합니다. 이전 로컬 checkpoint는 필요하지 않습니다. 기존 실험의 가중치를 복원하는 명령은 아니므로 과거 표의 수치와 동일하다고 보장하지 않습니다.

```sh
git clone https://github.com/mrcha033/newsvendor-router.git
cd newsvendor-router
uv sync --frozen
uv run newsvendor restore-eval

# 모델 다운로드 없이 입력·출처 분리 검사
uv run python scripts/run_research.py \
  --config configs/research-from-scratch.json \
  --stage check --output results/research-data-check
```

학습은 사용할 L40S의 UUID를 `NEWSVENDOR_GPU_UUID`에 지정한 뒤 실행합니다. Linux의 `uv sync`는 CPU PyTorch 환경입니다. 아래 `--no-project` 명령은 script에 고정된 CUDA PyTorch 환경을 사용합니다.

```sh
export CUDA_VISIBLE_DEVICES="$NEWSVENDOR_GPU_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 1. 판매 Train에서 GRU·분포 모수를 학습하고 Train 내부 cross-fitting으로 종류 선택 학습
uv run --no-project scripts/train_demand.py \
  --config configs/research-from-scratch.json --device cuda

# 2. 고정 revision의 ModernBERT를 불러와 encoder·모수 구성기·요청 정책 학습
uv run --no-project scripts/run_research.py \
  --config configs/research-from-scratch.json --stage train
```

[`research-from-scratch.json`](configs/research-from-scratch.json)은 원본 snapshot의 문서·판매 Train과 원래 생성 Train을 사용합니다. Dev로 선택하며 Test는 학습·모델 선택에 사용하지 않습니다. `results/research-from-scratch/demand/model.pt`에 GRU가, `results/research-from-scratch/base/42/model.pt`에 전체 모델이 저장됩니다. 두 단계는 기존 실행을 덮어쓰지 않습니다. 공개 snapshot에서 GRU 학습과 전체 모델 연결·encoder 갱신까지 확인했습니다. [재현 경로 검증](docs/evidence/research-reproduction-checks.json) CPU에서 GRU만 학습하려면 `uv run python scripts/train_demand.py --device cpu`를 사용할 수 있습니다.

현재 주간 수요 학습은 품절일이 있는 합계를 하한으로 취급합니다. 이 점수는 날짜별로 검열된 관측의 정확한 likelihood가 아닐 수 있으며 편향을 보이는 수치 예제를 확인했습니다. 재현 경로를 추가한 것이 수요 모형의 결함을 해결한 것은 아닙니다. [관측 모형 수치 검사](docs/evidence/research-aggregate-censoring-check.json)

날짜별 정확 관측과 품절을 구분하는 새 likelihood도 구현하고 L40S에서 seed 3개를 비교했습니다. 수치 검증은 통과했으나 사전 채택 조건에 미달해 기존 GRU와 기본 설정을 유지합니다. 이 후보는 선택 옵션으로 보존하며 Test로 선택하지 않았습니다. [관측 모형과 비교 결과](docs/demand-observations.md)

## 공개 자료와 재현 범위

| 항목 | 저장소 제공 범위 |
| --- | --- |
| 원본 상보적 과제·주문 과제 | [압축 snapshot](cases/evaluation.tar.xz), [파일 hash·revision·라이선스](cases/evaluation.json); `restore-eval`로 복원 |
| 원래 통제 시나리오 | [생성 설정](configs/full.json)과 [생성 코드](src/newsvendor/corpus.py); 600건, 출처 family 120개 |
| 현재 ModernBERT·GRU | 학습·추론 코드와 고정 설정 제공. 과거 실행의 checkpoint 자체는 아직 공개되지 않음 |
| 확장 문서·판매 자료 | [문서 확장 기록](docs/evidence/research-document-expansion.json), [판매 확장 기록](docs/evidence/research-retail-expansion.json), [분할 검사](docs/evidence/research-expansion-audit.json). 기본 snapshot에는 미포함 |
| 실험 요약과 검증 | [연구 범위](docs/research-scope.md), [추적되는 evidence JSON](docs/evidence/). JSON 안의 `results/` 경로는 로컬 원시 파일의 위치와 hash이며 다운로드 링크가 아님 |
| 과거 native-v3 | [CPU 가중치와 원시 결과](models/native/), [이전 실행 안내](docs/native-history.md). 현재 ModernBERT의 가중치가 아님 |

따라서 새 학습은 저장소만으로 시작할 수 있지만, **기존 ModernBERT 결과의 정확한 checkpoint 재평가와 확장 자료 실험의 완전 재현은 아직 제공하지 못합니다.** 주 결과의 작은 표본, 생성된 문서 조건, 미공개 원시 파일도 결과의 한계로 남깁니다. 공개 snapshot과 기존 작업 자료는 retail 요청 문구·manifest가 달라 이전 checkpoint 신원 검사에서 거부되며, 판매 관측·정답·분할이 같다는 [대조 기록](docs/evidence/research-snapshot-audit.json)을 남겼습니다. 같은 head의 base/large 비교는 과거 고정 실험 기록이며, 새 코드로 실행한 수치로 덮어쓰지 않습니다.

## 검증과 이전 실험

```sh
uv run pytest -q
uv run ruff check src tests
uv run newsvendor check-suite
uv run newsvendor check-orders
uv run newsvendor smoke --config configs/full.json
```

CI는 테스트·Ruff·snapshot 복원과 출처 검사·주 연구 입력 검사·수치 smoke를 실행합니다. GPU 학습과 성능 검증은 CI에 포함하지 않습니다. [CI 설정](.github/workflows/ci.yml)

이전 ABCD 90% 목표와 실패 결과는 [구조화 모델 실험 이력](docs/structured-history.md)에, MiniLM·Qwen·native-v3의 실행 방법과 수치는 [과거 실험 안내](docs/native-history.md)에 보존합니다. 기존 생성 Test의 총손실 240.35 기준은 원래 분할과 손실 정의에만 적용하며 retail 손실과 직접 비교하지 않습니다.

[데이터·모델 출처](docs/sources.md) · [연구 범위와 최신 측정](docs/research-scope.md) · [기존 실험 명세](docs/protocol.md)
