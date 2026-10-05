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
uv run newsvendor experiment --config configs/pilot.json
```

600개 episode를 사용하는 설정과 5개 seed 반복 실행도 준비되어 있습니다.

```sh
uv run newsvendor experiment --config configs/full.json
uv run newsvendor evaluate --config configs/full.json --checkpoint results/full/model.pt
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
| Reference planner | 당시 공개된 근거와 가능한 응답 분기로 학습 표적 생성 | `policy.py` |
| Value / recovery heads | 행동별 후속 손실과 복구 행동을 학습 | `train.py` |
| Updater / action selector | 관측 응답을 반영해 상태를 다시 구성하고 허용 행동 선택 | `corpus.py`, `policy.py` |

고정 목록, 모든 누락 요청, 불확실성 우선, 한 단계 영향, reference planner, 학습형 정책을 **규칙/학습형 모수 구성과 교차**하여 평가합니다. full-information oracle은 성과 상한을 위한 별도 평가용 결과입니다.

## 외부 비교

`.env.example`을 `.env`로 복사해 endpoint와 실제 model 이름, 필요한 key를 설정합니다. 값은 Git에 올리지 않습니다. Jev와 SGLang은 `/v1/systemone` choice 형식, agent는 OpenAI-compatible chat 형식을 사용합니다.

```sh
uv run newsvendor external --provider sglang --suite public --limit 20
uv run newsvendor external --provider jev --suite business --config configs/pilot.json --limit 12
uv run newsvendor external --provider agent --suite business --config configs/pilot.json --limit 12
uv run newsvendor benchmark --provider sglang --config configs/pilot.json --max-calls 1000
```

`benchmark`는 기존 예측기로 evidence/type/state/expression을 구성하고, 그 예측 상태에서 공통 reference planner 표적을 만들어 value/recovery head를 학습합니다. 규칙/기존 예측기 모수 구성과 여섯 요청 정책을 교차 평가합니다. 원시 예측을 hash와 함께 cache하므로 다시 실행할 때 이미 확보한 상태를 재호출하지 않습니다. `--max-calls`는 새 HTTP 호출 수의 상한이며 생략하면 cache만 사용합니다. 위 1,000회는 설정 예시이며 완료에 필요한 호출 수나 요금의 보장이 아닙니다.

외부 실행은 직접 명령을 내렸을 때만 API를 호출합니다. 모델·endpoint가 없으면 실행 전에 중단합니다. 호출 수·token usage·지연을 실제 응답으로 기록하며 비용 단가는 추정하지 않습니다. protocol과 학습 연결 검사는 로컬 서버·명시된 test fixture로 통과했지만 실제 외부 모델 결과는 아직 없습니다.

## 실험 범위

현재 실행은 제한된 영어 업무 문서와 유한 모수 후보를 사용하는 **통제 실험**입니다. `c,p,v,b`의 숫자 근거와 `F`의 관측 조건을 다루며, 임의의 한국어 문서·다기간 재고·잠재수요 복원을 구현한 결과는 아닙니다. 자동 주석의 사람 검토 수는 0으로 기록합니다.

TAT-QA 200건, ShARC 200건, FreshRetailNet 50개 시계열을 준비합니다. 외부 public 진단은 각각 제한된 수치 후보 선택과 추가 질문 필요성 분류입니다. business 진단은 공통 규칙 모수 구성 위에서 기존 예측기의 직접 행동 선택을 검사합니다. **기존 예측기로 construction을 구성한 뒤 동일 학습 표적으로 value head를 적합하는 비교**는 `benchmark`로 실행합니다. 외부 모델을 확정한 뒤 cache·예측·API 비용을 실제 결과로 채워야 합니다.

제거 실험은 현재 추론 시 진단이며 재학습한 ablation 결과가 아닙니다. 본 연구의 효과를 주장하기 전에 [실험 명세](docs/protocol.md)의 사람 검토와 비교 조건을 충족해야 합니다.

[검증 기록](docs/verification.md) · [데이터·모델 출처](docs/sources.md)
