# 선택 모델의 추론 가중치와 실행

[`research-forecast-v1` 공개 묶음](https://github.com/mrcha033/newsvendor-router/releases/tag/research-forecast-v1)은 기준 모델의 ModernBERT-base, 기능 head, 수요 GRU 전체 추론 가중치를 포함한다. 원래 checkpoint의 tensor 값을 그대로 보존했다. 최신 관측 실패 보정 모델은 아래처럼 이 공개 가중치에 작은 공개 head를 결합해 만든다.

묶음에는 encoder 설정·토크나이저·파일 hash manifest·관측 입력 예제 8개가 들어 있다. [전용 loader](../src/newsvendor/bundle.py)는 각 파일과 로드된 tensor의 hash를 검증한다. 원래 학습 자료의 로컬 경로나 Hugging Face 접속, 별도 사전학습 가중치 다운로드는 추론에 필요하지 않다. 기존 학습 checkpoint loader의 자료 검증은 유지한다.

## 실행

저장소 루트에서 Python 3.12·uv와 `curl`, `zstd`를 사용한다. 압축 파일은 약 564 MB다.

```sh
uv sync --frozen
mkdir -p results/released-model
curl --fail --location \
  https://github.com/mrcha033/newsvendor-router/releases/download/research-forecast-v1/forecast-v1.tar.zst \
  --output results/released-model/forecast-v1.tar.zst

echo '44869c17a4e2d41f4dd4cfe9331fa995e3f9e7e40b29cad21c02ec34e090fd79  results/released-model/forecast-v1.tar.zst' | sha256sum --check
zstd --long=31 -dc results/released-model/forecast-v1.tar.zst \
  | tar -xf - -C results/released-model

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/predict_research.py \
  --bundle results/released-model/forecast-v1 \
  --inputs results/released-model/example-inputs.jsonl \
  --output results/released-model/predictions.jsonl
```

각 JSONL 행의 `input`에 현재 관측한 과제·문서·판매 이력·응답 이력·남은 요청 횟수를 넣는다. 예제는 기존 Train에서 이미 관측한 상태이며 숨겨진 응답 확률과 평가 정답은 제거했다. 출력은 모수·근거·상태와 다음 행동을 담는다. 이 명령은 현재 입력마다 한 번 판단하며, 실제 매니저에게 메시지를 보내거나 다음 응답을 생성하지 않는다. 새 응답을 관측한 다음에는 갱신한 입력으로 다시 실행한다. 기존 출력 파일은 덮어쓰지 않는다.

CUDA PyTorch 환경에서는 `--device cuda`를 추가하고 사용할 GPU를 `CUDA_VISIBLE_DEVICES`로 지정한다. CPU가 기본값이며, 서로 다른 장치의 부동소수점 결과가 동일하다고 보장하지 않는다.

## 관측 실패 보정 모델 조립

[후속 비교](recovery-scope.md)의 고정 후보는 `scoped-v2`, seed 42, epoch 40이다. 기존 bundle의 encoder·모수 구성기·GRU를 보존하고, 공개한 작은 head를 결합한다. 전체 600 MB 가중치를 다시 내려받을 필요는 없다. 위 명령으로 기준 bundle을 준비한 다음 실행한다. 조립은 검증 당시의 소스를 요구하므로 해당 커밋을 별도 작업 폴더에 고정한다. 이후 학습 정답 연결 코드가 변경되어도 같은 가중치를 재현하기 위한 절차다.

```sh
git worktree add --detach results/released-model/recovery-source \
  2b0b6506d4dd3b92be76639a63ade29e48caacd7

tar -xJf docs/evidence/artifacts/recovery-scope-records-v1.tar.xz \
  -C results/released-model/recovery-source \
  --wildcards 'results/l40s-recovery-scoped-v2/*' \
  'results/l40s-recovery-scoped-dev-v1/*'

(
  cd results/released-model/recovery-source
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    uv run python scripts/compose_recovery.py \
    --bundle ../forecast-v1 \
    --study results/l40s-recovery-scoped-v2 \
    --dev results/l40s-recovery-scoped-dev-v1 \
    --output ../recovery-v1
)

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/predict_research.py \
  --bundle results/released-model/recovery-v1 \
  --inputs results/released-model/example-inputs.jsonl \
  --output results/released-model/recovery-predictions.jsonl
```

조립 코드는 사전 고정한 seed·epoch, 세 seed의 Dev 기준 통과, 후보·부모·동결 구성기의 tensor hash와 실행 소스 hash를 검증한다. 새 학습이나 Test 평가는 하지 않는다. 조립된 bundle에는 encoder 설정·토크나이저·전체 추론 가중치·출처 manifest가 들어가므로 이후에는 기준 bundle이나 학습 자료 없이 로드한다. 이 경로는 CPU에서도 조립하며, GPU 추론은 CUDA PyTorch 환경과 `--device cuda`를 사용한다.

보정본은 작업용 데이터·checkpoint·네트워크를 차단하고 공개 자료로 조립했다. L40S에서 고정 Dev 경로 720개의 행동·발주량·손실을 재현했으며, 위 CPU 추론 명령도 공개 Train 관측 예제 8개에서 실행했다. 전체 tensor hash는 `e01f07f8ab6325411975733303a277c95d5c914170fd833d18c452ab4bd105dd`다. [조립·loader·재현 기록](evidence/research-recovery-scope-results.json)

이번 보정의 이점은 통제 Dev에서 관측한 무응답 이후 질문 비용 감소다. 단순 중단 규칙과 행동·발주 결과가 같았으므로 학습 고유의 우위를 입증한 배포로 해석하지 않는다.

## 기준 bundle의 검증과 제공 범위

원래 checkpoint와 공개 묶음을 CPU와 L40S에서 각각 대조해, 8개 사례의 전체 구조화 출력이 장치별로 정확히 일치했다. 별도 CPU 프로세스에서도 빈 작업 폴더·빈 모델 cache·네트워크 차단 조건으로 같은 결과를 확인했다. 전체 tensor hash도 일치한다. 이는 저장·로딩 경로의 재현 검사이며 새로운 성능 평가가 아니다. [manifest·원시 출력·검증 기록](evidence/research-model-bundle.json)

| 항목 | 내용 |
| --- | --- |
| 선택 checkpoint SHA-256 | `e2fb23ee44b7a6c7158b111d122280d41f3626027f33323ff98f0fddc765904a` |
| 전체 tensor hash | `736836c735ab9e61662ea14371b699139bd60f00d80a7fdfd40e6b7139b6137e` |
| ModernBERT revision | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| 공개 archive SHA-256 | `44869c17a4e2d41f4dd4cfe9331fa995e3f9e7e40b29cad21c02ec34e090fd79` |

원래 학습 자료의 hash와 모델 revision을 보존하며 upstream Apache-2.0 라이선스·출처·변경 설명을 함께 제공한다. 저장소 전체에 새로운 라이선스를 지정한 것은 아니다.

Optimizer 상태와 확장 Train 원본 전체는 이 묶음에 포함하지 않는다. 따라서 기존 학습을 완전히 재개하는 배포나 모든 과거 표의 자동 재평가까지 제공하는 것은 아니다. 공개 snapshot에서 새 모델을 학습하는 경로는 [README](../README.md)에, 공개 작은 GRU와 새 출처 27개로 수요·발주를 재계산하는 경로는 [별도 재현 안내](retail-holdout.md)에 있다.

선택 모델의 통제 Dev 모수·근거 정확도를 실제 조직 문서의 효과로 해석하지 않는다. 공개 문서 추출, 수요 calibration, 기존 생성 Test 목표 미달은 여전히 남아 있다. [현재 연구 범위와 제한](research-scope.md)
