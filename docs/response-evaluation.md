# 같은 매니저 응답 조건에서 정책 비교

과거 Retail 응답 누락 평가는 난수의 키에 질문 순번을 넣었다. 같은 항목을 두 번째로 물으면 첫 번째로 물을 때와 응답 여부가 달라질 수 있었다. 따라서 질문 순서가 다른 정책의 한 번 실행 손실을 그대로 비교하면 정책과 응답 표본의 효과가 섞인다. [기존 수치와 당시 해석](research-scope.md#경제-가치-학습-제거-비교)은 보존한다.

이번에는 현재 선택한 `forecast-v1`과 같은 문서 encoder·모수 구성기·수요 GRU를 쓰는 `no_value`, 고정 질문 정책을 비교한다. 가중치를 새로 학습하거나 선택하지 않는다. 이전 표의 모델은 GRU 교체 전 v4이므로 두 표의 손실 차이를 이번 평가 수정의 성능 향상으로 해석하지 않는다.

## 비교 방식

원가 `c`, 판매가 `p`, 잔존가치 `v`, 부족비용 `b` 각각의 응답 유무를 고정한다. 가능한 **16개 조합을 모두 실행**하고, 항목별 독립 누락률 0·10·20%에서 각 조합의 확률로 가중 평균한다. 예를 들어 누락률이 `r`이고 네 항목 중 두 항목이 누락되는 특정 조합의 확률은 `r²(1−r)²`다. 질문 순서나 추출한 난수에 따라 응답 표본이 바뀌지 않는다.

- 같은 사례·항목·응답 조합을 세 정책에 적용한다. 응답 유무는 실제 질문 이후에만 관측되며 입력·특성에 넣지 않는다.
- 매 조합에서 모델이 스스로 상태를 구성하고 다음 행동과 발주량을 결정한다. 모든 답변이 있어야 성공한다고 가정해 손실을 대입하는 방식은 아니다.
- 동일한 관측 상태의 결정론적 계산은 재사용한다. 전체 실행 시간은 이 평가의 시간이며 운영 환경의 응답 지연 측정이 아니다.
- 누락이 없는 조합은 보존된 원래 Dev의 발주량과 모수·근거·행동·손실 지표에 대조한다.
- 이 경로는 Retail Dev에만 허용한다. 기존 Train 응답 재표집과 기본 Dev·Test 실행 규칙은 보존한다.

[실행 전 등록](../configs/research-responses-v1.json)에는 입력·모델·소스·평가 코드의 hash와 비교 조건을 고정했다. [평가 스크립트](../scripts/evaluate_responses.py)는 모든 원시 rollout과 조합별 측정을 보존한다.

## L40S 결과

| 응답 누락률 | 가치 정책 | no_value | 고정 질문 |
| --- | ---: | ---: | ---: |
| 0% | 24.3925 | 24.3925 | 24.3925 |
| 10% | 71.9563 | 71.9563 | 71.9563 |
| 20% | 118.1415 | 118.1415 | 118.1415 |

총 **336개 통제 사례 × 16개 조합 × 3개 정책 = 16,128개 경로**를 실행했다. 정확한 평균 총손실은 그중 완전 관측인 **고유 판매 기간 5개·출처 4개 × 7개 조건 = 35사례**에 한정한다. 응답 조합을 늘린다고 독립 판매 자료가 늘어나지는 않는다. 전체 실행은 423.61초, 최대 GPU 할당은 약 0.638 GiB였다.

세 정책 모두 누락 없는 원래 발주량과 측정 지표를 재현했다. 모든 응답 조합에서 불필요한 재질문과 잘못된 발주 진행은 0건이었다. 당시 관측으로 확인할 수 있는 모수 값·누락 여부 및 채택 근거의 정확도는 100%였으며, 미응답으로 숨겨진 값을 알아냈다는 뜻은 아니다. 상태 정확도는 누락률 0·10·20%에서 각각 **100%·97.86%·95.71%**다. 예를 들어 원가 질문에 응답이 없으면 기준 상태는 `unavailable`인데 모델은 `unconfirmed`로 남기는 오류가 있다.

현재 조건에서 가치 학습의 추가 경제 이점은 확인되지 않았다. 응답 누락에 따라 보류가 늘고 세 정책의 기대 총손실도 동일하게 증가했다. 이는 특정 모델·작은 Dev·항목별 독립 누락이라는 가정 아래의 결과이며 실제 매니저의 응답 행태나 정책의 일반적 우열을 입증하지 않는다. Test와 별도 27개 출처는 다시 평가하지 않았다.

[전체 지표·출처 비교·파일 hash·검증](evidence/research-paired-response-results.json) · [가치 정책 원시 경로](evidence/artifacts/response-evaluation-value-v1.tar.xz) · [no_value 원시 경로](evidence/artifacts/response-evaluation-no_value-v1.tar.xz) · [고정 질문 원시 경로](evidence/artifacts/response-evaluation-checklist-v1.tar.xz) · [입력·등록·가중치·측정·소스](evidence/artifacts/response-evaluation-records-v1.tar.xz)

## 공개 가중치로 재현

먼저 [선택 모델 다운로드 안내](model-bundle.md)에 따라 `research-forecast-v1` 묶음을 받는다. 아래 기록에는 원래 관측 입력과 별도 채점 환경, 모든 조합의 측정값, 약 269 KB의 `no_value` head 가중치가 있다. `no_value`는 공개된 전체 모델의 행동 head만 바꿔 복원하며, 전체 tensor hash가 원래 checkpoint와 같은지 검사한다. 원래 무시된 학습 자료나 로컬 checkpoint는 필요하지 않다.

```bash
tar -xJf docs/evidence/artifacts/response-evaluation-records-v1.tar.xz \
  --wildcards 'results/l40s-response-eval-v1/*'

CUDA_VISIBLE_DEVICES=GPU-674d64b8-4bdf-7006-1791-5dc7f7245409 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONPATH=src \
.venv-l40s/bin/python scripts/reproduce_responses.py \
  --bundle results/released-model/forecast-v1 \
  --data results/l40s-response-eval-v1 \
  --output results/reproduced-responses-v1
```

CUDA PyTorch 환경과 한 장의 L40S를 사용한다. UUID는 재현 장치의 것으로 지정한다. `--smoke`를 추가하면 각 통제 조건의 첫 사례만 확인한다. 재현은 평가 수치의 일치 검사이며 새로운 성능 증거가 아니다. 관측 판매와 생성된 계약·매니저 조건을 구분하며, 실제 조직에서의 유효성으로 해석하지 않는다.

공개 기록을 빈 디렉터리에 복원하고, 기존 작업용 `data/`·`results/` 읽기와 네트워크를 차단한 별도 L40S 프로세스에서 확인했다. 고정된 7사례 × 16응답 조합 × 3정책의 **336개 측정값이 모두 정확히 일치**했다. 이 로딩 검사는 나머지 329사례를 다시 실행하지 않았다. 전체 tensor hash는 원래 두 checkpoint와 일치한다. [격리 재현의 입력 접근 검사·원시 출력·로그](evidence/artifacts/response-reproduction-v1.tar.xz)

전체 테스트 303개, Ruff, 다섯 데이터 버전의 출처 분리 검사, 수치 smoke 30개도 통과했다. 출처별 가중치를 맞춘 별도의 Train 수요 후보는 내부 검증에서 개선되지 않아 확대하지 않았다. [실패한 후보의 판정과 보존 기록](retail-train-expansion.md#후속-train-진단-출처별-학습-비중)
