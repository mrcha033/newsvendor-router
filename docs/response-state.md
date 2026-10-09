# 관측한 매니저 응답을 모수 상태에 반영

매니저에게 원가를 물었는데 답이 오지 않았다면, 상태에는 ‘원가 질문에 응답 없음’이 남아야 한다. 기존 경로는 숫자 응답만 복사해서, 무응답 뒤에도 `unconfirmed`로 남는 경우가 있었다. 반대로 질문하기 전에 모델이 `unavailable`을 예측하는 경우도 있었다. 이 두 상태를 관측 이력에 맞게 정리했다. **가중치 재학습이나 새로운 숫자 추출 성과는 아니다.**

[수정 코드](../src/newsvendor/structured_forecast.py)는 다음 규칙을 적용한다.

- 실제로 질문한 항목의 응답이 없고 완전한 근거도 없으면 `unavailable`로 기록한다. 아직 질문하지 않은 항목의 `unavailable` 예측은 `unconfirmed`로 되돌린다.
- 현재 SKU·기간에 맞는 부분 자료가 있으면 `candidate`를 유지한다. 다른 상품이나 기간의 부분 자료는 사용하지 않는다.
- 이미 확인한 값, 완전한 근거, 충돌 상태를 무응답으로 덮어쓰지 않는다. 이후 적용 가능한 숫자 응답이 오면 값과 근거를 다시 연결한다.
- 실제 응답의 이력 위치·hash와 부분 자료의 hash를 남긴다. 원래 head 출력은 `rawFields`에 보존한다. 숨겨진 응답 확률이나 평가 정답을 상태 변경에 사용하지 않는다.

## 같은 가중치에서 수정 전후 비교

먼저 보존된 Train 경로의 고유 관측 상태 1,153개·85개 family를 검사했다. 네 모수의 상태 4,612개 중 오류는 **198개 → 0개**였다. 무응답 반영 159개, 아직 질문하지 않은 항목의 잘못된 `unavailable` 14개, 부분 자료 상태 25개를 바로잡았다. 값·식·근거·유형은 그대로였다. 이는 고정된 Train 상태의 진단이며 독립 평가나 실제 업무 성능의 증거가 아니다.

이어서 [생성 Dev 등록](../configs/research-response-state-generated-v1.json)의 기준을 통과한 뒤 [판매 Dev 등록](../configs/research-response-state-retail-v1.json)에 따라 L40S에서 실행했다. 문서 encoder·모수 head·수요 GRU·행동 head 가중치는 모두 고정했다. Test와 별도 27개 출처 holdout은 다시 평가하지 않았다.

| 판매 Dev 응답 누락률 | 수정 전 상태 정확도 | 수정 후 상태 정확도 | 수정 전후 평균 총손실 |
| --- | ---: | ---: | ---: |
| 0% | 100% | 100% | 24.3925 |
| 10% | 97.86% | 100% | 71.9563 |
| 20% | 95.71% | 100% | 118.1415 |

가치 정책·`no_value`·고정 질문 정책 모두 위 결과였다. 336사례 × 16응답 조합 × 3정책의 **16,128개 실제 경로**를 비교했으며, 조합별 발주량·행동·총손실과 상태 정확도 이외의 측정값은 모두 같았다. 누락 조건은 질문한 뒤에만 관측한다. 원래 head의 최종 상태·값·유형 정확도도 변하지 않았다. **100%는 응답 이력을 반영한 구조화 상태의 정확도이며 신경망 원출력의 정확도가 아니다.**

생성 Dev 60사례·12개 family에서도 평균 총손실은 가치 정책 **147.6813**, 나머지 두 정책 **181.6758**로 변하지 않았다. 가치 정책이 방문한 고유 상태 80개의 상태 오류는 원래 0개였고, 나머지 정책은 각각 117개 상태 중 모수 상태 오류 11개가 0개로 줄었다. 이때 원출력의 상태 오류 11개는 그대로 남아 있다.

생성 비교는 35.15초, 판매 비교는 416.08초였다. 같은 관측 상태의 결정론적 연산을 재사용한 평가 시간이며 운영 지연 측정이 아니다. 판매의 정확한 총손실 표본은 여전히 **고유 완전 관측 기간 5개·출처 4개 × 7조건 = 35사례**다. 관측 판매에 통제 문서·매니저 조건을 결합했으며 실제 조직에서의 효과를 입증하지 않는다. 가치 학습의 추가 경제 이점, 공개 문서 추출과 분포 보정 문제도 해결됐다고 볼 수 없다.

Train의 기존 경제 손실 경로를 추가로 점검하니, 응답 실패 후에도 다른 질문을 선택할 수 있는 상태의 측정 표본은 단 1개였다. 그 사례는 추가 질문 뒤에도 보류하면서 비용만 더 냈다. 일반적인 회복 능력을 판단하기에는 부족하므로, 이 진단만으로 정책 우열이나 예상 절감액을 주장하지 않는다.

[전체 결과·검증·파일 hash](evidence/research-response-state-results.json) · [수정 전 동일 조건 비교](response-evaluation.md) · [등록·원시 Train 상태·생성 경로·소스·재현 입력](evidence/artifacts/response-state-records-v1.tar.xz) · [가치 정책 원시 판매 경로](evidence/artifacts/response-state-value-v1.tar.xz) · [고정 질문 원시 판매 경로](evidence/artifacts/response-state-checklist-v1.tar.xz) · [no_value 원시 판매 경로](evidence/artifacts/response-state-no_value-v1.tar.xz)

## 공개 가중치로 재현

[선택 모델 다운로드](model-bundle.md)로 `research-forecast-v1` 묶음을 받은 뒤 아래 명령을 실행한다. 기록된 소스 hash와 모든 입력 hash를 검사한다. 별도 학습 checkpoint는 필요하지 않으며, `no_value`는 공개 모델에 기록 묶음의 작은 정책 가중치 차이를 적용하고 전체 tensor hash를 검증한다.

```bash
tar -xJf docs/evidence/artifacts/response-state-records-v1.tar.xz \
  --wildcards 'results/l40s-response-state-retail-v1/*'

CUDA_VISIBLE_DEVICES=GPU-674d64b8-4bdf-7006-1791-5dc7f7245409 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONPATH=src \
.venv-l40s/bin/python scripts/reproduce_responses.py \
  --bundle results/released-model/forecast-v1 \
  --evidence docs/evidence/research-response-state-results.json \
  --data results/l40s-response-state-retail-v1 \
  --output results/reproduced-response-state-v1
```

CUDA PyTorch 환경과 L40S 한 장을 사용하며 UUID는 재현 장치에 맞춘다. `--smoke`는 각 통제 조건의 첫 사례 7개로 로딩과 결과 일치를 확인한다. 전체 재현과 이 작은 검사는 구분한다. 전체 테스트 317개, Ruff, 다섯 데이터 버전의 출처 분리 검사, 수치 smoke 30개를 통과했다.

공개 기록을 빈 디렉터리에 복원하고 기존 `data/`·`results/`와 네트워크 접근을 차단한 L40S 프로세스에서 **7사례 × 16응답 조합 × 3정책의 측정값 336개가 모두 정확히 일치**했다. 이 로딩 검사는 나머지 329사례를 다시 실행하지 않았다. [격리 재현 원시 출력·접근 차단 확인·로그](evidence/artifacts/response-state-reproduction-v1.tar.xz)
