# 응답별 학습 빈도 보정과 전체 재학습 비교

**보정한 가치 head는 채택하지 않았다.** Train 내부 비교와 새로운 응답을 사용한 Train 순차 실행에서는 손실이 낮아졌으나, 전체 Train 재학습 뒤 원래 Dev에서는 개선이 사라졌다. 선택 모델은 `forecast-v1`을 유지한다. Test와 별도 27개 출처 holdout은 사용하지 않았다.

## 왜 이 비교를 했나

[후속 상태 확대](recovery-replay.md)만으로는 매니저 무응답 뒤의 불필요한 후속 질문을 줄이지 못했다. 기존 내부 적합 상태 928개 중, 판매 모수 질문이 실패하고 다른 질문이 가능한 상태는 4개였다. 균일 표집에서 이 상태들의 비중은 0.431%다. 초기 head에서 이 네 상태의 평균 gradient와 전체 평균 gradient의 cosine은 −0.794였다. 서로 반대 방향의 갱신 압력이 있다는 진단이며, 효과의 증명은 아니다.

학습 표본의 절반은 기존처럼 균일하게 뽑고, 절반은 **자료 종류 → 이미 관측한 응답 종류 → 출처 family → 상태** 순서로 같은 비중을 준다. 응답 종류는 아직 질문 전·답변 있음·부분 답변·무응답이며, 조회는 매니저 답변으로 세지 않는다. 미래 응답·정답·손실·검증 출처는 표집 확률을 결정하지 않는다.

같은 1,204개 관측 상태, 11,044개 실제 후속 경로 표적, 94차원 숫자 입력과 기존 문서 표현을 사용했다. 모델과 표적을 바꾸지 않고 표집만 비교한다. 새 조건에서 위 네 상태의 표집 비중은 1.257%였고, seed 42·43·44의 실제 방문 횟수는 446·520·466회였다. 각 실행은 총 38,400개 표본과 600번의 갱신을 사용했다.

## 내부 검증에서 실제 순차 실행까지

68개 Train family로 적합하고 17개 family의 276개 상태로 epoch를 선택했다. 부모 encoder는 이전에 이 Train 출처를 보았으므로 독립 일반화 평가가 아니다. 기존 균일 조건의 고정된 가중치·입력·예산을 재사용해 불필요한 재학습을 피했다.

세 seed 모두 내부 선택 행동의 표본 손실이 **0.194831 → 0.194518**로 낮아졌다. 이는 부모 정책의 후속 경로를 표적으로 삼은 점수이며 새 정책의 실제 총손실과 구별한다. 한 판매 실패 상태에서 추가 질문 대신 보류를 골랐고, 표적 손실이 같았던 일부 초기 상태에서는 질문 순서도 달라졌다.

다음 단계에서는 원래 Train 514사례·85개 family에서 새로운 응답 표본 4개로 실제 순차 실행을 했다. 같은 모수 채널에는 질문 순서와 관계없이 두 정책에 같은 응답을 제공했다. 총 12,336개 경로다.

| Seed | 생성 총손실: 균일 → 보정 | 판매 총손실: 균일 → 보정 |
| --- | ---: | ---: |
| 42 | 130.8336 → 129.1152 | 107.9386 → 107.7750 |
| 43 | 130.8645 → 129.0383 | 107.9386 → 107.8735 |
| 44 | 130.4058 → 129.1585 | 107.9386 → 107.8735 |

판매는 종료 손실이 같고 질문 비용만 줄었다. 생성은 종료 손실이 각각 2.0964·0.8823·0.3957 늘고 질문 비용이 더 많이 줄어 총손실이 낮아졌다. 잘못된 발주 진행은 모두 0건이었다. 이 수치는 Train의 통제된 응답 조건 결과이며, 조직 업무에서의 효과나 정확한 최적 정책을 뜻하지 않는다.

## 전체 Train 재학습 뒤에는 개선이 유지되지 않았다

전체 1,204개 상태·85개 family로 두 조건을 다시 학습했다. 모두 40 epochs, epoch당 19번의 batch 64 갱신을 수행했다. 최종 평가에 쓸 epoch는 앞선 내부 검증에서 고정했다. 균일 조건은 3·6·11, 보정 조건은 21·34·25 epoch이며 전체 Train이나 Dev 점수로 다시 고르지 않았다.

원래 생성 Dev 60사례와 판매 Dev 336조건을 평가했다. 판매는 16가지 공통 응답 가능 여부를 모두 실행해 누락률별 기댓값을 구했다. 여섯 조건의 총 32,616개 경로를 저장했다.

| Dev 지표 | 유지 중인 모델 | 균일·보정 재학습, 각 3개 seed |
| --- | ---: | ---: |
| 생성 총손실 | 147.6813 | 148.3549 |
| 판매 총손실, 누락 0% | 24.3925 | 24.3925 |
| 판매 총손실, 누락 10% | 71.9563 | 71.9563 |
| 판매 총손실, 누락 20% | 118.1415 | 118.1415 |

모수·상태·근거, 필요한 질문과 회복, 잘못된 발주 진행, 공개 문서 지표의 보존 기준은 통과했다. 그러나 생성 손실이 기존보다 0.6736 높고 기존 모델 대비 개선도 없어 세 seed 모두 채택 기준에 미달했다. 공개 문서 추출이나 수요 calibration이 해결됐다는 결과도 아니다. 판매의 정확한 손실 근거는 여전히 완전 관측 5개 기간·4개 출처의 35조건이다.

L40S에서 같은 Train 판매 실패 상태 6개를 대조하자, 내부 선택 head는 세 seed 모두 4개에서 보류했으나 전체 재학습 head는 0개에서 보류했다. 이미 학습에 들어 있던 상태에서도 행동이 유지되지 않았다. 명시적 상태만 바꾼 CPU 민감도 검사와 활성값도 보존했다. 초기 및 재학습 head에서 Tanh 포화가 많지만, 이 실험은 포화를 따로 조작하지 않았으므로 원인으로 단정하지 않는다. 표집 빈도만의 문제로 결론 내리지 않고, 재학습에 따른 행동 변화와 숫자 상태의 전달 경로를 다음 검사 대상으로 남긴다.

학습·검증 시간은 내부 비교 70.74초, Train 순차 비교 231.35초, 전체 Train 재학습 165.99초, Dev 평가 705.10초였다. encoder 표현과 구성 상태를 공유한 여러 정책의 실행 시간이며 운영 지연으로 해석하지 않는다. 순차 비교 준비 중 tensor·tuple 키를 직렬화하지 못한 두 실패 실행도 보존했다. 비교가 시작되기 전의 오류였으며 같은 조건으로 수정 후 실행했다.

## 기록과 재현

[내부 비교 등록](../configs/research-recovery-balanced-v1.json) · [순차 Train 등록](../configs/research-recovery-balanced-sequential-v1.json) · [전체 재학습과 Dev 기준](../configs/research-recovery-balanced-refit-v1.json) · [고정 Dev 실행](../configs/research-recovery-balanced-dev-v1.json) · [학습 코드](../scripts/study_recovery_sampling.py) · [결과·검증·hash](evidence/research-recovery-sampling-results.json)

테스트 328개, Ruff, 다섯 자료 버전의 출처 분리 검사, 수치 smoke 30개를 통과했다. 후보 head·optimizer·모든 epoch 손실·실제 batch와 실행 소스를 [기록 archive](evidence/artifacts/recovery-sampling-records-v1.tar.xz)에, 실제 경로를 [원시 JSON archive](evidence/artifacts/recovery-sampling-rollouts-v1.tar.xz)에 보존한다. JSON 바이트와 순서는 그대로 두고 gzip을 풀어 tar.xz로 압축했다. 원래 gzip hash와 원시 바이트 hash를 `raw-format.json`에 함께 기록한다.

다음은 [공개 base bundle](model-bundle.md)과 공개된 후보 head·관측 자료로 고정 간격의 Dev 사례를 다시 실행하는 명령이다. 결과 재현이며 새로운 효과 평가가 아니다.

```bash
tar -xJf docs/evidence/artifacts/recovery-sampling-records-v1.tar.xz \
  --wildcards 'results/l40s-recovery-balanced-refit-v1/*' \
  'results/l40s-recovery-balanced-dev-v1/*'
tar -xJf docs/evidence/artifacts/recovery-sampling-rollouts-v1.tar.xz \
  --wildcards 'results/l40s-recovery-balanced-dev-v1/*'

CUDA_VISIBLE_DEVICES=GPU-674d64b8-4bdf-7006-1791-5dc7f7245409 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONPATH=src \
.venv-l40s/bin/python docs/evidence/scripts/reproduce-recovery-sampling.py \
  --bundle /path/to/forecast-v1 \
  --refit results/l40s-recovery-balanced-refit-v1 \
  --dev results/l40s-recovery-balanced-dev-v1 \
  --output results/reproduced-recovery-sampling.json
```

CUDA PyTorch와 L40S가 필요하며 GPU UUID는 실제 장치에 맞춘다. 이 비교에서 새 가중치를 기본 모델로 배포하지 않았다.

공개 자료 57개 파일을 빈 디렉터리에 복원하고 원래 작업용 자료·checkpoint와 네트워크 접근을 차단해 확인했다. 여섯 head에서 생성 8사례와 판매 7사례의 16가지 응답 조건, 총 **720개 경로의 행동·발주량·손실이 정확히 일치**했다. 원시 경로 44,952개의 시작 입력·비용 합계·보고 지표도 전수 대조했다. [복원·접근 차단·원시 재현 출력](evidence/artifacts/recovery-sampling-reproduction-v1.tar.xz)
