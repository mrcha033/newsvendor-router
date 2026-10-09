# 도구·상태·절차 학습 변경

`configs/l40s-tools-v3.json`과 `scripts/run_tools.py`는 기존 base의 공개 자료·수요 학습 체크포인트에서 시작해 자연어 encoder 전체와 새 head를 미세조정한다. 원래 Dev/Test와 source-family 분할은 유지한다. 실행은 완료했으며 [결과와 한계](tool-upgrade-results.md)를 정리했다. 도구 선택 재현율은 올랐지만 대화 상태 연결의 인자 오류와 경제적 손실은 악화돼 새 전체 구성의 채택은 권하지 않는다. 원시 비교는 `results/l40s-tools-v3/base/42/tool-comparison.json`에 있다.

## 구현

1. **역할·자료형·조립**: ABCD ontology의 값 목록을 열린 후보로 처리한다. 이름 있는 JSON Schema 필드의 enum과 자료형은 엄격히 검증한다. 순서 인자 중간이 비면 질문으로 반환하며 뒤 인자를 앞으로 당겨 실행하지 않는다. 각 순서 인자의 의미 역할은 공유 query pointer로 예측하며 실제로 구별 가능한 Train 주석만 학습한다.
2. **공유 추출·상태 복사**: span, schema choice, 관측 entity, 이전 자기 예측 값에 대한 entity pointer, 계산기 경로를 연결한다. 문자열 복사 후보는 원문 위치와 값이 일치해야 한다. 계산값은 원래 연산·순서 있는 피연산자 위치를 보존하며 관측 원문에서 재계산해 일치해야 한다. 사용 중인 필드의 충돌은 이전 값을 제거한다. 질문용 보조 필드의 비지도 추출값은 memory에 넣지 않으며, 임시 후보값이 있다는 이유로 질문 대상을 제외하지 않는다. schema 선택값에 임의의 원문 span을 근거로 붙이지 않는다. 도구 실행 controller는 대화 사이의 자기 예측 memory를 보존하고 새 episode에서 초기화한다.
3. **절차·진행**: 공개 정책의 55개 절차를 원문 위치와 함께 색인한다. 절차 query와 진행 단계 head를 Train 원천 주석으로 학습한다. 예측한 절차 상위 3개는 추가 조회의 chunk 우선순위에 사용한다. 진행 단계 표적은 앞서 관측된 도구 행동 수(최대 15)다. 분기·반복이 있는 정확한 workflow node를 정답이라고 가정하지 않으며, 절차·단계 점수로 도구를 강제 차단하지 않는다.
4. **후보 재평가**: 행동 표현, 예측 절차·단계, 인자의 누락·충돌·사용 점수로 추가 점수를 학습한다. 기본 행동 loss도 유지한다. 동일 checkpoint의 `encoder.rerank`만 켜고 꺼서 비교한다.
5. **자기 상태 행동 가치**: 자기 정책과 관측 가능한 checklist 탐색을 섞어 상태를 수집한다. 가능한 첫 행동을 실제 simulator에서 실행하고 뒤의 자기 예측 정책이 만든 최종 손실·요청 비용을 학습한다. 언어 replay와 고정 teacher를 사용하며, 선택된 최선 checkpoint보다 공개 Dev 지표가 낮아진 후보를 채택하지 않는다.

ABCD에서 역할이 모호하면 가능한 역할을 marginal loss로 학습하거나 loss를 비운다. 숨은 인자에 임의의 누락·충돌 정답을 만들지 않는다. 절차 주석은 Train ID에만 붙이며 입력 생성 함수는 그 주석을 받지 않는다. state 복사에는 모델이 예측한 값만 사용한다.

## 학습·선택

L40S 한 장, BF16, SDPA, layer compile, 길이별 batch, batch 안의 공통 schema query 재사용을 사용한다. encoder 표현은 optimizer 갱신 후 다시 계산한다. 최종 설정은 case batch/accumulation 32이며 긴 batch는 재귀적으로 나눈다. 수요 GRU와 분포 head는 이번 변경 대상이 아니므로 기존 수요 가중치와 cross-fit 원시 측정을 재사용한다.

Train 전체 pool을 섞어 사용하며 사례 수를 임의의 diagnostic limit으로 자르지 않는다. 8,192건마다 Dev의 도구+인자 완전 일치, 도구 일치, 행동 정확도 순으로 선택하고, 동률이면 전체 구성 Dev loss로 선택한다. 최소 16,384건 이후 정확도 지표가 3회 연속 개선되지 않으면 조기 종료한다. 최대 6 epoch다. 실제 처리 건수·선택 시점·조기 종료 여부를 `language.checks`에 저장하므로 전체 pool을 한 번 이상 보았는지도 구분할 수 있다. 클래스 가중치는 Train 행동 빈도로만 계산한다.

`progress.json`은 현재 학습 단계, `job.json`은 비교 평가를 포함한 전체 실행 상태다. `run.json`/`source.tar.gz`/`tools-parent.json`은 이번 코드·설정·입력 hash와 이전 가중치의 출처를 보존한다. 같은 코드·설정·자료로 `--resume`하면 optimizer/RNG와 선택 상태를 복원한다.

```sh
CUDA_VISIBLE_DEVICES=<L40S UUID> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=src <CUDA PyTorch Python> scripts/run_tools.py
# 중단 후 동일 실행을 이어가기
CUDA_VISIBLE_DEVICES=<L40S UUID> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=src <CUDA PyTorch Python> scripts/run_tools.py --resume
```

## 비교와 해석

비교에는 기존 base의 인자 조립 수정 조건, 새 구성의 행동 가치 추가 학습 전·후, 각 checkpoint의 재평가 켜기·끄기, 최종 checkpoint에서 경제적 가치 사용을 끄는 진단이 포함된다. 마지막 조건은 별도의 재학습 ablation이 아니다.

ABCD 비교는 고정 Test의 관측된 대화 prefix에 이전 **자기 예측 memory**를 이어 붙인다. memory는 conversation ID로 격리하고 source-family는 통계 집계에만 사용한다. 기존 모델에는 이전과 같은 상태 없는 입력을 제공한다. 이후 사용자·도구 응답은 공개 자료에 기록된 값이므로 완전한 대화 환경 rollout이라고 부르지 않는다. strict 도구·인자 지표와 source-family paired bootstrap을 저장한다. Test로 학습 시점이나 hyperparameter를 선택하지 않는다. 평가 코드의 provenance와 source archive는 학습 당시 코드와 구분해 보존한다.

`scripts/summarize_tools.py`는 저장된 예측을 추가 채점한다. 도구 정확도의 분모는 정답 도구 호출 수, 도구+인자 완전 일치의 분모는 인자가 관측 가능한 정답 호출 수다. 잘못 실행한 호출을 놓치지 않도록 예측 호출을 분모로 하는 precision도 별도 보고한다. 절차·진행 정답은 이 사후 채점 단계에만 사용한다. 질문 필드 점수는 참조 질문이 schema 필드를 명시한 경우에만 계산하므로 전체 질문 필요성 정확도로 해석하지 않는다.

생성 발주 환경은 자기 구성 상태로 실행하며 응답 오류율 0/0.1/0.2에서 손실·상호작용·잘못된 handoff·구성 오류 누적·회복을 평가한다. **손실 정의 변경**: 유효하지 않은 handoff의 terminal loss는 최소 hold loss로 처리한다. 숨은 참모수에서 우연히 좋은 수량을 냈다는 이유로 잘못된 handoff를 보상하지 않기 위한 것이다. 원래 경제 손실(`economicLoss`)과 추가된 `invalidHandoffPenalty`를 별도로 저장하며 기존 checkpoint도 같은 정의로 재평가한다. 생성 시나리오의 결과는 실제 조직 업무의 효과성 근거가 아니다.

## 구현 검증

추가 회귀 테스트를 포함한 59개 테스트, source-group/보호 split 검증, 30개 수치 smoke가 통과했다. L40S에서 새 추출·절차·자기 상태 추가 학습 경로를 포함한 optimizer step이 유한한 loss/gradient로 실행됐다. 처리량 측정은 `results/l40s-tools-bench.json`과 `results/l40s-tools-bench32.json`에 있으며 이 작은 실행은 성능 평가 자료가 아니다.

Train 원천 인자를 그대로 조립하는 검사에서는 27,866건 중 기존 규칙이 거부하던 2,541건이 수정 후 0건이 됐다. 원시 결과는 `results/l40s-tools-v3/assembly-train.jsonl`에 보존했다. 이는 정답 인자를 주었을 때의 조립 가능성 검사이며 모델 정확도가 아니다.
