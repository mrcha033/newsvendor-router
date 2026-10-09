# 정보 요청 head에 현재 모수를 전달하는 경로

작은 모델은 자신이 구성한 발주 모수를 보고 질문·계산 진행·보류를 선택해야 한다. 그런데 그 상태를 JSON 문장으로만 전달하던 경로에서 뒤쪽 값이 잘리는 결함을 확인했다. 모수 값·유형·상태와 공개 요청 비용을 작은 수치 벡터로도 전달하는 선택 경로를 구현했다. **텍스트 잘림을 우회하는 수치 입력을 추가했지만, BF16 정밀도 한계가 남고 추가 학습의 Dev 개선도 확인되지 않았다.**

## 확인한 결함

이전 실행에서 저장한 **Train 자기 예측 상태 928개**를 검사했다. 상태 JSON은 키 순서로 정렬되고, 질문 prefix에는 앞 128 tokens만 들어간다. 판매 이력이 연결된 사례에서는 긴 수요분포 메타데이터가 앞부분을 차지했다. 원문 문서가 남아 있어도, 모델이 그 문서에서 *현재 채택한 값과 상태*는 같은 방식으로 보존되지 않았다.

| 직접 바꿔 본 자기 예측 상태 | 검사 수 | 전체 토큰 입력·허용 행동이 그대로인 수 |
| --- | ---: | ---: |
| Retail 발주량 `q`를 1 증가 | 136 | 136 |
| Retail 부족 비용 `b`의 유형을 사실/선호 간 변경 | 308 | 308 |
| Retail 판매가 `p`를 1% 증가 | 284 | 284 |
| 원래 생성 사례의 판매가 `p`를 1% 증가 | 620 | 441 |

이는 입력 경로의 검사다. 발주량만 바꾸는 조작은 다른 상태·수요분포와 일관된 대안 시나리오를 생성한 것이 아니며, 숨겨진 정답이나 실제 성능의 증거로 사용하지 않는다. Dev·Test 정답으로 이 결함을 찾지 않았다. [감사와 검증 원시 기록](evidence/research-numeric-state-checks.json)

## 수정 내용

`encoder.numericState=true`이면 행동별 256차원 표현에 다음 수치 입력 94개를 이어 붙인다.

- 모델이 채택한 `c/p/v/b`, 각 값의 존재 여부·유형·상태, 예상 발주량과 비용.
- 모델이 구성한 수요 후보의 평균·표준편차 범위와 후보 수.
- 공개된 보류 비용·요청 비용·발주 범위·남은 질문 예산 및 관측 질문 이력.
- 현재 후보 행동과 그 행동의 비용·기존 요청 횟수.

통화량은 공개 보류 비용으로, 수량은 공개 발주 범위로 정규화한다. 없는 값과 숫자 0은 구분한다. 응답 가능성 `rho/partial`, 미관측 수요, 숨긴 참모수는 읽지 않는다. 근거와 원문 위치를 보존하는 기존 구성 경로에 더해, 행동 head가 현재 숫자를 직접 받게 한다.

가치 head와 회복 head의 첫 층이 각각 `256 → 128`에서 `350 → 128`로 늘어난다. 추가 파라미터는 **24,064개**, 전체는 **151,872,438개**다. 이전 checkpoint를 가져올 때 기존 256개 입력 열을 복사하고 새 열은 0으로 시작한다. 초기 행동과 손실을 그대로 보존한 뒤 새 입력의 가중치를 학습할 수 있다.

같은 Train 검사에서 생성된 수치 벡터는 모든 유효한 변경을 구분했다. 두 batch fusion 경로의 초기 출력 보존, 새 열의 gradient, CPU에서의 cached critic 학습과 추론의 일치, 통화 단위 변경 불변성, 숨긴 라벨 변조에 대한 입력 불변성을 검사했다. **전체 테스트 213개, Ruff, 세 자료 버전의 출처·입력 중복 검사, 30개 수치 smoke가 통과했다.**

다만 이것이 GPU 연산까지 모든 작은 차이를 보존한다는 뜻은 아니다. 실제 설정처럼 linear 입력을 BF16으로 변환하는 추가 검사에서는 판매가 1% 변경이 Retail 13/284건, 원래 생성 사례 136/620건에서 같은 벡터로 반올림됐다. 생성 사례에서는 `q` 변경 22/620건, `gamma` 변경 53/620건도 같아졌다. 유형 변경은 모두 구분했다. 이 검사는 CPU에서 해당 dtype 변환을 계산한 것으로, 새로운 GPU rollout이나 성능 비교는 아니다. **작은 수치 차이의 정밀도와 GPU 추론·cached 학습의 수치 일치는 추가 검증이 필요하다.**

이후 실제 L40S에서 해당 차이를 확인하고 `actionPrecision="float32"` 경로를 추가했다. Train 928개 상태에서 수치 입력과 실제 추론·cached 학습 출력이 정확히 일치했으나, 동일 예산의 후속 정책 비교에서도 Dev 개선은 없었다. [후속 정밀도 수정과 결과](action-precision.md)

## 동일 예산의 L40S 비교

같은 replay-v1 base checkpoint에서 seed 42로 시작했다. 문서 encoder·구성기·GRU를 고정하고 행동 head만 학습한다. 각 조건은 같은 Train/Dev, 두 번의 rollout 수집, 반복당 최대 40 epochs, 10 epochs마다 Dev 평가와 연속 세 번 미개선 종료 규칙을 사용한다. 원래 생성·Retail의 `총손실 / 공개 보류 비용` 평균에 같은 가중치를 주어 선택한다.

대조군은 기존 입력으로 추가 학습한다. 수정 조건은 수치 입력을 추가하고 경제 가치 표적을 학습한다. `no_value` 조건은 같은 수치 입력에서 관측상 누락된 항목의 질문·회복만 학습한다. 세 조건 모두 학습 전 평가를 포함한다. 이 비교에서 Test는 평가하거나 선택에 사용하지 않는다.

기존 입력과 수치 입력의 초기 Dev 행동·손실은 같았다. 첫 반복의 Train 표적 928개도 파일 단위로 완전히 같다. 세 조건은 모두 종료됐으며 두 가치 조건은 새 후보를 채택하지 않고 초기 부모를 복원했다.

| 조건 | 생성 Dev 평균 총손실 | Retail 완전 관측 기간 평균 총손실 | 새 가치 가중치 채택 |
| --- | ---: | ---: | --- |
| 기존 입력 추가 학습 | 147.6813 | 44.3274 | 없음 |
| 수치 상태 입력 추가 | 147.6813 | 44.3274 | 없음 |
| 수치 상태 입력 + `no_value` | 181.6758 | 44.3274 | 회복 head만 학습 |

학습·평가 시간은 차례로 386.51초, 383.94초, 295.07초였다. `no_value`는 첫 반복 10 epochs의 회복 head를 선택했으며 같은 구성기의 기존 `no_value` 결과와 손실이 같다. 세 조건 모두 잘못된 발주 진행 0건, Retail 불필요한 요청 0건이었다. [전체 비교와 선택 이력](evidence/research-numeric-state-results.json)

수치 입력의 새 열은 후보 checkpoint에서 실제로 갱신됐다. 첫 반복 10 epochs에서 그 열의 norm은 2.6768, optimizer step은 150이었다. 그러나 학습된 후보의 생성 Dev 손실은 기존보다 높았고, 선택 결과는 초기 모델과 같았다. **가시성 결함 수정과 정책 효과 입증은 별개다.** 기본 설정과 선택 모델은 유지한다.

생성 Dev는 60사례·12개 독립 family다. Retail은 336사례지만 정확한 경제 손실은 고유 판매 기간 **5개 × 통제 조건 7개**에서만 계산된다. 한 학습 seed와 반복 사용한 Dev의 결과이며, 실제 조직 효과나 Test 목표 달성을 뜻하지 않는다. 이 수정은 수요 GRU의 calibration을 개선하는 학습도 아니다.

## 기록 재검사

[당시 소스](evidence/artifacts/numeric-state-source.tar.gz)와 [Train 입력 감사 묶음](evidence/artifacts/numeric-state-audit.tar.gz)을 각각 보존했다. 감사 묶음에는 사용한 928개 Train 상태, 원래 토큰 검사 코드, 수치 입력 검사 코드와 결과가 들어 있다. 파일 hash는 검증 JSON에서 확인한다. 토큰 검사는 고정 revision ModernBERT tokenizer의 로컬 cache를 요구한다. 모델 학습이나 Test 정답은 요구하지 않는다.

```sh
# 저장소 루트에서; 기존 결과 파일을 보존하려면 별도 checkout에서 실행
tar -xzf docs/evidence/artifacts/numeric-state-audit.tar.gz
PYTHONPATH=src uv run python docs/evidence/scripts/check-numeric-state.py
PYTHONPATH=src uv run python docs/evidence/scripts/audit-policy-context.py
PYTHONPATH=src uv run python docs/evidence/scripts/check-state-precision.py
```

학습 checkpoint와 확장 자료 전체의 공개 배포 문제는 [공개 재현 범위](research-scope.md#공개-재현-범위)에 남아 있다.

이번 비교의 초기·최종 Dev rollout, 모든 후보의 Dev rollout 및 작은 행동 head 가중치·optimizer, 실행 설정·버전·로그는 [원시 결과 묶음](evidence/artifacts/numeric-state-results.tar.xz)에 있다. 전체 모델 checkpoint는 포함하지 않는다. 결과 JSON에는 각 원시 파일과 archive의 hash를 기록했다. full checkpoint가 있는 환경에서는 `docs/evidence/scripts/compare-numeric-state.py`로 같은 구성기·GRU의 가중치 일치까지 다시 검사할 수 있다.
