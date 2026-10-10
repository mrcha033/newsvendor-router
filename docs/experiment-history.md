# 이전 실험과 원시 근거의 색인

현재 연구 질문·상태는 [통합 연구계획](research.md), 최신 분석은 [decision-effects.md](decision-effects.md)를 기준으로 읽는다. 아래 문서는 각 실험 당시의 설정·성공·실패·원시 측정 기록이다. 문서 안의 ‘현재’와 ‘다음’은 당시 시점을 뜻하며 새 연구의 진행 상태가 아니다.

## 기록 보존

2026-10-10 통합 전 README·범위·명세·목표·검증 문서·Word 제안서·출처 metadata를 [원본 압축](evidence/artifacts/research-documents-before-economics-v1.tar.xz)에 바이트 그대로 보존했다. [파일 hash](evidence/research-document-archive.json)와 기준 commit `3dba348`로 대조할 수 있다. 개별 실험 문서와 evidence·가중치·실패 기록은 삭제하지 않았다.

## 실험 문서

| 문서 | 당시 기록의 내용 |
| --- | --- |
| [action-adaptation](action-adaptation.md) | Encoder 변경 뒤 행동 가치 head의 재학습 |
| [action-precision](action-precision.md) | 행동 가치 학습과 추론의 정밀도 일치 |
| [constructor-training](constructor-training.md) | 모수 학습과 실제 추출 입력의 일치 |
| [demand-concentration](demand-concentration.md) | 날짜별 수요 배분 가정 점검 |
| [demand-observations](demand-observations.md) | 날짜별 품절 관측을 사용하는 수요 학습 |
| [demand-selection](demand-selection.md) | 수요 모수의 출처별 학습 횟수 선택 |
| [demand-selector](demand-selector.md) | 수요 분포 선택의 평균 손실 학습 |
| [language-retention](language-retention.md) | 문서 학습에서 기존 추출 능력 보존 |
| [native-history](native-history.md) | native-v3와 이전 실험의 재현 기록 |
| [operand-selection](operand-selection.md) | 계산할 숫자를 함께 고르는 작은 head |
| [parameter-contrasts](parameter-contrasts.md) | 모수 근거의 적용 범위와 문서 버전 |
| [parameter-decoding](parameter-decoding.md) | 관측 근거에서 발주 모수를 다시 선택하는 경로 |
| [parameter-state](parameter-state.md) | 매니저 응답 후 모수 상태의 Train 진단 |
| [policy-state](policy-state.md) | 정보 요청 head에 현재 모수를 전달하는 경로 |
| [recovery-replay](recovery-replay.md) | 응답 후 경로를 가치 학습에 추가한 비교 |
| [recovery-sampling](recovery-sampling.md) | 응답별 학습 빈도 보정과 전체 재학습 비교 |
| [recovery-scope](recovery-scope.md) | 관측한 응답 실패에 한정한 행동 보정 |
| [recovery-stability](recovery-stability.md) | 숫자 상태 보정과 재학습 안정성 |
| [replay-coverage](replay-coverage.md) | 공개 Train 범위를 복원한 모수 구성 학습 |
| [response-evaluation](response-evaluation.md) | 같은 매니저 응답 조건에서 정책 비교 |
| [response-state](response-state.md) | 관측한 매니저 응답을 모수 상태에 반영 |
| [response-supervision](response-supervision.md) | 질문의 기대 손실을 학습하는 응답 표본 |
| [retail-holdout](retail-holdout.md) | 새 판매 출처에서 고정 모델 검증 |
| [retail-train-expansion](retail-train-expansion.md) | 기존 출처 안에서 판매 학습 이력 확대 |
| [source-validity](source-validity.md) | 새로 추출한 근거의 버전과 충돌 검증 |
| [span-supervision](span-supervision.md) | 공개 문서의 정답 span 경계 수정 |
| [state-value](state-value.md) | 구성한 발주 상태로 행동 가치를 계산하기 |
| [structured-history](structured-history.md) | 범용 도구 확장과 이전 구조화 학습 이력 |
| [tool-repair](tool-repair.md) | 도구·상태 연결과 행동 가치 학습 수정 |
| [tool-upgrade-results](tool-upgrade-results.md) | 도구·상태·절차 실험 결과 |
| [tool-upgrade](tool-upgrade.md) | 도구·상태·절차 학습 변경 |

ABCD와 범용 도구 지표는 보조 실험이다. 기존 Test·동결 holdout은 이미 사용한 평가라는 이력을 유지하며 새로운 확증 자료로 재명명하지 않는다. 같은 사례에서 기록한 시나리오 변형·응답 mask·학습 seed의 반복을 독립 경제 표본으로 세지 않는다.
