# 검증 기록

2026-10-10 현재 연구는 [통합 연구계획](research.md)을 따른다. 과거 검증 문서의 원문은 [보존 압축](evidence/artifacts/research-documents-before-economics-v1.tar.xz)에 있다. 아래는 문서 통합과 첫 탐색 분석을 실제로 실행한 기록이다.

## 이번 변경의 검증

| 항목 | 결과와 범위 |
| --- | --- |
| 전체 Python 테스트 | 387개 통과. 새 짝짓기·조건 균형·가용성·출처 가중·수리 항등식 검사 포함 |
| 정적·포맷 검사 | Ruff와 diff 검사 통과 |
| 출처 분리 | complementary, core-retail-train-v1, core-tatqa-train-v1, core-retail-holdout-v1, core-retail-linked-v1 검사 통과 |
| 수치 smoke | 600건 corpus 감사와 30개 시나리오, 잘못된 발주 진행 0건 |
| 공개 측정 분석 | 3,960행·14개 짝지은 회귀 및 출처 제외 분석; 기존 Dev의 재분석 |
| 수리 모형 | 연속 균등수요 내부해 6조건의 곡률·선호 민감도 항등식 통과 |
| 독립 재현 | 원래 소스·결과·네트워크 접근 차단; 행의 바이트·분석·수치 검사·소스/입력 hash 일치 |
| Word 배포본 | 현재 연구 Markdown에서 생성; 표시 수식 7개가 Word 수식으로 변환됨; source.json에 문서·원문·exporter hash 보존 |
| 원본 문서 보존 | 이전 문서 7개의 바이트와 압축 member hash 대조 완료 |

전체 수치와 검증 로그의 파일 hash는 [분석 evidence](evidence/research-economics-results.json)에 있다. 재현 준비에서 uv.lock을 누락한 첫 실행, source 감사 래퍼의 잘못된 경로, 첫 Word 변환의 수식 경고도 보존했다. 분석식·데이터·측정값을 바꾸지 않고 준비 경로와 수식 표기를 수정한 뒤 확인했다.

## 실행하지 않은 것

이번 변경에서 모델을 학습하거나 새 GPU 추론을 실행하지 않았다. 기존 Test와 27개 출처 holdout의 모델 평가도 반복하지 않았다. 출처 검사는 분할·입력에 대한 감사이며 새 성능 평가가 아니다.

AI 사용×질문 허용의 새 2×2 실험, 새 출처 확증, 실제 참여자 실험은 미실행이다. 현재 회귀는 모델 구성 수정의 탐색 분석이며 AI 도입 전체·인간 행동·조직 효과의 확인으로 표현하지 않는다.

## 과거 고정 모델의 검증

L40S의 모수 구성 수정은 기존 374개 테스트와 540경로 재현을 통과했다. 과거 가중치·입력·원시 측정과 제한은 [해당 비교](parameter-decoding.md) 및 [이력 색인](experiment-history.md)에 그대로 남긴다. 이 수치를 이번 CPU 분석의 새 모델 성능으로 옮겨 적지 않는다.
