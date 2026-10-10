# 검증 기록

2026-10-10 현재 연구는 [통합 연구계획](research.md)을 따른다. 과거 검증 문서의 원문은 [보존 압축](evidence/artifacts/research-documents-before-economics-v1.tar.xz)에 있다. 첫 문서 통합·탐색 회귀 기록은 [기존 evidence](evidence/research-economics-results.json)에 고정돼 있다.

## 이번 변경의 검증

| 항목 | 결과와 범위 |
| --- | --- |
| 전체 Python 테스트 | 407개 통과. 비AI·2×2 짝짓기, 참가자 배정·누락·철회·내보내기·기록 무결성 포함 |
| 정적·포맷 검사 | Ruff와 diff 검사 통과 |
| 출처 분리 | complementary, core-retail-train-v1, core-tatqa-train-v1, core-retail-holdout-v1, core-retail-linked-v1 검사 통과 |
| 수치 smoke | 600건 corpus 감사와 30개 시나리오, 잘못된 발주 진행 0건 |
| 기존 공개 측정 분석 | 3,960행·14개 짝지은 회귀 및 출처 제외 분석; 앞선 고정 결과 유지 |
| 수리 모형 | 연속 균등수요 내부해 6조건의 곡률·선호 민감도 항등식 통과 |
| 새 비AI 비교 | Train 154사례·616경로 검증 후 Dev 336사례·6,720경로를 L40S에서 실행 |
| 독립 GPU 재현 | 원래 작업 소스·자료·결과·네트워크 접근 차단; 8출처×5문서×4조건, 160경로의 수량·결과·초기/최종 상태 hash 일치 |
| 참여자 브라우저 | 두 집단 소프트웨어 참여 4명·연습 8건·본 과제 32건·질문 80건; 오류 0, 저장·재접속·개인 링크·마감·export·분석 확인 |
| Word 배포본 | 현재 연구 Markdown에서 생성; 표시 수식 8개를 Word 수식으로 변환. 별도 참여자 운영 안내도 Word로 제공 |
| 원본 문서 보존 | 이전 문서 7개의 바이트와 압축 member hash 대조 완료 |

새 비교·출처·테스트·원시 측정 hash는 [AI 비교 evidence](evidence/research-ai-comparison-results.json), 실험 소프트웨어·고정 과제·리허설·운영 패키지 hash는 [참여자 준비 evidence](evidence/human-experiment-readiness.json)에 둔다. 과거 387개 테스트의 문서 통합 결과와 이번 407개 테스트를 구분한다. 자동 리허설 기록은 실제 참여자 결과에 합치지 않는다.

## 실행하지 않은 것

이번 변경에서 고정 가중치의 L40S 추론은 실행했으며 신경망을 재학습하지 않았다. 기존 Test와 27개 출처 holdout의 모델 평가도 반복하지 않았다. 출처 검사는 분할·입력에 대한 감사이며 새 성능 평가가 아니다.

새 출처 확증과 실제 참여자 실험은 미실행이다. 준비한 서버는 같은 LAN/VPN용이며 공개 HTTPS 사이트를 배포하지 않았다. 현재 회귀는 특정 비AI 기준선과 AI 묶음의 기존 Dev 비교이며 일반적인 AI 도입·인간 행동·조직 효과의 확인으로 표현하지 않는다.

## 과거 고정 모델의 검증

L40S의 모수 구성 수정은 기존 374개 테스트와 540경로 재현을 통과했다. 과거 가중치·입력·원시 측정과 제한은 [해당 비교](parameter-decoding.md) 및 [이력 색인](experiment-history.md)에 그대로 남긴다. 이 수치를 이번 CPU 분석의 새 모델 성능으로 옮겨 적지 않는다.
