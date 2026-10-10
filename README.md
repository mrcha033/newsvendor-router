# AI와 발주 의사결정의 경제적 효과

문서와 판매 이력에서 발주 모수를 구성하고, 부족한 정보를 매니저에게 요청하는 Newsvendor 연구입니다. **AI의 정보처리·질문이 어떤 조건에서 의사결정 손실을 줄이는지 수리 모형과 통제 실험·회귀분석으로 설명**하는 데 중심을 둡니다.

## 읽는 순서

| 문서 | 역할 |
| --- | --- |
| [통합 연구계획](docs/research.md) · [Word 배포본](docs/proposal.docx) | 연구 질문, 수리 모형, 가설, 효과 식별, 현재 근거와 한계 |
| [실험 명세](docs/protocol.md) | 분석 단위·비교군·손실 정의·자료 분리와 실행 명령 |
| [첫 탐색 회귀](docs/decision-effects.md) | 기존 공개 Dev의 구성 수정 효과와 출처별 민감도 |
| [AI×질문 비교 결과](docs/ai-comparison.md) | 비AI 기준선과 고정 AI의 네 조건을 L40S에서 비교 |
| [참여자 실험 운영](docs/human-study-guide.md) · [Word 안내서](docs/human-study-guide.docx) | 모집문·개인 링크·동의·과제·저장·분석을 갖춘 한국어 실험 |
| [모델과 추론 실행](docs/model-bundle.md) | 선택 가중치·토크나이저·복원·추론 |
| [평가·완료 기준](docs/performance-goal.md) · [검증 기록](docs/verification.md) | 무엇을 확인했으며 무엇이 아직 미완료인지 |
| [실험 이력 색인](docs/experiment-history.md) | 이전 성공·실패·원시 기록·보조 ABCD 실험 |

2026-10-10 문서를 통합했습니다. 이전 문서 일곱 개의 원본 바이트는 [보존 압축](docs/evidence/artifacts/research-documents-before-economics-v1.tar.xz)과 [파일 hash](docs/evidence/research-document-archive.json)에 남깁니다. 과거 가중치·측정·분할을 덮어쓰지 않습니다.

## 현재 상태

선택한 실험 모델은 **ModernBERT-base + 2층 수요 GRU + 모수·근거·상태·정보 요청 head**입니다. 공개 `recovery-v1` 가중치와 `--parameter-decoding scoped` 추론 경로를 사용합니다. 실제로 구성한 `F/c/p/v/b`를 공통 optimizer에 전달하며 자연어 생성은 필요하지 않습니다.

L40S의 최근 모수 구성 비교에서는 다른 상품·기간·구버전 문서가 섞였을 때 정확도·질문·발주 손실이 개선됐고 원래 조건은 유지됐습니다. 이는 고정 가중치의 추론 수정 효과이며 AI 도입 전체의 효과는 아닙니다. [전체 비교와 재현](docs/parameter-decoding.md)

이번 탐색 분석은 공개 결과 3,960행으로 14개 짝지은 회귀·출처 제외 분석을 실행합니다. 생성 환경에서는 평균 손실이 낮아져도 조건별 60건 중 7·9·10건은 악화됐습니다. 판매의 다른 상품 조건에서 종료 손실 감소는 한 사례에 집중됐습니다. 평균 효과와 이질성을 함께 보고합니다. 현재 자료는 이미 개발에 사용한 Dev이며 실제 사람·조직 효과를 입증하지 않습니다.

추가로 비AI 기준선의 Train 검증과 AI 사용 × 매니저 질문 허용의 **2×2 파일럿을 완료**했습니다. 원래 문서에서 질문 허용 시 평균 총손실은 비AI 72.1592, AI 20.9941입니다. 다만 문서 혼입에서 질문을 금지하면 AI가 더 나빴습니다. 주 분석의 정확한 손실은 25조건·고유 기간 5개·출처 4개이며 상태·질문은 240사례입니다. 선호 미선택 층까지 합친 이전 35조건·336사례와 구분합니다. [분모·한계·원시 기록](docs/ai-comparison.md)

**실제 참여자 실험도 실행 준비를 마쳤습니다.** AI 없는 사람과 AI 지원을 받는 사람을 자동 배정하며 연습 2개·본 과제 8개, 개인 링크·재접속·삭제·결과 분석을 제공합니다. 실제 사람 데이터는 아직 없습니다. 이미 평가한 Test와 27개 출처 holdout을 새 확증 자료로 재사용하지 않습니다.

## 참여자 모집 후 바로 실행

고정 과제와 모델 제안이 포함된 실행 패키지에서 다음을 실행합니다. 현재 작업공간에는 `results/human-materials-v2`가 준비돼 있습니다. 처음 시작할 때 연구 문의 연락처를 입력하며 보상·모집 목표는 모집 전에 지정합니다.

```sh
bash experiments/participant/start.sh
```

진행자 화면에서 개인별 완성 링크 CSV를 내려받아 한 명에게 한 링크를 전달합니다. 기본은 같은 LAN/VPN·PC 브라우저이며 공개 인터넷 사이트 배포는 포함하지 않습니다. 운영에는 GPU가 필요 없습니다. [모집·운영·분석 안내](docs/human-study-guide.md), [패키지와 검증 hash](docs/evidence/human-experiment-readiness.json)를 참조하세요. 새 clone에는 개인 배포용 과제 ZIP이 없으므로 [자료 재생성 명령](experiments/participant/README.md)을 먼저 따릅니다.

## 공개 자료만으로 탐색 분석 재현

Python 3.12·CPU PyTorch를 사용하며 GPU나 모델 가중치가 필요하지 않습니다.

```sh
uv sync --frozen
uv run python scripts/analyze_decisions.py \
  --config configs/research-economics-v1.json \
  --output results/decision-effects-reproduction
```

입력 archive·원시 행·분할·출처 hash와 짝지은 사례를 검사합니다. 출력 디렉터리가 이미 있으면 중단해 이전 결과를 보존합니다. 분석은 모델을 학습하거나 새 Test 추론을 하지 않습니다. [분석 자료·회귀식·실행 결과](docs/decision-effects.md)

모델의 별도 추론은 [공개 bundle 안내](docs/model-bundle.md), 기존 L40S 학습 재현은 [실험 명세](docs/protocol.md)를 따릅니다. 기존 cold-start 설정이 현재 선택 가중치를 자동 재현한다고 주장하지 않습니다.

## 검증

```sh
uv sync --frozen --extra experiment
uv run --extra experiment pytest -q
uv run ruff check src tests scripts/analyze_decisions.py
uv run newsvendor restore-eval
uv run python scripts/run_research.py \
  --config configs/research-from-scratch.json \
  --stage check --output results/research-data-check
uv run newsvendor smoke --config configs/full.json
```

[CI](.github/workflows/ci.yml)는 코드·테스트·출처 분리·수치 smoke를 검사합니다. GPU 성능 실험은 CI에 포함되지 않습니다. [공개 자료·모델의 출처](docs/sources.md), [이전 MiniLM·Qwen·native-v3 실행](docs/native-history.md)은 별도 보존합니다.
