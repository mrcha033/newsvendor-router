# 데이터와 모델 출처

| 자료 | 공식 출처 | 고정 및 처리 |
|---|---|---|
| TAT-QA | [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) | commit `870accc41953dcde885aabeb963d94aabdc0fbc3`; official train 120건/dev 80건; CC BY 4.0 |
| ShARC | [official data](https://sharc-data.github.io/data.html) | official ZIP의 SHA-256 보존; train 120건/dev 80건; CC BY-SA 3.0 |
| FreshRetailNet-50K | [Dingdong-Inc dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) | Hub commit `08c1fab7f9257bc73679d415d65d644165d351d4`; 50개 매장/상품 시계열; CC BY 4.0 |
| all-MiniLM-L6-v2 | [sentence-transformers model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; safetensors; 384차원 masked mean pooling; Apache 2.0 |

TAT-QA는 완전한 표·문단 context를 source group으로 사용한다. raw schema에서 원래 보고서의 식별자를 복원할 수 없으므로 전체 보고서 단위 분할을 보장했다고 표현하지 않는다. ShARC는 규칙 snippet hash로 묶고 두 공식 분할에서 중복 source를 제외한 뒤 표본 수를 채운다. 정답·derivation·mapping·evidence는 모델 입력에서 제거한다.

FreshRetail rows API에는 revision 고정 기능이 없다. Hub SHA를 수집 전후 확인하고 원시 API 응답 각각의 hash를 기록하므로 **실제 수집 snapshot**을 식별할 수 있다. API 응답이 해당 parquet commit과 동일하다고 완전히 보장하지 않는다. 처음 60일 이상 관측된 50개 store/product를 선택하고 각 시계열을 시간 순서로 60/20/20 분할한다. 수집 표본은 90일 × 50 = 4,500개 daily row다. `sale_amount`는 정규화 판매이며 stockout row를 실제 잠재수요로 사용하지 않는다.

원시 자료·가공 자료·encoder weights는 Git에 재배포하지 않는다. 수집 명령과 license·URL·SHA·표본 수는 manifest로 보존한다. 원본 proposal의 Newsvendor 참고자료는 Lariviere & Porteus의 논문이다. 교과서로 표기하지 않는다.

상보적 영어 suite는 위의 축소 public 진단과 별도로 `prepare-suite`로 준비한다. `configs/complementary.json`에 Git commit과 파일 SHA-256을 고정하고 [준비 manifest](../cases/complementary/manifest.json)에 실제 수집 hash·분할·개수를 기록한다.

| 추가 원천 | 고정 및 원천의 성격 |
|---|---|
| [CUAD](https://www.atticusprojectai.org/cuad/) | `The-Atticus-Project/cuad@67faa0e6023b04fcaae6cc09497ab00e5d63a2a2`, CC BY 4.0. 510개 실제 상업 계약 중 제목의 supply/manufacturing/distribution/purchase 기준 75개를 사용한다. 기존 전문가의 근거 주석을 유지하며 발주 모수로 다시 주석하지 않는다. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | `stanfordnlp/contract-nli@eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`, CC BY 4.0. 607개 NDA 중 60개, 기존 함의/모순/미기재와 근거. 공급 계약의 비용 정답으로 사용하지 않는다. |
| [OR-ShARC](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) | commit `309c4ea3a7cc4c7a815d7f46292d91fc9a0de581`, CC BY-SA 3.0. 실제 영어 정부 규칙, crowd 작성 상황·대화. 전체 규칙 collection을 제공하지만 gold snippet ID와 evidence는 입력에 없다. 실제 조달 질의응답 로그로 표기하지 않는다. |
| [ABCD](https://github.com/asappresearch/abcd) | commit `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`, MIT. 가상의 소매 업체 정책을 사용한 사람 간 역할극이다. 영어 원문 prefix와 전체 정책/도구를 사용하고 scenario·flow·subflow·targets·후속 대화는 제공하지 않는다. |

TAT-QA의 60개 context/240개 질문과 FreshRetail의 50개 시계열도 같은 suite에 포함한다. FreshRetail은 450만 행의 여러 위치에서 60개 응답을 수집해 매장 범위를 넓히고 과거 60일/후속 7일을 분리한다. 같은 매장 또는 상품은 연결해 분할한다. 이 suite의 판매 평가는 관측 판매 예측이며, 비검열·검열 행 오차를 각각 보고한다. 질문 비용이나 실제 발주 결과, 잠재수요 정답은 없다.

Suite의 원천에 기존 사람 주석이 있어도 새 업무 사례를 검토한 사람 수는 0이다. 영어 공개 계약과 역할극·crowd 대화의 성과를 실제 조직의 전체 의사결정 성과로 확장하지 않는다. 서로 다른 원천을 연결한 합성 발주 사례는 이번 suite에 포함하지 않았다.

외부 API의 [TypeSafe choice](https://docs.typesafe.ai/primitives/choice)와 [SGLang decision models](https://docs.sglang.io/docs/supported-models/decision_models) 명세를 확인해 `/v1/systemone` 호출을 구성했다. vendor confidence나 내부 학습 방식에 관한 설명을 검증된 학술 결과로 취급하지 않는다.

의존성은 `uv.lock`으로 고정하고 Linux에서는 PyTorch CPU index를 사용한다. Endor 검토 profile의 exact-version risk 판정은 서비스/CLI가 제공되지 않아 `UNKNOWN`이다. PyPI version 확인과 설치 성공은 보안 승인 판정을 의미하지 않는다.
