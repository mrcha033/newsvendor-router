# 데이터와 모델 출처

| 자료 | 공식 출처 | 고정 및 처리 |
|---|---|---|
| TAT-QA | [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) | commit `870accc41953dcde885aabeb963d94aabdc0fbc3`; official train 120건/dev 80건; CC BY 4.0 |
| ShARC | [official data](https://sharc-data.github.io/data.html) | official ZIP의 SHA-256 보존; train 120건/dev 80건; CC BY-SA 3.0 |
| FreshRetailNet-50K | [Dingdong-Inc dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) | Hub commit `08c1fab7f9257bc73679d415d65d644165d351d4`; 50개 매장/상품 시계열; CC BY 4.0 |
| all-MiniLM-L6-v2 | [sentence-transformers model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; safetensors; 384차원 masked mean pooling; Apache 2.0 |
| ModernBERT-base — 다음 모델 후보 | [공식 모델](https://huggingface.co/answerdotai/ModernBERT-base) | 예정 revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`; 양방향 encoder 149M / 22층 / hidden 768 / 8,192 tokens; Apache 2.0 |
| ModernBERT-large — 크기 비교 후보 | [공식 모델](https://huggingface.co/answerdotai/ModernBERT-large) | 예정 revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`; 양방향 encoder 395M / 28층 / hidden 1,024 / 8,192 tokens; Apache 2.0 |

TAT-QA는 완전한 표·문단 context를 source group으로 사용한다. raw schema에서 원래 보고서의 식별자를 복원할 수 없으므로 전체 보고서 단위 분할을 보장했다고 표현하지 않는다. ShARC는 규칙 snippet hash로 묶고 두 공식 분할에서 중복 source를 제외한 뒤 표본 수를 채운다. 정답·derivation·mapping·evidence는 모델 입력에서 제거한다.

FreshRetail rows API에는 revision 고정 기능이 없다. Hub SHA를 수집 전후 확인하고 원시 API 응답 각각의 hash를 기록하므로 **실제 수집 snapshot**을 식별할 수 있다. API 응답이 해당 parquet commit과 동일하다고 완전히 보장하지 않는다. 처음 60일 이상 관측된 50개 store/product를 선택하고 각 시계열을 시간 순서로 60/20/20 분할한다. 수집 표본은 90일 × 50 = 4,500개 daily row다. `sale_amount`는 정규화 판매이며 stockout row를 실제 잠재수요로 사용하지 않는다.

대용량 원시 다운로드와 encoder weights는 Git에 재배포하지 않는다. 재현에 필요한 가공 입력·원래 주석·공개 조회 collection은 `cases/evaluation.tar.xz`에 분리 파일로 보존한다. `cases/evaluation.json`에 파일별 SHA-256, 원천 URL/revision/라이선스를 기록한다. CUAD·ContractNLI·TAT-QA·FreshRetail 부분은 CC BY 4.0, OR-ShARC 부분은 CC BY-SA 3.0, ABCD·τ² retail 부분은 MIT를 그대로 적용한다. 가공 내용은 원래 문서를 공통 schema에 담고 출처 묶음으로 재분할한 것이다. 원본 proposal의 Newsvendor 참고자료는 Lariviere & Porteus의 논문이다. 교과서로 표기하지 않는다.

상보적 영어 suite는 위의 축소 public 진단과 별도로 `prepare-suite`로 준비한다. `configs/complementary.json`에 Git commit과 파일 SHA-256을 고정하고 [준비 manifest](../cases/complementary/manifest.json)에 실제 수집 hash·분할·개수를 기록한다.

| 추가 원천 | 고정 및 원천의 성격 |
|---|---|
| [CUAD](https://www.atticusprojectai.org/cuad/) | `The-Atticus-Project/cuad@67faa0e6023b04fcaae6cc09497ab00e5d63a2a2`, CC BY 4.0. 510개 실제 상업 계약 중 제목의 supply/manufacturing/distribution/purchase 기준 75개를 사용한다. 기존 전문가의 근거 주석을 유지하며 발주 모수로 다시 주석하지 않는다. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | `stanfordnlp/contract-nli@eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`, CC BY 4.0. 607개 NDA 중 60개, 기존 함의/모순/미기재와 근거. 공급 계약의 비용 정답으로 사용하지 않는다. |
| [OR-ShARC](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) | commit `309c4ea3a7cc4c7a815d7f46292d91fc9a0de581`, CC BY-SA 3.0. 실제 영어 정부 규칙, crowd 작성 상황·대화. 전체 규칙 collection을 제공하지만 gold snippet ID와 evidence는 입력에 없다. 실제 조달 질의응답 로그로 표기하지 않는다. |
| [ABCD](https://github.com/asappresearch/abcd) | commit `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`, MIT. 가상의 소매 업체 정책을 사용한 사람 간 역할극이다. 영어 원문 prefix와 전체 정책/도구를 사용하고 scenario·flow·subflow·targets·후속 대화는 제공하지 않는다. |

TAT-QA의 60개 context/240개 질문과 FreshRetail의 50개 시계열도 같은 suite에 포함한다. FreshRetail은 450만 행의 여러 위치에서 60개 응답을 수집해 매장 범위를 넓히고 과거 60일/후속 7일을 분리한다. 같은 매장 또는 상품은 연결해 분할한다. 수요 모형은 다음 7일 총수요의 종류 head와 종류별 모수 head를 공동 학습한다. 연속·비음수 판매 단위에 맞춰 절단 정규·로그정규·Weibull 후보와 각 종류의 0수요 질량을 사용한다. 선택한 종류·모수와 그 분포를 이산화한 `F`를 출력한다. 품절이 없는 기간은 관측 총판매의 density, 품절 기간은 관측 총판매 이상의 survival likelihood를 사용한다. 학습·검증·테스트의 rolling window도 원래 매장·상품 묶음 분할을 유지하며 입력 cutoff 뒤의 관측을 feature에 넣지 않는다. 정확한 CRPS·발주 손실은 비품절 기간에서 측정하고, 품절 기간은 tail likelihood와 손실 하한을 기록한다. 부족/잉여 비용 비율 1:1·3:1·9:1은 공개 실험 설정이며 정규화 판매 단위에 적용한다.

Suite의 원천에 기존 사람 주석이 있어도 새 업무 사례를 검토한 사람 수는 0이다. 영어 공개 계약과 역할극·crowd 대화의 성과를 실제 조직의 전체 의사결정 성과로 확장하지 않는다. 서로 다른 원천을 연결한 합성 발주 사례는 이번 suite에 포함하지 않았다.

[τ²-bench retail](https://github.com/sierra-research/tau2-bench)은 revision `5bfa7e37b36656b37dc6d022156be6563c1007f3`, MIT를 사용한다. `configs/orders.json`에 DB·정책·과제·공식 분할·원래 도구/모델 코드의 SHA-256을 고정한다. 데이터는 연결된 모의 고객·주문·상품이다. 실무에서 수집한 고객 기록이나 공급자 발주 자료로 표기하지 않는다. 참고 행동은 최종 DB 채점만 읽고 사용자 시나리오는 공통 사용자 simulator만 읽는다. 공식 Train/Test의 고객 중복 22명 때문에 53개 고객 묶음으로 재분할하며 공식 분할도 manifest에 보존한다.

`src/newsvendor/retail/`은 위 revision의 Pydantic 자료 모델과 도구를 MIT notice와 함께 적용한다. 프레임워크 DB를 BaseModel로, toolkit 생성자를 로컬 DB로 교체하고 데코레이터·미사용 실행 예제를 제거했다. 이미 길이를 검사하는 zip에 strict를 추가했다. 나머지 도구 함수의 AST가 원천과 같은지 검사했다. 원래 환경 채점도 참고 도구 오류를 기록하고 계속하므로 동일하게 처리한다. 실패한 참고 변경만 있고 성공한 변경이 없는 과제는 주석 모호성을 표시한다.

현재 GPU 비교 모델은 [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), revision `a09a35458c702b33eeacc393d103063234e8bc28`, Apache 2.0이다. 동일 고정 가중치를 typed·agent·기존 native-v3의 인자·질문·발화 helper 및 공통 사용자 simulator에 사용한다. 주문 경로의 helper는 누락 인자 질문과 변경 확인 요청도 구성한다. 다음 학습형은 이를 내부 인자·행동 head와 템플릿으로 처리하고 Qwen helper를 호출하지 않는다. Qwen은 비교군·공통 simulator·참고 judge에 유지한다. 사용자 simulator와 사후 NL assertion judge는 비교 방법의 입력에 숨겨진 목표·정답을 전달하지 않는다. MiniLM의 고정 256-token chunk 평균은 현재 구현 기록이다.

다음 모델의 ModernBERT 출처는 Warner, Chaffin, Clavié 등 (2024), [Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663)와 위 공식 모델 카드·config이다. 표의 revision은 계획에 사용할 후보 가중치를 고정한 것이며 현재 `configs/native.json`·저장 가중치는 MiniLM 기반이다. 토큰 표현을 보존한 encoder 공동 미세조정, 256차원 2층 상태 결합, 2층 GRU hidden 128 및 공유 구조화 head는 다음 구현 계획이다. 152–155M 전체 파라미터는 설계 예산이며, 동일 head의 large를 약 400M 크기 비교군으로 둔다. 확보한 ABCD 원천의 Train 대화 8,034개 등으로 학습 자료를 확대하되 기존 Dev/Test와 연결된 출처·template를 제외한다. 독립 공개 문서를 FreshRetail의 SKU·기간에 임의로 연결하지 않는다.

외부 API의 [TypeSafe choice](https://docs.typesafe.ai/primitives/choice)와 [SGLang decision models](https://docs.sglang.io/docs/supported-models/decision_models) 명세를 확인해 `/v1/systemone` 호출을 구성했다. vendor confidence나 내부 학습 방식에 관한 설명을 검증된 학술 결과로 취급하지 않는다.

의존성은 `uv.lock`으로 고정하고 Linux에서는 PyTorch CPU index를 사용한다. Endor 검토 profile의 exact-version risk 판정은 서비스/CLI가 제공되지 않아 `UNKNOWN`이다. PyPI version 확인과 설치 성공은 보안 승인 판정을 의미하지 않는다.
