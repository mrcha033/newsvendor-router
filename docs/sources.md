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

외부 API의 [TypeSafe choice](https://docs.typesafe.ai/primitives/choice)와 [SGLang decision models](https://docs.sglang.io/docs/supported-models/decision_models) 명세를 확인해 `/v1/systemone` 호출을 구성했다. vendor confidence나 내부 학습 방식에 관한 설명을 검증된 학술 결과로 취급하지 않는다.

의존성은 `uv.lock`으로 고정하고 Linux에서는 PyTorch CPU index를 사용한다. Endor 검토 profile의 exact-version risk 판정은 서비스/CLI가 제공되지 않아 `UNKNOWN`이다. PyPI version 확인과 설치 성공은 보안 승인 판정을 의미하지 않는다.
