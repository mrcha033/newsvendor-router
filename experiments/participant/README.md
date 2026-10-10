# 참여자 실험 실행 파일

진행자는 [운영 안내](../../docs/human-study-guide.md)를 따른다. 참여자는 개인 브라우저 링크만 사용하며 이 폴더·정답 자료·진행자 키를 받지 않는다. 실제 사람의 자료는 아직 없다.

현재 작업공간과 별도 실행 ZIP에는 `results/human-materials-v2`가 준비돼 있다. `bash experiments/participant/start.sh`로 시작한다. 기본 대상은 같은 LAN/VPN의 PC 브라우저이며 운영에는 GPU가 필요 없다.

## 파일 역할

| 파일 | 역할 |
| --- | --- |
| `prepare.py` | 고정한 생성 과제와 관측된 질문 경로 100개의 실제 AI 제안 준비 |
| `protocol.json` | 처치·측정·배정·표본·중단·누락 분석의 기본 명세 |
| `run.py` / `start.sh` | 연구 초기화, 서버, 백업, 결과 export |
| `app.py` / `web/` | 서버에만 정답을 보관하고 동의·과제·질문·제출·철회 처리 |
| `analyze.py` | 내보낸 원시 기록 검증과 참여자 단위 분석·보고서 |
| `rehearse.py` | demo 표시·4명 목표인 연구에서만 자동 브라우저 리허설 |

## 새 clone에서 자료 재생성

[공개 모델 bundle](../../docs/model-bundle.md)의 recovery-v1을 복원한 뒤 다음을 실행한다. 가중치와 관측 입력을 고정한다. 현재 고정 자료는 L40S에서 생성했으므로 다른 장치의 부동소수점 결과가 바이트까지 같다고 가정하지 않는다. 이미 참여자를 받은 연구의 자료를 바꾸지 않는다.

```sh
export CUDA_VISIBLE_DEVICES="$NEWSVENDOR_GPU_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-l40s/bin/python experiments/participant/prepare.py \
  --bundle /path/to/recovery-v1 --device cuda \
  --output results/human-materials-v2
```

새 CPU 환경에서는 `uv sync --frozen --extra experiment` 후 `uv run --extra experiment python`과 `--device cpu`로 준비할 수 있다. 과제 생성 seed는 고정이고, 실제 연구에 쓸 자료·코드·설정을 초기화 시 hash로 고정한다. 미래 수요·참모수는 별도 환경 파일에 두고 모델 입력에는 관측한 내용만 준다.

## 소프트웨어 리허설

별도 폴더를 초기화하고 서버를 켠다. 다음 명령은 실제 사람을 모집하지 않는다.

```sh
uv run --extra experiment python experiments/participant/run.py start \
  --directory results/browser-rehearsal --target 4 --invitations 8 \
  --contact '소프트웨어 리허설' --demo
```

다른 터미널에서 Playwright와 Chromium을 준비한 뒤 실행한다.

```sh
uv run --extra experiment --with playwright==1.55.0 python -m playwright install chromium
uv run --extra experiment --with playwright==1.55.0 python experiments/participant/rehearse.py \
  --directory results/browser-rehearsal --output results/browser-rehearsal-check
uv run --extra experiment python experiments/participant/analyze.py \
  --export results/browser-rehearsal-check/software-rehearsal-results.zip \
  --output results/browser-rehearsal-analysis.json
```

리허설의 자동 답변·회귀 결과는 인간의 성과로 보고하지 않는다. 진행자 URL에 포함되는 키·초대 목록·운영 SQLite·백업은 공개 저장소에 넣지 않는다. 원격 HTTPS 호스팅을 추가하는 경우 별도 접속·로그 구성을 검증한다.
