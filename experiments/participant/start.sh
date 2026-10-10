#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --frozen --extra experiment
uv run --extra experiment python experiments/participant/run.py start \
  --materials results/human-materials-v2 \
  --directory results/human-study-live --host 0.0.0.0 "$@"
