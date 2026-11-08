#!/usr/bin/env bash
# Create the pinned Python 3.12 environment (Linux, CUDA >= 12.6 driver) and fetch the supplied data.
#   DATA_URL=<ATML-PA3-Data Google Drive folder URL> bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
EOF

# Task 3 rollouts (ATML-PA3-Data): not redistributed in this repository
if [ ! -f data/supplied/README.MD ]; then
  if [ -z "${DATA_URL:-}" ]; then
    echo "Place ATML-PA3-Data in data/supplied/ or set DATA_URL to the course Drive folder." >&2
    exit 1
  fi
  python -m gdown --folder "$DATA_URL" -O data/supplied
fi
ls -la data/supplied/*
