#!/usr/bin/env bash
# Serve the gpt-oss-120B reward-hacker checkpoints as an OpenAI endpoint.
#   ./serve.sh [PORT]     (default 8000)
# Then set in any OpenAI client:
#   base_url = http://127.0.0.1:<PORT>/v1   api_key = anything
#   model    = <a `path` from models.json>, e.g. the step-856 tinker:// URI
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8000}"
exec uv run python server.py \
  --renderer gpt_oss_medium_reasoning \
  --tokenizer openai/gpt-oss-120b \
  --host 0.0.0.0 --port "$PORT"
