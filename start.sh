#!/bin/bash
# SoyLM - Start vLLM + App
# vLLM on 8100 (SoyLM-optimized), App on 8080

set -e

VLLM_BIN="/home/soy/nemotron/.venv/bin/vllm"
PARSERS="/home/soy/nemotron/parsers"
VLLM_PORT=8100
APP_PORT=8080
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"

# Check if 8100 is already in use
if ss -tlnp | grep -q ":${VLLM_PORT}\b"; then
    echo "[!] Port ${VLLM_PORT} is already in use. Kill existing process first or use it as-is."
    echo "    To use existing: just run the app with 'uv run uvicorn app:app --port ${APP_PORT}'"
    read -p "    Start app only? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        cd "$(dirname "$0")"
        exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port ${APP_PORT}
    fi
    exit 1
fi

echo "=== Starting vLLM (port ${VLLM_PORT}) ==="
$VLLM_BIN serve "$MODEL" \
    --port ${VLLM_PORT} \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32 \
    --reasoning-parser nemotron_nano_v2 \
    --reasoning-parser-plugin "${PARSERS}/nemotron_nano_v2_reasoning_parser.py" \
    --enable-prefix-caching \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for vLLM health
echo "Waiting for vLLM..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM ready (${i}s)"
        break
    fi
    sleep 2
done

if ! curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo "[ERROR] vLLM failed to start within 240s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "=== Starting SoyLM (port ${APP_PORT}) ==="
cd "$(dirname "$0")"
.venv/bin/uvicorn app:app --host 0.0.0.0 --port ${APP_PORT}

# Cleanup: stop vLLM when app exits
kill $VLLM_PID 2>/dev/null
