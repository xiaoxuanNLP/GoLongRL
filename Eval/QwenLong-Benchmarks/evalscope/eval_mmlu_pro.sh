#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/results/outputs_mmlu_pro/${MODEL_NAME}}"

TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-8004}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "========== MMLU-Pro Eval: ${MODEL_NAME} =========="

pkill -9 -f "vllm" 2>/dev/null || true
sleep 3
ls /dev/shm/ 2>/dev/null | grep -i vllm | xargs -I{} rm -f /dev/shm/{} 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=${GPUS} \
setsid python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    > "${SCRIPT_DIR}/vllm_server_mmlu_pro.log" 2>&1 &

VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= -p ${VLLM_PID} 2>/dev/null | tr -d ' ')
trap 'kill -- -${VLLM_PGID:-} 2>/dev/null || true; pkill -9 -f "vllm" 2>/dev/null || true' EXIT
echo "vLLM PID=${VLLM_PID}"
echo "等待服务启动..."

elapsed_time=0
while [ $elapsed_time -lt 600 ]; do
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0}') || true
    if [ "$response" -eq 200 ]; then
        echo "vLLM 服务已就绪"
        break
    fi
    sleep 5
    elapsed_time=$((elapsed_time + 5))
done

if [ $elapsed_time -ge 600 ]; then
    echo "错误: vLLM 启动超时，检查 vllm_server_mmlu_pro.log"
    kill -- -${VLLM_PGID} 2>/dev/null
    exit 1
fi

evalscope eval \
  --model "${MODEL_NAME}" \
  --eval-type openai_api \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --datasets mmlu_pro \
  --dataset-args '{"mmlu_pro": {"filters": {"remove_until": "</think>"}}}' \
  --generation-config '{"max_tokens":32768,"temperature":0.7,"top_p":0.95}' \
  --repeats 1 \
  --eval-batch-size 2 \
  --use-cache "${OUTPUT_PATH}" \
  --work-dir "${OUTPUT_PATH}"

echo "评测完成，关闭 vLLM 服务 (PGID=${VLLM_PGID})..."
kill -- -${VLLM_PGID} 2>/dev/null || true
sleep 3
pkill -9 -f "vllm" 2>/dev/null || true
sleep 3
ls /dev/shm/ 2>/dev/null | grep -i vllm | xargs -I{} rm -f /dev/shm/{} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
echo "========== ${MODEL_NAME} 完成 =========="
