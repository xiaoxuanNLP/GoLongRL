#!/bin/bash
set -e

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/results/outputs_frames/${MODEL_NAME}}"

JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:?JUDGE_MODEL_ID environment variable must be set}"
JUDGE_API_URL="${JUDGE_API_URL:?JUDGE_API_URL environment variable must be set}"

JUDGE_ARGS="{\"model_id\":\"${JUDGE_MODEL_ID}\",\"api_url\":\"${JUDGE_API_URL}\",\"api_key\":\"${JUDGE_API_KEY}\"}"

echo $MODEL_PATH

TP_SIZE=8
PORT=8001

start_model_service() {
    echo "启动 vLLM 服务..."
    echo "使用端口: ${PORT}"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len 182272 \
        --port "${PORT}" \
        > vllm_server_frames.log 2>&1 &

    echo "vLLM 已启动后台进程 (日志: vllm_server_frames.log)"
    echo "等待服务启动..."
    VLLM_PID=$!
    echo "vLLM PID: ${VLLM_PID}"
    local max_wait_time=600
    local wait_interval=5
    local elapsed_time=0
    local health_url="http://127.0.0.1:${PORT}/v1/completions"

    while [ $elapsed_time -lt $max_wait_time ]; do
        response=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${health_url}" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"${MODEL_NAME}"'",
                "prompt": "你好，请介绍一下你自己。",
                "max_tokens": 10,
                "temperature": 0
            }') || true

        if [ "$response" -eq 200 ]; then
            export base_url="http://127.0.0.1:${PORT}/v1"
            echo "vLLM 服务已就绪，访问地址: $base_url"
            return 0
        fi

        sleep $wait_interval
        elapsed_time=$((elapsed_time + wait_interval))
    done

    echo "错误: vLLM 服务启动超时 (${max_wait_time}s)，请检查 vllm_server_frames.log"
    exit 1
}

start_model_service
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT

evalscope eval \
  --model ${MODEL_NAME} \
  --eval-type openai_api \
  --api-url http://127.0.0.1:${PORT}/v1 \
  --api-key EMPTY \
  --datasets frames \
  --dataset-args '{"frames": {"filters": {"remove_until": "</think>"}, "max_length": 131072, "truncation_strategy": "middle"}}' \
  --generation-config '{"max_tokens":51200,"temperature":0.7,"top_p":0.95}' \
  --judge-model-args "${JUDGE_ARGS}" \
  --repeats 1 \
  --eval-batch-size 8 \
  --use-cache ${OUTPUT_PATH} \
  --work-dir ${OUTPUT_PATH}

# 关闭 vLLM 服务
kill ${VLLM_PID} && echo "vLLM 服务已关闭"
wait ${VLLM_PID} 2>/dev/null || true