#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-sglang056}"

# ============================================================
# 前置依赖: BFCL-V4 评测需要安装指定版本 bfcl-eval
# pip install bfcl-eval==2025.10.27.1
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/results/outputs_bfcl_v4/${MODEL_NAME}}"
PORT=7810
GPUS="0,1,2,3,4,5,6,7"

echo $MODEL_PATH

TP_SIZE=8
start_model_service() {
    echo "启动 SGLang 服务..."
    echo "使用端口: ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPUS} \
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --tp "${TP_SIZE}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --tool-call-parser qwen25 \
        --reasoning-parser qwen3 \
        --max-running-requests 512 \
        --context-length 182272 \
        --mem-fraction-static 0.8 \
        > sglang_server_bfcl_v4_${MODEL_NAME}.log 2>&1 &

    SGLANG_PID=$!
    export SGLANG_PID
    echo "SGLang 已启动后台进程 PID=${SGLANG_PID} (日志: sglang_server_bfcl_v4_${MODEL_NAME}.log)"
    echo "等待服务启动..."

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
            }')

        if [ "$response" -eq 200 ]; then
            export base_url="http://127.0.0.1:${PORT}/v1"
            echo "SGLang 服务已就绪，访问地址: $base_url"
            return 0
        fi

        sleep $wait_interval
        elapsed_time=$((elapsed_time + wait_interval))
    done

    echo "错误: SGLang 服务启动超时 (${max_wait_time}s)，请检查 sglang_server_bfcl_v4_${MODEL_NAME}.log"
    exit 1
}

start_model_service
trap 'kill ${SGLANG_PID:-} 2>/dev/null || true; wait ${SGLANG_PID:-} 2>/dev/null || true' EXIT

evalscope eval \
  --model ${MODEL_NAME} \
  --eval-type openai_api \
  --api-url http://127.0.0.1:${PORT}/v1 \
  --api-key EMPTY \
  --datasets bfcl_v4 \
  --dataset-args '{"bfcl_v4": {"subset_list": ["memory_kv", "memory_vector", "memory_rec_sum"], "extra_params": {"underscore_to_dot": true, "is_fc_model": true}}}' \
  --generation-config '{"max_tokens":51200,"temperature":0.7,"top_p":0.95,"parallel_tool_calls":true}' \
  --repeats 1 \
  --eval-batch-size 16 \
  --use-cache ${OUTPUT_PATH} \
  --work-dir ${OUTPUT_PATH}

echo "评测完成，正在关闭 SGLang 服务 (PID=${SGLANG_PID})..."
kill ${SGLANG_PID}
wait ${SGLANG_PID} 2>/dev/null
echo "SGLang 服务已关闭"