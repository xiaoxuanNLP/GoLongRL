#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/LBv1QA_eval/data1}"
PORT="${PORT:-8002}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"

# ========== Judge API 配置 ==========
JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL environment variable must be set}"
JUDGE_MODEL="${JUDGE_MODEL:?JUDGE_MODEL environment variable must be set}"
echo "========== LongBench v1 QA =========="
echo "模型: ${MODEL_NAME}"
mkdir -p "${WORK_DIR}/data" "${WORK_DIR}/runs" "${WORK_DIR}/evals"

CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP} \
    --max-model-len 182272 \
    --port ${PORT} \
    > "${WORK_DIR}/vllm_lbv1qa_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID: ${VLLM_PID}"

echo "等待服务就绪..."
for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${MODEL_NAME}"'","prompt":"Hi","max_tokens":5,"temperature":0}' | grep -q 200; then
        echo "vLLM 就绪"
        break
    fi
    [ $i -eq 120 ] && echo "启动超时，检查 ${WORK_DIR}/vllm_lbv1qa.log" && exit 1
    sleep 5
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "${SCRIPT_DIR}/lbv1qa_eval.py" \
    --model "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:${PORT}/v1" \
    --concurrency 8 \
    --data_dir "${WORK_DIR}/data" \
    --runs_dir "${WORK_DIR}/runs" \
    --evals_dir "${WORK_DIR}/evals" \
    --judge_api_key "${JUDGE_API_KEY}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_model "${JUDGE_MODEL}" \
    > "${WORK_DIR}/lbv1qa_eval_${MODEL_NAME}.log"

echo "========== 完成 =========="
echo "推理: ${WORK_DIR}/runs/"
echo "评测: ${WORK_DIR}/evals/"
# 关闭 vLLM 服务
kill ${VLLM_PID} && echo "vLLM 服务已关闭"