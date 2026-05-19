#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/LongMemEval_eval}"
PORT="${PORT:-5003}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"

export JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:?JUDGE_MODEL_ID environment variable must be set}"
JUDGE_API_URL="${JUDGE_API_URL:?JUDGE_API_URL environment variable must be set}"

mkdir -p "${WORK_DIR}/runs" "${WORK_DIR}/evals" "${WORK_DIR}/data"
DATA_FILE="${WORK_DIR}/data/longmemeval_s_cleaned.json"

if [ ! -f "${DATA_FILE}" ]; then
    echo "下载数据集..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('xiaowu0162/longmemeval-cleaned', 'longmemeval_s_cleaned.json',
                repo_type='dataset', local_dir='${WORK_DIR}/data')
"
else
    echo "数据集已存在: ${DATA_FILE}"
fi

echo ""
echo "========== LongMemEval: ${MODEL_NAME} =========="

echo "[1/4] 启动 vLLM 服务..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP} \
    --max-model-len 182272 \
    --port ${PORT} \
    > "${WORK_DIR}/vllm_longmemeval_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID=${VLLM_PID}"

echo "等待服务就绪..."
READY=0
for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${MODEL_NAME}"'","prompt":"Hi","max_tokens":5,"temperature":0}' | grep -q 200; then
        echo "vLLM 就绪"
        READY=1
        break
    fi
    sleep 5
done

if [ ${READY} -eq 0 ]; then
    echo "启动超时，检查 ${WORK_DIR}/vllm_longmemeval_${MODEL_NAME}.log"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

echo "[2/4] 运行推理和评测..."
python "${SCRIPT_DIR}/longmemeval_eval.py" \
    --data_file "${DATA_FILE}" \
    --model "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:${PORT}/v1" \
    --concurrency 4 \
    --infer_output "${WORK_DIR}/runs/${MODEL_NAME}_longmemeval_s.jsonl" \
    --eval_output "${WORK_DIR}/evals/${MODEL_NAME}_longmemeval_s_eval.jsonl" \
    --judge_api_key "${JUDGE_API_KEY}" \
    --judge_model "${JUDGE_MODEL_ID}" \
    --judge_base_url "${JUDGE_API_URL}"

echo "评测完成，关闭 vLLM 服务 (PID=${VLLM_PID})..."
kill ${VLLM_PID}
wait ${VLLM_PID} 2>/dev/null || true

echo "========== ${MODEL_NAME} 完成 =========="
