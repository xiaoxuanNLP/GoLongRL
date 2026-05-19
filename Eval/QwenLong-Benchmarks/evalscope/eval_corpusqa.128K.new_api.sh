#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

cd "${SCRIPT_DIR}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/CorpusQA_eval}"
PORT=7000
GPUS="0,1,2,3,4,5,6,7"
TP=8

# ========== Judge API 配置（Volcengine/Ark - DeepSeek V3.2） ==========
JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL environment variable must be set}"
JUDGE_MODEL="${JUDGE_MODEL:?JUDGE_MODEL environment variable must be set}"

# echo "========== CorpusQA (≤128K) =========="
echo "模型: ${MODEL_NAME}"

# mkdir -p "${WORK_DIR}/runs" "${WORK_DIR}/evals" "${WORK_DIR}/data"
DATA_FILE="${WORK_DIR}/data/128k_4domains.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
    echo "[1/4] 下载数据集..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Tongyi-Zhiwen/CorpusQA', '128k_4domains.jsonl',
                repo_type='dataset', local_dir='${WORK_DIR}/data')
"
else
    echo "[1/4] 数据集已存在 ($(wc -l < ${DATA_FILE}) samples)"
fi
CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP} \
    --max-model-len 182272 \
    --port ${PORT} \
    > "${WORK_DIR}/vllm_corpusqa_128k.log" 2>&1 &

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
    [ $i -eq 120 ] && echo "启动超时，检查 ${WORK_DIR}/vllm.log" && exit 1
    sleep 5
done


python "${SCRIPT_DIR}/corpusqa_eval.py" \
    --data_file "${DATA_FILE}" \
    --model "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:${PORT}/v1" \
    --concurrency 8 \
    --infer_output "${WORK_DIR}/runs/${MODEL_NAME}_128k.jsonl" \
    --eval_output "${WORK_DIR}/evals/${MODEL_NAME}_128k_eval.jsonl" \
    --judge_api_key "${JUDGE_API_KEY}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_model "${JUDGE_MODEL}"

# 关闭 vLLM 服务
kill ${VLLM_PID} && echo "vLLM 服务已关闭"