#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH environment variable must be set}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/results/outputs_docmath/${MODEL_NAME}}"

PORT="${PORT:-8802}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"

export JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:?JUDGE_MODEL_ID environment variable must be set}"
JUDGE_API_URL="${JUDGE_API_URL:?JUDGE_API_URL environment variable must be set}"

echo "========== DocMath Eval: ${MODEL_NAME} =========="
mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP} \
    --max-model-len 182272 \
    --port ${PORT} \
    > "${OUTPUT_PATH}/vllm_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID: ${VLLM_PID}"

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
    echo "启动超时，检查 ${OUTPUT_PATH}/vllm_${MODEL_NAME}.log"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

python "${SCRIPT_DIR}/docmath_eval.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --port "${PORT}" \
    --output_path "${OUTPUT_PATH}" \
    --judge_api_key "${JUDGE_API_KEY}" \
    --judge_model_id "${JUDGE_MODEL_ID}" \
    --judge_api_url "${JUDGE_API_URL}" \
    --dataset_path "${DATASET_PATH}" \
    --subsets "complong_testmini,compshort_testmini,simplong_testmini,simpshort_testmini" \
    --max_input_tokens 131000 \
    --max_tokens 51200 \
    --eval_batch_size 32 \
    2>&1 | tee "${OUTPUT_PATH}/eval_${MODEL_NAME}.log"

kill ${VLLM_PID} && echo "vLLM 服务已关闭"
wait ${VLLM_PID} 2>/dev/null || true

echo "========== ${MODEL_NAME} 完成 =========="
echo "结果目录: ${OUTPUT_PATH}"
