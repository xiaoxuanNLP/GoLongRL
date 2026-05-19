#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export VLLM_USE_FLASHINFER_SAMPLER=0

unset LD_LIBRARY_PATH

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/results/outputs_gpqa_diamond/${MODEL_NAME}}"

TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-5002}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "========== GPQA Eval: ${MODEL_NAME} =========="

CUDA_VISIBLE_DEVICES=${GPUS} \
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    > "${SCRIPT_DIR}/vllm_server_gpqa_diamond.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID=${VLLM_PID}"
echo "等待服务启动..."

elapsed_time=0
while [ $elapsed_time -lt 600 ]; do
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${MODEL_NAME}"'","prompt":"hi","max_tokens":5,"temperature":0}') || true
    if [ "$response" -eq 200 ]; then
        echo "vLLM 服务已就绪"
        break
    fi
    sleep 5
    elapsed_time=$((elapsed_time + 5))
done

if [ $elapsed_time -ge 600 ]; then
    echo "错误: vLLM 启动超时，检查 vllm_server_gpqa_diamond.log"
    kill ${VLLM_PID} 2>/dev/null
    exit 1
fi

evalscope eval \
  --model "${MODEL_NAME}" \
  --eval-type openai_api \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --datasets gpqa_diamond \
  --dataset-hub huggingface \
  --dataset-args '{"gpqa_diamond": {"dataset_id": "Idavidrein/gpqa", "subset_list": ["gpqa_diamond"], "eval_split": "train", "few_shot_num": 0, "filters": {"remove_until": "</think>"}}}' \
  --generation-config '{"max_tokens":51200,"temperature":0.7,"top_p":0.95,"stream":true}' \
  --repeats 4 \
  --eval-batch-size 16 \
  --use-cache "${OUTPUT_PATH}" \
  --work-dir "${OUTPUT_PATH}"

echo "评测完成，关闭 vLLM 服务 (PID=${VLLM_PID})..."
kill ${VLLM_PID}
wait ${VLLM_PID} 2>/dev/null || true
echo "========== ${MODEL_NAME} 完成 =========="
