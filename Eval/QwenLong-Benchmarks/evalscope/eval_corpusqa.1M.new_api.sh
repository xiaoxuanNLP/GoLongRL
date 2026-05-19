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
PORT="${PORT:-7000}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"

# ========== YaRN RoPE Scaling 配置 ==========
# Qwen3-4B 原生 max_position_embeddings = 262144 (256K)
# 扩展到 1M 需要 factor = 1048576 / 262144 = 4.0
# 注意: rope_theta 保持模型原始值 5000000
ORIGINAL_MAX_POS=262144
YARN_FACTOR=4.0
ROPE_THETA=5000000
MAX_MODEL_LEN=1048576  # 1M tokens

# ========== Judge API 配置（Volcengine/Ark - DeepSeek V3.2） ==========
JUDGE_API_KEY="${ARK_API_KEY:?ARK_API_KEY environment variable must be set}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL environment variable must be set}"
JUDGE_MODEL="${JUDGE_MODEL:?JUDGE_MODEL environment variable must be set}"

echo "========== CorpusQA (≤1M) =========="
echo "模型: ${MODEL_NAME}"
echo "YaRN factor: ${YARN_FACTOR}, 原始max_pos: ${ORIGINAL_MAX_POS}, 目标长度: ${MAX_MODEL_LEN}"

# ========== 1/4: 下载 1M 数据集 ==========
DATA_FILE="${WORK_DIR}/data/1m_4domains.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
    echo "[1/4] 下载 1M 数据集..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Tongyi-Zhiwen/CorpusQA', '1m_4domains.jsonl',
                repo_type='dataset', local_dir='${WORK_DIR}/data')
"
else
    echo "[1/4] 数据集已存在 ($(wc -l < ${DATA_FILE}) samples)"
fi

# ========== 2/4: 启动 vLLM (带 YaRN) ==========
# 允许超过模型原生 max_position_embeddings 的上下文长度
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size ${TP} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 6 \
    --max-num-batched-tokens 65536 \
    --rope-scaling "{\"rope_type\": \"yarn\", \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": ${ORIGINAL_MAX_POS}}" \
    --port ${PORT} \
    > "${WORK_DIR}/vllm_corpusqa_1m.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID: ${VLLM_PID}"

# 等待时间延长到 300s（1M 模型加载 + KV cache 分配更慢）
echo "等待服务就绪（1M 上下文，预计需要较长时间）..."
for i in $(seq 1 300); do
    if curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${MODEL_NAME}"'","prompt":"Hi","max_tokens":5,"temperature":0}' | grep -q 200; then
        echo "vLLM 就绪 (等待了 $((i*5)) 秒)"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "启动超时，检查 ${WORK_DIR}/vllm_corpusqa_1m.log"
        kill ${VLLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 5
done

# ========== 3/4: 推理 + 评测 ==========
# 注意: concurrency 降低到 2，1M 上下文每个请求显存占用极大
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "${SCRIPT_DIR}/corpusqa_eval.py" \
    --data_file "${DATA_FILE}" \
    --model "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:${PORT}/v1" \
    --concurrency 6 \
    --infer_output "${WORK_DIR}/runs/${MODEL_NAME}_1m.jsonl" \
    --eval_output "${WORK_DIR}/evals/${MODEL_NAME}_1m_eval.jsonl" \
    --judge_api_key "${JUDGE_API_KEY}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_model "${JUDGE_MODEL}"

# ========== 4/4: 关闭 vLLM 服务 ==========
kill ${VLLM_PID} && echo "vLLM 服务已关闭"