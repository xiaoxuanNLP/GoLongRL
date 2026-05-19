#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR environment variable must be set}"
DATA_PATH="${DATA_PATH:?DATA_PATH environment variable must be set}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"

echo "========== AIME25 Eval: ${MODEL_NAME} =========="

CUDA_VISIBLE_DEVICES=${GPUS} python "${SCRIPT_DIR}/inference_aime.py" \
    --model "${MODEL_PATH}" \
    --n 32 \
    --output-data "${OUTPUT_DIR}" \
    --max_tokens 51200 \
    --dataset_path "${DATA_PATH}" \
    --temperature 0.7 \
    --top-p 0.95 \
    --tensor-parallel-size ${TP}

echo "========== ${MODEL_NAME} 完成 =========="
