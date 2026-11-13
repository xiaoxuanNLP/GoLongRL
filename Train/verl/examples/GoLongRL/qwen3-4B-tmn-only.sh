#!/usr/bin/env bash
set -euo pipefail

# Reuse the joint run's hyperparameters, changing only the ablated component.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3-4B-tmn-only}
export ADV_ESTIMATOR=tmn_grpo
export DIFFICULTY_REWEIGHT=False
exec bash "${script_dir}/qwen3-4B-tmn-reweight.sh" "$@"
