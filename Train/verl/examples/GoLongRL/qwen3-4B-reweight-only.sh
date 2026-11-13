#!/usr/bin/env bash
set -euo pipefail

# Keep per-prompt GRPO normalization and the joint run's difficulty weights.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3-4B-reweight-only}
export ADV_ESTIMATOR=grpo
export DIFFICULTY_REWEIGHT=True
exec bash "${script_dir}/qwen3-4B-tmn-reweight.sh" "$@"
