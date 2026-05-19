#!/usr/bin/env bash
set -x

mkdir -p output

LLM=${LLM:-"/path/to/Qwen3-4B"}

HOME=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")

project_name='GoLongRL'
experiment_name='Qwen3-4B-GRPO'

# ============================================================================
# Offload
# ============================================================================
ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}

CKPT_DIR=${CKPT_DIR:-"${HOME}/ckpts/${project_name}/${experiment_name}/"}
mkdir -p $CKPT_DIR
ROLLOUT_DATA_DIR=${CKPT_DIR}rollout_outputs
mkdir -p $ROLLOUT_DATA_DIR

TRAIN_FILE=${TRAIN_FILE:-"/path/to/train.jsonl"}
TEST_FILE=${TEST_FILE:-"/path/to/test.jsonl"}

# ============================================================================
# Hostfile
# ============================================================================
HOSTFILE="${1:-/etc/mpi/hostfile}"

# ============================================================================
# Cluster and parallelism
# ============================================================================
NODES=16
N_GPUS_PER_NODE=8

TP=8
PP=1
CP=8
EP=1
ETP=1

INFER_TP=8
rollout_mode="async"
rollout_name="sglang"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ============================================================================
# Algorithm
# ============================================================================
adv_estimator=grpo

loss_mode="vanilla"

difficulty_reweight=False

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_filter_groups=True
filter_groups_metric=score
max_num_gen_batches=32

train_prompt_bsz=128
gen_prompt_bsz=512
n_resp_per_prompt=16
train_prompt_mini_bsz=128

# ============================================================================
# Rollout buffer
# ============================================================================
NUM_WORKER=$NODES
ROLLOUT_BUFFER_CAPACITY=32
CUDA_GRAPH_MAX_BS=512

# Rollout IS (Importance Sampling)
rollout_is=true
rollout_is_threshold=5.0
rollout_is_threshold_lower=0.5
rollout_is_level=token
rollout_is_mode=clip
rollout_is_veto_threshold=1e-4

loss_agg_mode="token-mean"

# ============================================================================
# Data
# ============================================================================
filter_overlong_prompts=True
max_prompt_length=$((1024 * 160))
max_response_length=$((1024 * 16))
val_response_length=$((1024 * 16))

use_filtered_cache=True
rebuild_filtered_cache=False

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ============================================================================
# Sampling
# ============================================================================
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.6
val_top_p=0.95

# ============================================================================
# Performance
# ============================================================================
use_dynamic_bsz=False
ppo_micro_batch_size_per_gpu=1

max_total_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * ppo_micro_batch_size_per_gpu / CP))
infer_ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / CP))

recompute=True

# ============================================================================
# Profiling
# ============================================================================
PROFILE_STEPS="[]"
PROFILE_RANKS_ALL=True
DISCRETE=True

# ============================================================================
# Environment
# ============================================================================
export HYDRA_FULL_ERROR=1

# ============================================================================
# Launch
# ============================================================================
PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo --config-name='dapo_megatron_trainer' \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_ECE_ENABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="32" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONWARNINGS="ignore" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG="VERSION" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_GID_INDEX="3" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_ASYNC_ERROR_HANDLING="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="bond0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_HCA="mlx5" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PXN_DISABLE="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_QPS_PER_CONNECTION="4" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="${WANDB_API_KEY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WORK_DIR=$HOME \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="bond0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TIMESTAMP=$timestamp \
    +ray_kwargs.ray_init.runtime_env.env_vars.SGL_ENABLE_JIT_DEEPGEMM="False" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=$return_raw_chat \
    data.trust_remote_code=True \
    data.shuffle=True \
    +data.filtered_cache_dir=${CKPT_DIR} \
    +data.use_filtered_cache=${use_filtered_cache} \
    +data.rebuild_filtered_cache=${rebuild_filtered_cache} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation='left' \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.model.path="${LLM}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.max_model_len=${max_total_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.buffer_capacity=${ROLLOUT_BUFFER_CAPACITY} \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKER} \
    +algorithm.difficulty_reweight=${difficulty_reweight} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_requests=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_requests_level=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=${CUDA_GRAPH_MAX_BS} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=${CUDA_GRAPH_MAX_BS} \
    reward_model.reward_manager=dapo \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    ++reward_model.enable_reward_workers=False \
    ++reward_model.compute_in_agent_loop=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='flash' \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.ref.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.ref.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.ref.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    global_profiler.steps=${PROFILE_STEPS} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=5 \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=60 \
    trainer.val_only=False \
    trainer.log_val_generations=15 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False 2>&1 | tee output/${experiment_name}_${timestamp}.log
