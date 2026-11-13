# GoLongRL

**GoLongRL** is a reinforcement learning training recipe built on [verl](https://github.com/volcengine/verl) for improving long-context understanding in large language models. It trains models on the [LongBench Pro](https://github.com/THUDM/LongBench) benchmark suite using task-aware GRPO variants with optional difficulty reweighting.

> Chinese version: [README_zh.md](README_zh.md)

---


## Key Algorithms

### GRPO

Standard Group Relative Policy Optimization. Advantages are normalized within each prompt group.

### TMN-GRPO (Task-Mixed Normalization GRPO)

An extension of GRPO that normalizes advantages within **reward-type groups** rather than globally. This prevents high-variance tasks from dominating the gradient signal when training on a mixture of heterogeneous tasks.

#### Difficulty Reweighting

When `difficulty_reweight=True`, GRPO and TMN use the same difficulty weights. The relative pass rate is the fraction of raw rollout rewards exceeding the smoothed threshold below. Positive advantages receive `exp(gamma * (0.5 - pass_rate))`; negative advantages receive `exp(gamma * (pass_rate - 0.5))`. Thus hard prompts amplify positive advantages and attenuate negative ones, with the reverse for easy prompts.

The pass rate is computed using a smoothed mean:

```
uid_smoothed_mean = α × uid_mean + (1 - α) × mode_mean
```

where `difficulty_smooth_alpha` (α) controls the blend between prompt-level mean (α=1) and task-type-level mean (α=0).

---

## Training Scripts

| Script | Model | Algorithm | Scale |
|--------|-------|-----------|-------|
| `qwen3-4B-grpo.sh` | Qwen3-4B | GRPO | 16 nodes × 8 GPUs |
| `qwen3-4B-tmn-only.sh` | Qwen3-4B | TMN without difficulty reweighting | 16 nodes × 8 GPUs |
| `qwen3-4B-reweight-only.sh` | Qwen3-4B | GRPO + difficulty reweighting | 16 nodes × 8 GPUs |
| `qwen3-4B-tmn-reweight.sh` | Qwen3-4B | TMN-GRPO + difficulty reweight | 16 nodes × 8 GPUs |
| `qwen3-30B-A3B-grpo.sh` | Qwen3-30B-A3B (MoE) | GRPO | 16 nodes × 8 GPUs |

### Quick Start

```bash
# Set paths
export LLM=/path/to/Qwen3-4B
export TRAIN_FILE=/path/to/train.jsonl
export TEST_FILE=/path/to/test.jsonl

# GRPO baseline
bash examples/GoLongRL/qwen3-4B-grpo.sh

# TMN-GRPO with difficulty reweighting
bash examples/GoLongRL/qwen3-4B-tmn-reweight.sh

# Component ablations (same shared training settings as the joint run)
bash examples/GoLongRL/qwen3-4B-tmn-only.sh
bash examples/GoLongRL/qwen3-4B-reweight-only.sh

# Multi-node launch (pass hostfile as first arg)
bash examples/GoLongRL/qwen3-4B-tmn-reweight.sh /etc/mpi/hostfile
```

The two ablation scripts reuse `qwen3-4B-tmn-reweight.sh`, with separate experiment names and default checkpoint directories. TMN-only sets `adv_estimator=tmn_grpo` and `difficulty_reweight=False`. Reweight-only sets `adv_estimator=grpo` and `difficulty_reweight=True`, retaining per-prompt GRPO standard-deviation normalization and the exact same weighting rule as the joint method. Both weighted variants default to alpha 0.8 and gamma 1.0; these can be overridden with `DIFFICULTY_SMOOTH_ALPHA` and `DIFFICULTY_GAMMA`. Use distinct `CKPT_DIR` values if overriding the defaults, since training resumes automatically.

**Implementation fix:** the GRPO trainer now forwards `difficulty_reweight`, `difficulty_smooth_alpha`, and `difficulty_gamma`. Previously, enabling reweighting in a GRPO launch script was silently ignored. The old positive-only, clipped GRPO weighting branch has been replaced by the shared TMN weighting rule. Task labels (`reward_mode`, top-level or in `extra_info`) are required for Reweight-only as well as TMN. Vanilla GRPO still works without task labels. These scripts provide experiment configurations, not completed ablation results.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adv_estimator` | `tmn_grpo` / `grpo` | Advantage estimator |
| `difficulty_reweight` | `True` | Enable difficulty-based sample reweighting |
| `difficulty_smooth_alpha` | `0.8` | Smoothing factor for pass-rate estimation |
| `difficulty_gamma` | `1.0` | Exponential difficulty-weight strength |
| `n_resp_per_prompt` | `16` | Rollout samples per prompt |
| `train_prompt_bsz` | `128` | Training batch size (prompts) |
| `gen_prompt_bsz` | `512` | Generation batch size (prompts) |
| `clip_ratio_low` | `0.2` | PPO clip lower bound |
| `clip_ratio_high` | `0.28` | PPO clip upper bound |
| `rollout_mode` | `async` | SGLang async serving |

---

## Data Format

Each JSONL record should have the following structure:

```json
{
  "data_source": "longbench_pro",
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"ground_truth": "..."},
  "extra_info": {
    "reward_mode": "T5.1 Full-Sentence Citation Alignment",
    "question": "..."
  }
}
```

The `extra_info.reward_mode` field must match one of the 25 LongBench Pro task names listed in `TASK_METRIC_CONFIG` (see `verl/trainer/ppo/metric_utils.py`). It will be automatically mapped to one of the 9 reward types during training.

For non-LongBench-Pro data, set `data_source` to the reward type directly (e.g., `"selection"`, `"EM"`, `"math_longcot_math_verify"`).

---

## Reward Functions

Reward computation is handled in `verl/utils/reward_score/`:

| Module | Reward Type |
|--------|-------------|
| `long_bench_pro.py` | NDCG, Pairwise_Accuracy, Accuracy, F1_Score, SubEM, Summary |
| `math_dapo.py` | math_longcot_math_verify |
| `EM_judge.py` | EM |
| `multi_choices.py` | selection |

---

## Monitoring

Training metrics are logged to wandb. Key metrics include:

- `critic/rewards/mean` — overall mean reward
- `task_metrics/<type>/mean_reward` — per-reward-type mean reward (9 types)
- `task_metrics/<type>/std_reward` — per-reward-type reward std
- `task_metrics/<type>/rate_adv` — fraction of positive advantage samples
- `difficulty/mean_pass_rate` — average prompt pass rate (when `difficulty_reweight=True`)
- `actor/pg_loss`, `actor/entropy` — policy gradient metrics

Use `draw.py` in the repo root to plot training curves from wandb logs.

---

## Requirements

Inherits all verl dependencies. Additional packages for LongBench Pro evaluation:

```bash
pip install jieba rouge pytrec_eval
```

---

## Citation

If you use GoLongRL in your research, please cite:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and
             Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}å
```
