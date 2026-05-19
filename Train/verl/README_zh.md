# GoLongRL

**GoLongRL** 是基于 [verl](https://github.com/volcengine/verl) 构建的强化学习训练方案，旨在提升大型语言模型的长文本理解能力。训练数据集来自 [LongBench Pro](https://github.com/THUDM/LongBench)，支持任务感知的 GRPO 变体与难度重加权机制。

---

## 核心算法

### GRPO

标准的 Group Relative Policy Optimization，在每个 prompt 组内归一化优势值。

### TMN-GRPO（任务混合归一化 GRPO）

GRPO 的扩展版本，在 **reward 类型组内**（而非全局）归一化优势值，避免多任务混合训练时方差较大的任务主导梯度信号。

#### 难度重加权

开启 `difficulty_reweight=True` 时，样本按每个 prompt 的通过率进行重加权：
- 难题（通过率低）→ 权重高
- 简单题（通过率高）→ 权重低

通过率基于平滑均值计算：

```
uid_smoothed_mean = α × uid_mean + (1 - α) × mode_mean
```

`difficulty_smooth_alpha`（α）控制 prompt 级均值（α=1）与任务类型均值（α=0）的混合比例，推荐值为 0.8。

---

## 训练脚本

| 脚本 | 模型 | 算法 | 规模 |
|------|------|------|------|
| `qwen3-4B-grpo.sh` | Qwen3-4B | GRPO | 16 节点 × 8 GPU |
| `qwen3-4B-tmn-reweight.sh` | Qwen3-4B | TMN-GRPO + 难度重加权 | 16 节点 × 8 GPU |
| `qwen3-30B-A3B-grpo.sh` | Qwen3-30B-A3B (MoE) | GRPO | 16 节点 × 8 GPU |

### 快速启动

```bash
# 设置路径
export LLM=/path/to/Qwen3-4B
export TRAIN_FILE=/path/to/train.jsonl
export TEST_FILE=/path/to/test.jsonl

# GRPO 基线
bash examples/GoLongRL/qwen3-4B-grpo.sh

# TMN-GRPO + 难度重加权
bash examples/GoLongRL/qwen3-4B-tmn-reweight.sh

# 多节点启动（第一个参数传 hostfile）
bash examples/GoLongRL/qwen3-4B-tmn-reweight.sh /etc/mpi/hostfile
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adv_estimator` | `tmn_grpo` / `grpo` | 优势值估计算法 |
| `difficulty_reweight` | `True` | 是否开启难度重加权 |
| `difficulty_smooth_alpha` | `0.8` | 通过率估计的平滑系数 |
| `n_resp_per_prompt` | `16` | 每个 prompt 的 rollout 采样数 |
| `train_prompt_bsz` | `128` | 训练批大小（prompt 数） |
| `gen_prompt_bsz` | `512` | 生成批大小（prompt 数） |
| `clip_ratio_low` | `0.2` | PPO clip 下界 |
| `clip_ratio_high` | `0.28` | PPO clip 上界 |
| `rollout_mode` | `async` | SGLang 异步推理模式 |

---

## 数据格式

每条 JSONL 记录的结构如下：

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

`extra_info.reward_mode` 需匹配 25 个 LongBench Pro 任务名之一（见 `verl/trainer/ppo/metric_utils.py` 中的 `TASK_METRIC_CONFIG`），训练时会自动映射到 9 种 reward 类型之一。

非 LongBench Pro 数据可将 `data_source` 直接设为 reward 类型名（如 `"selection"`、`"EM"`、`"math_longcot_math_verify"`）。

---

## 奖励函数

奖励计算逻辑位于 `verl/utils/reward_score/`：

| 模块 | 负责的 Reward 类型 |
|------|-------------------|
| `long_bench_pro.py` | NDCG, Pairwise_Accuracy, Accuracy, F1_Score, SubEM, Summary |
| `math_dapo.py` | math_longcot_math_verify |
| `EM_judge.py` | EM |
| `multi_choices.py` | selection |

---

## 训练监控

训练指标记录到 wandb，主要包括：

- `critic/rewards/mean` — 整体平均 reward
- `task_metrics/<type>/mean_reward` — 各 reward 类型的平均 reward（9 种）
- `task_metrics/<type>/std_reward` — 各 reward 类型的 reward 标准差
- `task_metrics/<type>/rate_adv` — 正优势样本占比
- `difficulty/mean_pass_rate` — 平均 prompt 通过率（开启 `difficulty_reweight` 时）
- `actor/pg_loss`、`actor/entropy` — 策略梯度指标

可使用仓库根目录的 `draw.py` 脚本从 wandb 日志绘制训练曲线。

---

## 环境依赖

继承 verl 全部依赖。LongBench Pro 评估需额外安装：

```bash
pip install jieba rouge pytrec_eval
```

---

## 引用

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and
             Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
