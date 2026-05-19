# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import re
from collections import Counter, defaultdict
from functools import partial

# LongBench Pro 细粒度任务名 → 9种 reward 类型的映射
TASK_METRIC_CONFIG = {
    "T1.1 Global Cohesive Retrieval": "NDCG",
    "T1.2 Key-Snippet Retrieval": "NDCG",
    "T2.1 Global Timeline Reconstruction": "Pairwise_Accuracy",
    "T2.2 Local Causal Chain Sorting": "Pairwise_Accuracy",
    "T3.1 Multi-Doc Integration QA": "Accuracy",
    "T3.2 Single-Hop Fact QA": "Accuracy",
    "T4.1 Global-Coverage Constrained Summary": "Summary",
    "T4.2 Query-Focused Summary": "Summary",
    "T5.1 Full-Sentence Citation Alignment": "F1_Score",
    "T5.2 Key-Statement Citation Alignment": "F1_Score",
    "T6.1 Large-Scale Document Clustering": "SubEM",
    "T6.2 Targeted Subset Cluster Identification": "F1_Score",
    "T6.3 Global Frequency Analysis": "Pairwise_Accuracy",
    "T7.1 Global Conflict & Inconsistency Localization": "F1_Score",
    "T7.2 Targeted Rule or Condition Violation Detection": "F1_Score",
    "T7.3 Comprehensive Error & Anomaly Sweep": "F1_Score",
    "T8.1 Structured Multi-Source Consistency Verification": "SubEM",
    "T8.2 Single-Source Targeted Aggregation": "SubEM",
    "T8.3 Long-Context Procedural State Tracking": "SubEM",
    "T9.1 Dependency-Aware Multi-Version Impact Analysis": "F1_Score",
    "T9.2 Localized Interface Change Detection": "F1_Score",
    "T10.1 Large-Scale In-Context Rule Induction": "SubEM",
    "T10.2 Targeted Example-Based Rule Induction": "SubEM",
    "T11.1 Long-Range Entity & Commitment Tracking": "Accuracy",
    "T11.2 Short-Range Reference Resolution & State Query": "Accuracy",
}
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()
    score_std = torch.std(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()
    reward_std = torch.std(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        "critic/score/std": score_std,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        "critic/rewards/std": reward_std,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        "critic/advantages/std": torch.std(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        "critic/returns/std": torch.std(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                "critic/values/std": torch.std(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    if "uid" in batch.non_tensor_batch:
        uids = batch.non_tensor_batch["uid"]
        uid_to_lengths = defaultdict(list)
        for i, uid in enumerate(uids):
            if not aborted_mask[i].item():
                uid_to_lengths[uid].append(response_length[i].item())
        rollout_length_diffs = []
        for uid, lengths in uid_to_lengths.items():
            if len(lengths) > 1:
                rollout_length_diffs.append(max(lengths) - min(lengths))
        if rollout_length_diffs:
            metrics["rollout/length_diff/mean"] = np.mean(rollout_length_diffs)
            metrics["rollout/length_diff/max"] = np.max(rollout_length_diffs)
            metrics["rollout/length_diff/min"] = np.min(rollout_length_diffs)
            metrics["rollout/length_diff/std"] = np.std(rollout_length_diffs)

    adv_extra = batch.meta_info.get("adv_extra_metrics", {})
    metrics["difficulty/weight_clamp_upper_ratio"] = adv_extra.get("difficulty/weight_clamp_upper_ratio", 0.0)
    metrics["difficulty/weight_clamp_lower_ratio"] = adv_extra.get("difficulty/weight_clamp_lower_ratio", 0.0)
    metrics["difficulty/mean_pass_rate"] = adv_extra.get("difficulty/mean_pass_rate", 0.0)

    if "reward_mode" in batch.non_tensor_batch:
        task_metrics = compute_reward_mode_metrics(batch)
        metrics.update(task_metrics)

    task_metrics = compute_task_grouped_metrics(batch)
    metrics.update(task_metrics)

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val


def compute_task_grouped_metrics(batch: DataProto) -> dict[str, Any]:
    """
    计算按任务类型（task type）分组的统计指标

    对于每个任务类型k，计算以下5个指标：
    i.   mean_reward: 任务内所有rollout的平均奖励
    ii.  std_reward: 任务内所有rollout奖励的标准差
    iii. mean_std_reward: 任务内query-level标准差的平均值
    iv.  rate_adv: 任务的优势（代理梯度模长）在整个batch里的占比
    v.   rate_num: 任务的query占比

    注意：
    1. 这里的"优势"是指每个rollout的reward减去该prompt下所有rollout的reward平均值，
       不是框架中的advantage值。
    2. 对于T开头的longbench pro任务，按照评价指标（metric）分组，如F1_Score、Accuracy等。
       对于其他任务（selection、math_longcot_math_verify、EM），直接使用任务名称。

    Args:
        batch: DataProto对象，需要包含以下字段：
            - batch["token_level_rewards"]: token级别的奖励
            - non_tensor_batch["extra_info"]: 包含reward_mode（任务类型）
            - non_tensor_batch["uid"]: 每个样本的唯一标识符

    Returns:
        按任务类型分组的统计指标字典
    """
    # ===== 检查必需字段 =====
    if "extra_info" not in batch.non_tensor_batch:
        print(f"\n⚠️  WARNING in compute_task_grouped_metrics:")
        print(f"  - 'extra_info' field not found in batch.non_tensor_batch")
        print(f"  - Available keys: {list(batch.non_tensor_batch.keys())}")
        print(f"  - Cannot compute task-grouped metrics without 'extra_info'")
        print(f"  - Returning empty metrics dict\n")
        return {}

    if "uid" not in batch.non_tensor_batch:
        print(f"\n⚠️  WARNING in compute_task_grouped_metrics:")
        print(f"  - 'uid' field not found in batch.non_tensor_batch")
        print(f"  - Available keys: {list(batch.non_tensor_batch.keys())}")
        print(f"  - Cannot compute task-grouped metrics without 'uid'")
        print(f"  - Returning empty metrics dict\n")
        return {}

    if "token_level_rewards" not in batch.batch:
        print(f"\n⚠️  WARNING in compute_task_grouped_metrics:")
        print(f"  - 'token_level_rewards' field not found in batch.batch")
        print(f"  - Available keys: {list(batch.batch.keys())}")
        print(f"  - Cannot compute task-grouped metrics without 'token_level_rewards'")
        print(f"  - Returning empty metrics dict\n")
        return {}
    # ===== 检查结束 =====

    extra_infos = batch.non_tensor_batch["extra_info"]
    uids = batch.non_tensor_batch["uid"]

    # 提取任务类型
    # 对于T开头的longbench pro任务，使用其对应的metric作为任务类型
    # 对于其他任务（selection、math_longcot_math_verify、EM），直接使用任务名称
    task_types = []
    for info in extra_infos:
        if isinstance(info, dict) and "reward_mode" in info:
            reward_mode = info["reward_mode"]
            # 如果是longbench pro任务，查找对应的metric
            if reward_mode in TASK_METRIC_CONFIG:
                task_types.append(TASK_METRIC_CONFIG[reward_mode])
            else:
                # 非longbench pro任务，直接使用reward_mode
                task_types.append(reward_mode)
        else:
            task_types.append("unknown")

    # 计算每个样本的sequence-level reward
    sequence_rewards = batch.batch["token_level_rewards"].sum(dim=-1).cpu()  # (batch_size,)

    # 按uid分组，构建 uid -> indices 的映射
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    # 计算每个query（uid）内rollouts的奖励平均值和标准差
    query_reward_means = {}  # uid -> mean reward
    query_reward_stds = {}  # uid -> std of rewards (总体标准差)
    query_adv_norms = {}  # uid -> adv_norm (代理梯度模长)

    for uid, indices in uid_to_indices.items():
        query_rewards = sequence_rewards[indices]
        query_reward_means[uid] = torch.mean(query_rewards).item()

        if len(indices) > 1:
            # 使用总体标准差（unbiased=False）来匹配公式：sqrt(1/|G_x| * sum((R(x,y) - mean_R(x))^2))
            query_reward_stds[uid] = torch.std(query_rewards, unbiased=False).item()
        else:
            query_reward_stds[uid] = 0.0

        # 对于query x，其代理梯度模长为：sqrt(1/|G_x| * sum((R(x,y) - mean_R(x))^2))
        # 这就是该query内所有rollout奖励的总体标准差
        query_adv_norms[uid] = query_reward_stds[uid]

    # 按任务类型分组
    task_to_uids = defaultdict(set)  # task_type -> set of uids
    task_to_rewards = defaultdict(list)  # task_type -> list of rewards

    for i, task_type in enumerate(task_types):
        uid = uids[i]
        task_to_uids[task_type].add(uid)
        task_to_rewards[task_type].append(sequence_rewards[i].item())

    # 计算总的query数量和总的adv_norm（用于计算占比）
    total_num_queries = len(uid_to_indices)
    total_adv_norm = sum(query_adv_norms.values())

    # 为每个任务计算5个指标
    metrics = {}
    for task_type, task_uids in task_to_uids.items():
        # 获取该任务的所有rewards
        task_rewards = task_to_rewards[task_type]

        # i. 整体平均reward（任务内所有rollout的平均）
        mean_reward = float(np.mean(task_rewards))

        # ii. 整体标准差 (使用总体标准差，ddof=0)
        std_reward = float(np.std(task_rewards, ddof=0))

        # iii. query-level标准差的平均值
        task_query_stds = [query_reward_stds[uid] for uid in task_uids]
        mean_std_reward = float(np.mean(task_query_stds)) if task_query_stds else 0.0

        # iv. 优势（代理梯度模长）占比
        task_adv_norm = sum(query_adv_norms[uid] for uid in task_uids)
        rate_adv = task_adv_norm / total_adv_norm if total_adv_norm > 0 else 0.0

        # v. query占比
        rate_num = len(task_uids) / total_num_queries if total_num_queries > 0 else 0.0

        # 保存指标（使用safe的task名称作为key）
        safe_task_name = task_type.replace(" ", "_").replace(".", "-")
        prefix = f"task_metrics/{safe_task_name}"

        metrics[f"{prefix}/mean_reward"] = mean_reward
        metrics[f"{prefix}/std_reward"] = std_reward
        metrics[f"{prefix}/mean_std_reward"] = mean_std_reward
        metrics[f"{prefix}/rate_adv"] = rate_adv
        metrics[f"{prefix}/rate_num"] = rate_num
        metrics[f"{prefix}/num_queries"] = len(task_uids)
        metrics[f"{prefix}/num_rollouts"] = len(task_rewards)

    return metrics


def compute_reward_mode_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Compute per-reward-mode statistics for a batch.

    For each reward_mode k with query set T_k where each query x has rollout group G_x:
      1. mean_reward
      2. std_reward
      3. mean_std_reward: avg of within-query std
      4. sqrt_mean_var:   sqrt(mean_x Var(R(x,:)))
      5. rate_adv
      6. rate_num
    """
    reward_modes = batch.non_tensor_batch["reward_mode"]
    uids = batch.non_tensor_batch["uid"]
    seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).cpu()  # (batch_size,)

    # Group indices by uid
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    # Compute per-query std (population std) and var
    query_std = {}
    query_var = {}
    for uid, indices in uid_to_indices.items():
        r = seq_rewards[indices]
        if len(indices) > 1:
            s = torch.std(r, unbiased=False).item()
            query_std[uid] = s
            # Align with TMN-GRPO: sample variance (Bessel correction, / (N-1))
            query_var[uid] = torch.var(r, unbiased=True).item()
        else:
            query_std[uid] = 0.0
            # Align with TMN-GRPO: var=1.0 when only one sample
            query_var[uid] = 1.0

    # Group by reward_mode: collect uids and rewards
    mode_to_uids = defaultdict(set)
    mode_to_rewards = defaultdict(list)
    for i, mode in enumerate(reward_modes):
        mode_to_uids[mode].add(uids[i])
        mode_to_rewards[mode].append(seq_rewards[i].item())

    total_queries = len(uid_to_indices)
    total_std_sum = sum(query_std.values())

    metrics = {}
    for mode, mode_uids in mode_to_uids.items():
        rewards = mode_to_rewards[mode]
        stds = [query_std[uid] for uid in mode_uids]
        vars_ = [query_var[uid] for uid in mode_uids]

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards, ddof=0)
        mean_std_reward = np.mean(stds)

        mean_var = np.mean(vars_) if len(vars_) > 0 else 0.0
        sqrt_mean_var = float(np.sqrt(mean_var)) if mean_var > 0 else 0.0

        rate_adv = sum(stds) / total_std_sum if total_std_sum > 0 else 0.0
        rate_num = len(mode_uids) / total_queries if total_queries > 0 else 0.0

        prefix = f"reward_mode/{mode.replace(' ', '_').replace('.', '-')}"
        metrics[f"{prefix}/mean_reward"] = float(mean_reward)
        metrics[f"{prefix}/std_reward"] = float(std_reward)
        metrics[f"{prefix}/mean_std_reward"] = float(mean_std_reward)
        metrics[f"{prefix}/sqrt_mean_var"] = float(sqrt_mean_var)
        metrics[f"{prefix}/rate_adv"] = float(rate_adv)
        metrics[f"{prefix}/rate_num"] = float(rate_num)

    return metrics


def _compute_single_response_quality(text: str) -> dict[str, float]:
    """计算单个response的质量指标"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # 1. repeated_4gram_ratio
    if len(tokens) >= 4:
        ngrams = [tuple(tokens[i:i+4]) for i in range(len(tokens) - 3)]
        repeated_4gram_ratio = 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
    else:
        repeated_4gram_ratio = 0.0
    
    # 2. repetition_loop_score
    window_size = 100
    if len(text) >= window_size * 2:
        chunks = [text[i:i+window_size] for i in range(0, len(text) - window_size, 50)]
        if len(chunks) >= 2:
            chunk_counter = Counter(chunks)
            repetition_loop_score = sum(c - 1 for c in chunk_counter.values() if c > 1) / len(chunks)
        else:
            repetition_loop_score = 0.0
    else:
        repetition_loop_score = 0.0
    
    # 3. word_count
    word_count = len(tokens)
    
    # 4. unclosed_think_tag
    open_cnt = len(re.findall(r'<think>', text, re.IGNORECASE))
    close_cnt = len(re.findall(r'</think>', text, re.IGNORECASE))
    unclosed_think_tag = 1 if open_cnt > close_cnt else 0
    
    # 5. answer_in_output
    match = re.search(r'</think>(.*)', text, re.DOTALL | re.IGNORECASE)
    answer_in_output = 1 if match and re.search(r'\b[ABCD]\b', match.group(1)) else 0
    
    # 6. type_token_ratio
    type_token_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    
    return {
        'repeated_4gram_ratio': repeated_4gram_ratio,
        'repetition_loop_score': repetition_loop_score,
        'word_count': word_count,
        'unclosed_think_tag': unclosed_think_tag,
        'answer_in_output': answer_in_output,
        'type_token_ratio': type_token_ratio,
    }


def compute_response_quality_metrics(batch: DataProto, tokenizer) -> dict[str, Any]:
    """
    计算response质量指标，用于监控训练过程中的null预测风险
    
    Args:
        batch: DataProto对象
        tokenizer: tokenizer用于decode
        
    Returns:
        Dict包含6个核心指标的统计值
    """
    response_ids = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    max_response_length = response_ids.shape[-1]
    response_mask = attention_mask[:, -max_response_length:]
    
    all_metrics = {
        'repeated_4gram_ratio': [],
        'repetition_loop_score': [],
        'word_count': [],
        'unclosed_think_tag': [],
        'answer_in_output': [],
        'type_token_ratio': [],
    }
    
    batch_size = response_ids.shape[0]
    for i in range(batch_size):
        valid_length = response_mask[i].sum().item()
        valid_response_ids = response_ids[i, :int(valid_length)]
        response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        metrics = _compute_single_response_quality(response_text)
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    result = {}
    for metric_name, values in all_metrics.items():
        values = np.array(values)
        result[f"response_quality/{metric_name}/mean"] = float(np.mean(values))
        result[f"response_quality/{metric_name}/max"] = float(np.max(values))
        result[f"response_quality/{metric_name}/min"] = float(np.min(values))
        
        if metric_name in ['unclosed_think_tag', 'answer_in_output']:
            result[f"response_quality/{metric_name}/ratio"] = float(np.mean(values))
        if metric_name == 'repeated_4gram_ratio':
            result[f"response_quality/{metric_name}/abnormal_ratio"] = float(np.mean(values > 0.4))
        if metric_name == 'repetition_loop_score':
            result[f"response_quality/{metric_name}/abnormal_ratio"] = float(np.mean(values > 0.05))
        if metric_name == 'type_token_ratio':
            result[f"response_quality/{metric_name}/abnormal_ratio"] = float(np.mean(values < 0.15))
    
    return result