"""CPU regressions for the four GoLongRL component-ablation settings."""

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_tmn_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage


@pytest.fixture(autouse=True)
def register_tmn_for_each_test(monkeypatch):
    # Other core-algorithm tests clear this process-wide registry.
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "tmn_grpo", compute_tmn_grpo_outcome_advantage)


def make_batch(metadata="top_level"):
    # Three prompts; the first two share a reward type but have different means.
    scores = torch.tensor([0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0.2, 0.4, 0.8], dtype=torch.float64)
    rewards = torch.zeros(12, 3, dtype=scores.dtype)
    rewards[:, 0] = scores
    mask = torch.ones_like(rewards)
    mask[::2, 2] = 0
    labels = np.repeat(["F1_Score", "F1_Score", "NDCG"], 4)
    non_tensors = {"uid": np.repeat(["a", "b", "c"], 4)}
    if metadata == "top_level":
        non_tensors["reward_mode"] = labels
    elif metadata == "nested":
        # Distinct subtask names must map to the same F1 reward type.
        tasks = [
            "T5.1 Full-Sentence Citation Alignment",
            "T5.2 Key-Statement Citation Alignment",
            "T1.1 Global Cohesive Retrieval",
        ]
        non_tensors["extra_info"] = np.array([{"reward_mode": task} for task in np.repeat(tasks, 4)], dtype=object)
    return DataProto.from_dict(tensors={"token_level_rewards": rewards, "response_mask": mask}, non_tensors=non_tensors)


def expected_advantages(estimator, mask, pass_rates=None, gamma=1.0):
    # Hand-calculated prompt means: .2, .75, .35; sample variances: .14/3, .05/3, .35/3.
    centered = torch.tensor([-0.2, -0.1, 0, 0.3, -0.15, -0.05, 0.05, 0.15, -0.35, -0.15, 0.05, 0.45], dtype=mask.dtype)
    variances = [0.14 / 3, 0.05 / 3, 0.35 / 3]
    if estimator == "tmn_grpo":
        variances[:2] = [(variances[0] + variances[1]) / 2] * 2
    scales = torch.tensor(variances, dtype=mask.dtype).sqrt().repeat_interleave(4)
    scalars = centered / (scales + 1e-6)
    if pass_rates is not None:
        rates = torch.tensor(pass_rates, dtype=mask.dtype).repeat_interleave(4)
        weights = torch.where(centered > 0, torch.exp(gamma * (0.5 - rates)), torch.exp(gamma * (rates - 0.5)))
        scalars = scalars * weights
    return scalars.unsqueeze(-1) * mask


@pytest.mark.parametrize("estimator", ["grpo", "tmn_grpo"])
@pytest.mark.parametrize("metadata", ["top_level", "nested"])
@pytest.mark.parametrize(
    "alpha,gamma,pass_rates",
    [(0.8, 2.0, [0.25, 0.75, 0.5]), (0.0, 1.0, [0.25, 1.0, 0.5]), (1.0, 1.0, [0.25, 0.5, 0.5])],
)
def test_trainer_forwards_difficulty_settings(estimator, metadata, alpha, gamma, pass_rates):
    """Exercise the real trainer dispatcher; the old GRPO branch ignored all three settings."""
    data = make_batch(metadata)
    config = OmegaConf.create(
        {"difficulty_reweight": True, "difficulty_smooth_alpha": alpha, "difficulty_gamma": gamma}
    )
    result = compute_advantage(data, estimator, config=config)
    expected = expected_advantages(estimator, data.batch["response_mask"], pass_rates, gamma)
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], expected)
    np.testing.assert_array_equal(
        result.non_tensor_batch["reward_mode"], np.repeat(["F1_Score", "F1_Score", "NDCG"], 4)
    )


@pytest.mark.parametrize("estimator", ["grpo", "tmn_grpo"])
def test_disabling_reweight_preserves_normalization(estimator):
    data = make_batch()
    result = compute_advantage(data, estimator, config=OmegaConf.create({"difficulty_reweight": False}))
    torch.testing.assert_close(result.batch["advantages"], expected_advantages(estimator, data.batch["response_mask"]))


@pytest.mark.parametrize("config", [None, {}, {"difficulty_reweight": False}])
def test_plain_grpo_does_not_require_task_metadata(config):
    data = make_batch(metadata=None)
    result = compute_advantage(data, "grpo", config=config)
    torch.testing.assert_close(result.batch["advantages"], expected_advantages("grpo", data.batch["response_mask"]))


def test_reweight_requires_task_metadata():
    with pytest.raises(ValueError, match="reward_mode"):
        compute_advantage(make_batch(metadata=None), "grpo", config={"difficulty_reweight": True})
    data = make_batch()
    with pytest.raises(ValueError, match="reward_mode"):
        compute_grpo_outcome_advantage(
            data.batch["token_level_rewards"],
            data.batch["response_mask"],
            data.non_tensor_batch["uid"],
            difficulty_reweight=True,
        )


@pytest.mark.parametrize("estimator", ["grpo", "tmn_grpo"])
@pytest.mark.parametrize(
    "scores,uid,expected",
    [([0.8], ["a"], [0.8 / (1 + 1e-6) * math.exp(-0.5)]), ([0.2, 0.2, 0.2], ["a", "a", "a"], [0, 0, 0])],
)
def test_singleton_and_zero_variance_groups(estimator, scores, uid, expected):
    rewards = torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)
    kwargs = dict(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards),
        index=np.array(uid),
        reward_mode=np.array(["F1_Score"] * len(uid)),
        difficulty_reweight=True,
        difficulty_smooth_alpha=0.8,
    )
    fn = compute_grpo_outcome_advantage if estimator == "grpo" else compute_tmn_grpo_outcome_advantage
    advantages = fn(**kwargs)[0]
    torch.testing.assert_close(advantages[:, 0], torch.tensor(expected, dtype=rewards.dtype))
