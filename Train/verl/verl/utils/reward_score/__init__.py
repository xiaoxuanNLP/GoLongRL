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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def strict_string_matching(str1, str2):
    return str1 == str2


def uncase_string_matching(str1, str2):
    return str1.lower() == str2.lower()


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    enable_repetition_penalty=False,
    **kwargs,
):
    """Compute the score for a given solution based on the data source."""
    PENALTY = 0.0

    if "</think>" not in solution_str:
        return PENALTY
    if solution_str.count("<think>") > 1 or solution_str.count("</think>") > 1:
        return PENALTY

    if enable_repetition_penalty:
        from . import repetition_score
        if repetition_score.repetition_loop_score(solution_str):
            return PENALTY

    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)
    elif data_source.startswith("MATH##") or data_source.startswith("aime"):
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"]:
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo_math_verify":
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, is_use_math_verify=True)
    elif data_source == "math_longcot":
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth, is_longcot=True, is_use_math_verify=False)
    elif "math_longcot_math_verify" in data_source:
        from . import math_dapo
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        golden_label = ground_truth if isinstance(ground_truth, str) else ground_truth["golden_label"]
        res = math_dapo.compute_score(solution_str, golden_label, is_longcot=True, is_use_math_verify=True)
    elif data_source == "math_stem":
        from . import math_stem
        res = math_stem.compute_score(solution_str, ground_truth)
    elif data_source == "math_stem_longcot":
        from . import math_stem
        res = math_stem.compute_score(solution_str, ground_truth, is_longcot=True)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            from . import prime_code
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source == "longbench_pro":
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        from . import long_bench_pro
        res = long_bench_pro.compute_score(
            solution_str, ground_truth["doc_ids"], extra_info["reward_mode"], ground_truth["summary"]
        )
    elif data_source == "multitableqa_pretraining":
        from . import multitableqa
        res = multitableqa.compute_score(solution_str, ground_truth)
    elif data_source == "retrieval_calculate":
        resp_rm_think = solution_str.split("</think>")[-1]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        from . import retrieval_calculate
        res = retrieval_calculate.compute_score(resp_rm_think, ground_truth)
    elif data_source == "selection":
        resp_rm_think = solution_str.split("</think>")[-1]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        from . import multi_choices
        res = multi_choices.compute_score(resp_rm_think, ground_truth)
    elif data_source == "EM":
        resp_rm_think = solution_str.split("</think>")[-1]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[-1]
        from . import EM_judge
        res = EM_judge.compute_score(resp_rm_think, ground_truth)
    elif (
        "model_verify" in data_source
        or "retrieval_summary_calculate" in data_source
        or "model_judge" in data_source
        or "retrieval_model_judge" in data_source
    ):
        solution_str = solution_str.split("</think>")[-1]
        if extra_info["reward_mode"] == "math_verify":
            if strict_string_matching(solution_str, extra_info["ground_truth"]):
                return 1.0
            from . import math_dapo
            return float(math_dapo.compute_score(solution_str, ground_truth, is_longcot=True, is_use_math_verify=True))
        elif extra_info["reward_mode"] in ("model_judge", "model_judge_with_question", "model_compare"):
            if strict_string_matching(solution_str, extra_info["ground_truth"]):
                return 1.0
        raise NotImplementedError(
            f"genrm_remote scoring is not implemented. data_source={data_source!r}, reward_mode={extra_info.get('reward_mode')!r}"
        )
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


dapo_wi_rllm_compute_score = default_compute_score
