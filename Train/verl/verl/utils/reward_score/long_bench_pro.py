from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import jieba
from rouge import Rouge
import pytrec_eval
from itertools import combinations

# ==================== Text Normalization ====================

def get_answer_area(text: str) -> str:
    """Extract answer area from model output"""
    if "[Answer]" in text or "[答案]" in text:
        if "[Answer]" in text:
            last_answer_start = text.rfind('[Answer]')
            if last_answer_start != -1:
                text = text[last_answer_start + 8:]
        else:
            last_answer_start = text.rfind('[答案]')
            if last_answer_start != -1:
                text = text[last_answer_start + 4:]
    return text.strip()

def lower(text: str) -> str:
    return text.lower()

def split_by_new_line(text: str) -> List[str]:
    return text.split("\n")

def fix_space(text: str) -> str:
    """Cannot remove all spaces. E.g., "1 11" != "11 1" but "111" == "111" """
    return ' '.join(text.split())

def normalize_answers(answers: List[str]) -> List[str]:
    return [fix_space(lower(a).strip()) for a in answers]

def normalize_prediction(prediction: str) -> List[str]:
    return [fix_space(p.strip()) for p in split_by_new_line(lower(get_answer_area(prediction)))]

def normalize_prediction_abstract(abstract: str) -> str:
    return fix_space(lower(abstract).strip())

# ==================== Metrics ====================

def Accuracy(answers: List[str], prediction: str) -> float:
    """Exact match on first line"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    
    if len(answers) == 0 or len(predictions) == 0:
        return 0.0
    
    return 1.0 if answers[0] == predictions[0] else 0.0

def F1_Score(answers: List[str], prediction: str) -> float:
    """Set-based F1 score"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    
    answer_set = set(answers)
    prediction_set = set(predictions)
    
    common = answer_set & prediction_set
    if len(common) == 0 or len(prediction_set) == 0 or len(answer_set) == 0:
        return 0.0
    
    precision = len(common) / len(prediction_set)
    recall = len(common) / len(answer_set)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def SubEM(answers: List[str], prediction: str) -> float:
    """Subset exact match: each answer must appear in predictions"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    
    if len(answers) == 0 or len(predictions) == 0:
        return 0.0
    
    score = 0.0
    for a in answers:
        if a in predictions:
            score += 1.0
    return score / len(answers)

def Summary_Max_Rouge_L(answers: List[str], prediction: str, is_zh: bool) -> float:
    """Max Rouge-L score across multiple reference answers"""
    if is_zh:
        answers = [" ".join(list(jieba.cut(a, cut_all=False))) for a in answers]
        prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))

    rouge_evaluator = Rouge()
    try:
        scores = rouge_evaluator.get_scores([prediction] * len(answers), answers, avg=False)
    except:
        return 0.0

    return max([score["rouge-l"]["f"] for score in scores])


def Summary(answers: List[str], prediction: str, is_zh: bool) -> float:
    """Summary score: weighted combination of semantic similarity and Rouge-L"""
    answers = normalize_answers(answers)
    prediction = normalize_prediction_abstract(prediction)

    if len(answers) == 0 or not prediction:
        return 0.0

    rouge_score = Summary_Max_Rouge_L(answers, prediction, is_zh)
    
    return rouge_score

def NDCG(answers: List[str], prediction: str) -> float:
    """NDCG@k for ranking tasks"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    
    if len(answers) == 0 or len(predictions) == 0:
        return 0.0

    k_value = len(answers)

    # Convert to pytrec_eval format
    answers_dict = {
        'query': {a: len(answers) - i for i, a in enumerate(answers)}
    }
    predictions_dict = {
        'query': {p: len(predictions) - i for i, p in enumerate(predictions)}
    }

    ndcg_string = "ndcg_cut." + str(k_value)
    evaluator = pytrec_eval.RelevanceEvaluator(answers_dict, {ndcg_string})
    scores = evaluator.evaluate(predictions_dict)

    ndcg = 0.0
    for query_id in scores.keys():
        ndcg += scores[query_id]["ndcg_cut_" + str(k_value)]
    
    return ndcg / len(scores)

def Pairwise_Accuracy(answers: List[str], prediction: str) -> float:
    """Pairwise ordering accuracy for sequence reconstruction"""
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    
    if len(answers) <= 1 or len(predictions) <= 1:
        return 0.0

    n_total = len(predictions) * (len(predictions) - 1) // 2
    prediction_indices = {p: i for i, p in enumerate(predictions)}
    n_correct = 0

    for a, b in combinations(answers, 2):
        if a in prediction_indices and b in prediction_indices:
            if prediction_indices[a] < prediction_indices[b]:
                n_correct += 1

    return n_correct / n_total

# ==================== Task-Metric Mapping ====================

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
    "T11.2 Short-Range Reference Resolution & State Query": "Accuracy"
}

# ==================== Main Compute Function ====================

def compute_score(
    model_output: str, 
    ground_truth: List[str], 
    sub_task: str,
    language: str = "English",
) -> float:
    """
    Compute reward score for LongBench-Pro tasks
    
    Args:
        model_output: Raw model output string
        ground_truth: List of reference answers
        sub_task: Task identifier (e.g., "T1.1 Global Cohesive Retrieval")
        language: "Chinese" or "English"
    
    Returns:
        Score in range [0.0, 1.0]
    """
    # Handle empty cases
    if not model_output or not model_output.strip():
        return 0.0
    
    if not ground_truth or len(ground_truth) == 0:
        return 0.0
    
    # Get metric type
    metric_name = TASK_METRIC_CONFIG.get(sub_task)
    if not metric_name:
        raise ValueError(f"Unknown sub_task: {sub_task}")
    
    # Compute metric
    try:
        is_zh = (language == "Chinese")
        
        if metric_name == "NDCG":
            score = NDCG(ground_truth, model_output)
        elif metric_name == "Pairwise_Accuracy":
            score = Pairwise_Accuracy(ground_truth, model_output)
        elif metric_name == "Accuracy":
            score = Accuracy(ground_truth, model_output)
        elif metric_name == "F1_Score":
            score = F1_Score(ground_truth, model_output)
        elif metric_name == "SubEM":
            score = SubEM(ground_truth, model_output)
        elif metric_name == "Summary":
            score = Summary(ground_truth, model_output, is_zh)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # Validate score range
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"
        return score
        
    except Exception as e:
        # Return 0.0 for any errors during computation
        return 0.0