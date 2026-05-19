import json
from collections import Counter


def strip_think(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()

def strip_answer(text):
    return text.partition("<answer>")[2].partition("</answer>")[0].strip() if "<answer>" in text else text.strip()


def compute_score(solution_str, answer) -> float:
    if isinstance(answer, dict) and "golden_label" in answer:
        try:
            answer = json.loads(answer["golden_label"])
        except (json.JSONDecodeError, TypeError):
            return 0.0

    try:
        pred = json.loads(strip_answer(solution_str)).get("data", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pred = []

    def to_key(r):
        return tuple(str(v) for v in r) if isinstance(r, list) else str(r)

    if not isinstance(pred, list):
        return 0.0

    ans_list = answer.get("data", []) if isinstance(answer, dict) else []
    pred_counter = Counter(to_key(r) for r in pred)
    gold_counter = Counter(to_key(r) for r in ans_list)
    matches = sum((pred_counter & gold_counter).values())
    total = len(pred) + len(ans_list) - matches
    return matches / total if total > 0 else 0.0