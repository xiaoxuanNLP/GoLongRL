"""
score_jsonl.py — Standalone LongBench v1 QA scorer.

Accepts two input formats:
  - Raw inference files (runs/):  contain "response" field → extract_answer is applied.
  - Pre-evaluated files (evals/): contain "prediction" field → used directly.

Usage:
    python score_jsonl.py --input path/to/file.jsonl
    python score_jsonl.py --input path/to/file.jsonl --verbose
"""

import json
import re
import string
import argparse
from collections import Counter


# ========================= Normalisation =========================

def normalize_zh(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("P", "Z", "C"))
    return s.lower()

def normalize_en(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def is_chinese(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

def tokenize(text):
    return list(normalize_zh(text)) if is_chinese(text) else normalize_en(text).split()


# ========================= Scoring =========================

def f1_score(prediction, ground_truth):
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_f1(prediction, answers):
    """Max F1 over all reference answers (official LongBench protocol)."""
    return max(f1_score(prediction, ans) for ans in answers) if answers else 0.0


# ========================= Answer extraction =========================

def extract_answer(text):
    def clean(s):
        s = s.strip().strip('"').strip("'").rstrip(".")
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        return s

    # Layer 1: last line matching the expected format
    pattern = re.compile(r"[Tt]herefore,?\s*the answer is:?\s*(.+)", re.IGNORECASE)
    match = None
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            match = m
    if match:
        return clean(match.group(1))

    # Layer 2: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return clean(lines[-1])

    # Layer 3: final fallback
    return text.strip()


def get_prediction(item):
    """
    Extract the prediction string from a record, handling both file formats:
      - runs/  files: field is "response" (raw model output) → apply extract_answer
      - evals/ files: field is "prediction" (already extracted) → use directly
    """
    if "response" in item:
        return extract_answer(item["response"])
    if "prediction" in item:
        return item["prediction"]
    return ""


# ========================= Main =========================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Score a LongBench inference JSONL file.")
    parser.add_argument("--input",   required=True, help="Path to the JSONL file (runs or evals format).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample prediction, answers, and F1.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        print("No records found in the input file.")
        return

    scores = []
    for item in records:
        pred    = get_prediction(item)
        answers = item.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        score = compute_f1(pred, answers)
        scores.append(score)

        if args.verbose:
            print(f"[{item.get('id', '?')}]")
            print(f"  Prediction : {pred}")
            print(f"  Answers    : {answers}")
            print(f"  F1         : {score:.4f}")
            print()

    avg = sum(scores) / len(scores) * 100
    print("=" * 45)
    print(f"  File    : {args.input}")
    print(f"  Samples : {len(scores)}")
    print(f"  Avg F1  : {avg:.2f}")
    print("=" * 45)


if __name__ == "__main__":
    main()