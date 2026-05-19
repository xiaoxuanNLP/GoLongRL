#!/usr/bin/env python3
"""
MRCR (Multi-Round Co-reference Resolution) Score Calculator
============================================================
Reads evalscope openai_mrcr output jsonl, filters by n_needles (2/4/8),
and computes mrcr_score using the EXACT same scoring function as:
  - OpenAI official: https://huggingface.co/datasets/openai/mrcr
  - evalscope: https://github.com/modelscope/evalscope

Scoring function:
    1. Check if response starts with random_string_to_prepend; if not, score = 0
    2. Strip the random_string_to_prepend from both response and answer
    3. Return difflib.SequenceMatcher ratio

Usage:
    python score_mrcr.py --input /path/to/results.jsonl --output /path/to/report.txt
    python score_mrcr.py --input /path/to/results.jsonl --output /path/to/report.txt --remove_until "</think>\n\n"
"""

import argparse
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher


# =============================================================================
# Core scoring function — IDENTICAL to OpenAI official implementation
# https://huggingface.co/datasets/openai/mrcr
# =============================================================================
def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """
    Compare response and answer.
    Exactly matches the OpenAI MRCR official grading logic:
      1) If response does not start with random_string_to_prepend → 0
      2) Strip prefix from both, compute SequenceMatcher ratio
    """
    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


# =============================================================================
# Filter function — mirrors evalscope's "remove_until" filter for thinking models
# =============================================================================
def apply_remove_until_filter(text: str, remove_until: str) -> str:
    """
    Remove everything up to and including `remove_until` marker.
    Used by evalscope to strip <think>...</think> blocks from model output.
    """
    if remove_until and remove_until in text:
        idx = text.index(remove_until) + len(remove_until)
        return text[idx:]
    return text


# =============================================================================
# Data loading — supports evalscope output format
# =============================================================================
def load_samples(input_path: str, remove_until: str = None):
    """
    Load and parse evalscope openai_mrcr output jsonl.

    Expected record structure (evalscope output):
    {
        "index": 506,
        "input": "...",
        "target": "...",
        "sample_score": {
            "score": {
                "value": {"mrcr_score": 0.056},
                "extracted_prediction": "...",
                "prediction": "...(raw with think tags)...",
                "metadata": {"bin_index": 2}
            },
            "sample_id": 506,
            "sample_metadata": {
                "random_string_to_prepend": "96P3X3oCGW",
                "n_needles": 8,
                "bin_index": 2
            }
        }
    }
    """
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON at line {line_num}")
                continue

            # --- Extract fields from evalscope output structure ---
            sample_score = record.get("sample_score", {})
            score_obj = sample_score.get("score", {})
            sample_metadata = sample_score.get("sample_metadata", {})

            # Target (ground truth answer)
            target = record.get("target", "")

            # Random string to prepend (hash for prefix check)
            random_string = sample_metadata.get("random_string_to_prepend", "")

            # Number of needles
            n_needles = sample_metadata.get("n_needles", None)

            # Bin index (token length bucket)
            bin_index = sample_metadata.get("bin_index",
                            score_obj.get("metadata", {}).get("bin_index", None))

            # Prediction: prefer extracted_prediction (already filtered),
            # fall back to raw prediction with optional filter
            extracted_pred = score_obj.get("extracted_prediction", None)
            raw_pred = score_obj.get("prediction", None)

            if extracted_pred is not None:
                prediction = extracted_pred
            elif raw_pred is not None:
                prediction = raw_pred
                if remove_until:
                    prediction = apply_remove_until_filter(prediction, remove_until)
            else:
                print(f"[WARN] No prediction found at line {line_num}, skipping")
                continue

            # Original evalscope-computed score (for verification)
            orig_score = score_obj.get("value", {}).get("mrcr_score", None)

            # Sample index
            sample_id = record.get("index", sample_score.get("sample_id", line_num))

            samples.append({
                "sample_id": sample_id,
                "target": target,
                "prediction": prediction,
                "random_string_to_prepend": random_string,
                "n_needles": n_needles,
                "bin_index": bin_index,
                "orig_mrcr_score": orig_score,
            })

    return samples


# =============================================================================
# Scoring & Reporting
# =============================================================================
def compute_scores(samples):
    """Re-compute mrcr_score for each sample using the official grade function."""
    for s in samples:
        s["recomputed_mrcr_score"] = grade(
            s["prediction"], s["target"], s["random_string_to_prepend"]
        )
    return samples


def generate_report(samples):
    """Generate a detailed report grouped by n_needles and bin_index."""
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("MRCR Score Report")
    lines.append(sep)
    lines.append(f"Total samples: {len(samples)}")
    lines.append("")

    # --- Verification: check if recomputed scores match original ---
    mismatches = 0
    for s in samples:
        if s["orig_mrcr_score"] is not None:
            diff = abs(s["recomputed_mrcr_score"] - s["orig_mrcr_score"])
            if diff > 1e-6:
                mismatches += 1
    if mismatches > 0:
        lines.append(f"[WARNING] {mismatches} samples have score mismatch vs original evalscope output!")
    else:
        lines.append("[OK] All recomputed scores match original evalscope output.")
    lines.append("")

    # --- Overall score ---
    all_scores = [s["recomputed_mrcr_score"] for s in samples]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0
    lines.append(f"Overall MRCR Score: {overall_mean * 100:.2f}  (n={len(all_scores)})")
    lines.append("")

    # --- Group by n_needles ---
    by_needles = defaultdict(list)
    for s in samples:
        key = s["n_needles"] if s["n_needles"] is not None else "unknown"
        by_needles[key].append(s["recomputed_mrcr_score"])

    sorted_keys = sorted([k for k in by_needles.keys() if isinstance(k, int)])
    if "unknown" in by_needles:
        sorted_keys.append("unknown")

    lines.append("-" * 80)
    lines.append(f"{'n_needles':<12} {'Count':<8} {'Mean Score':<14} {'Score (x100)':<14}")
    lines.append("-" * 80)
    for k in sorted_keys:
        scores = by_needles[k]
        mean_s = sum(scores) / len(scores) if scores else 0
        lines.append(f"{str(k):<12} {len(scores):<8} {mean_s:<14.6f} {mean_s * 100:<14.2f}")
    lines.append("-" * 80)
    lines.append("")

    # --- Group by n_needles + bin_index ---
    by_needles_bin = defaultdict(list)
    for s in samples:
        nk = s["n_needles"] if s["n_needles"] is not None else "unknown"
        bk = s["bin_index"] if s["bin_index"] is not None else "unknown"
        by_needles_bin[(nk, bk)].append(s["recomputed_mrcr_score"])

    lines.append("Detailed breakdown by n_needles x bin_index:")
    lines.append("-" * 80)
    lines.append(f"{'n_needles':<12} {'bin_index':<12} {'Count':<8} {'Mean Score':<14} {'Score (x100)':<14}")
    lines.append("-" * 80)

    sorted_nb_keys = sorted(by_needles_bin.keys(), key=lambda x: (
        x[0] if isinstance(x[0], int) else 999,
        x[1] if isinstance(x[1], int) else 999
    ))
    prev_needle = None
    for (nk, bk) in sorted_nb_keys:
        if prev_needle is not None and nk != prev_needle:
            lines.append("")  # separator between needle groups
        prev_needle = nk
        scores = by_needles_bin[(nk, bk)]
        mean_s = sum(scores) / len(scores) if scores else 0
        lines.append(f"{str(nk):<12} {str(bk):<12} {len(scores):<8} {mean_s:<14.6f} {mean_s * 100:<14.2f}")

    lines.append("-" * 80)
    lines.append("")

    # --- Averages that papers typically report ---
    lines.append("=" * 80)
    lines.append("Summary for paper comparison:")
    lines.append("=" * 80)

    # Average across all n_needles (each n_needles weighted equally)
    if sorted_keys and all(isinstance(k, int) for k in sorted_keys):
        per_needle_means = []
        for k in sorted_keys:
            scores = by_needles[k]
            m = sum(scores) / len(scores) if scores else 0
            per_needle_means.append(m)
            lines.append(f"  n_needles={k}:  {m * 100:.2f}  (n={len(scores)})")
        macro_avg = sum(per_needle_means) / len(per_needle_means) if per_needle_means else 0
        lines.append(f"  ---")
        lines.append(f"  Macro-avg (equal weight per n_needles):  {macro_avg * 100:.2f}")
        lines.append(f"  Micro-avg (equal weight per sample):     {overall_mean * 100:.2f}")

        # Also show needle=4 only (what the paper may report)
        if 4 in by_needles:
            n4_scores = by_needles[4]
            n4_mean = sum(n4_scores) / len(n4_scores)
            lines.append(f"  Needle=4 only:                           {n4_mean * 100:.2f}  (n={len(n4_scores)})")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# =============================================================================
# JSON report (machine-readable)
# =============================================================================
def generate_json_report(samples):
    """Generate a machine-readable JSON report."""
    by_needles = defaultdict(list)
    for s in samples:
        key = s["n_needles"] if s["n_needles"] is not None else "unknown"
        by_needles[key].append(s["recomputed_mrcr_score"])

    all_scores = [s["recomputed_mrcr_score"] for s in samples]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0

    report = {
        "total_samples": len(samples),
        "overall_mrcr_score": overall_mean,
        "overall_mrcr_score_pct": round(overall_mean * 100, 2),
        "by_n_needles": {},
    }

    for k in sorted(by_needles.keys(), key=lambda x: x if isinstance(x, int) else 999):
        scores = by_needles[k]
        mean_s = sum(scores) / len(scores) if scores else 0
        report["by_n_needles"][str(k)] = {
            "count": len(scores),
            "mean_score": mean_s,
            "mean_score_pct": round(mean_s * 100, 2),
        }

    # Macro average
    int_keys = [k for k in by_needles.keys() if isinstance(k, int)]
    if int_keys:
        per_needle = [sum(by_needles[k]) / len(by_needles[k]) for k in sorted(int_keys)]
        report["macro_avg_score"] = sum(per_needle) / len(per_needle)
        report["macro_avg_score_pct"] = round(report["macro_avg_score"] * 100, 2)

    return report


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MRCR Score Calculator — reads evalscope output, "
                    "computes scores by n_needles using the official grade function."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to evalscope openai_mrcr output jsonl file."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to save the text report."
    )
    parser.add_argument(
        "--output_json", "-j", default=None,
        help="Optional: path to save machine-readable JSON report."
    )
    parser.add_argument(
        "--remove_until", default=None,
        help='Optional: filter marker to strip thinking tokens, e.g. "</think>\\n\\n". '
             'Only needed if extracted_prediction is not already in the jsonl.'
    )

    args = parser.parse_args()

    # Handle escaped newlines in remove_until
    if args.remove_until:
        args.remove_until = args.remove_until.replace("\\n", "\n")

    print(f"Loading samples from: {args.input}")
    samples = load_samples(args.input, remove_until=args.remove_until)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("[ERROR] No valid samples found. Exiting.")
        return

    # Distribution summary
    needle_counts = defaultdict(int)
    for s in samples:
        needle_counts[s["n_needles"]] = needle_counts.get(s["n_needles"], 0) + 1
    print(f"n_needles distribution: { {k: needle_counts[k] for k in sorted(needle_counts.keys(), key=lambda x: x if isinstance(x, int) else 999)} }")

    print("Computing scores...")
    samples = compute_scores(samples)

    # Generate text report
    report_text = generate_report(samples)
    print(report_text)

    # Save text report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nText report saved to: {args.output}")

    # Save JSON report
    if args.output_json:
        json_report = generate_json_report(samples)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        print(f"JSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()