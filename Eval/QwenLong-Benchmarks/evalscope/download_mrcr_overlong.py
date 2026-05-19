#!/usr/bin/env python3
"""
Download MRCR ultra-long context data from HuggingFace (openai/mrcr).
Filters records by Qwen token count to produce the 128K-512K or 512K-1M splits.

Usage:
    python download_mrcr_overlong.py --output_dir ./datasets/mrcr --range 128k_512k
    python download_mrcr_overlong.py --output_dir ./datasets/mrcr --range 512k_1m
"""

import argparse
import json
import os

RANGE_CONFIG = {
    "128k_512k": {
        "min_tokens": 128 * 1024,
        "max_tokens": 512 * 1024,
        "output_file": "mrcr_128K_512K.jsonl",
        "description": "128K–512K token range",
    },
    "512k_1m": {
        "min_tokens": 512 * 1024,
        "max_tokens": 1024 * 1024,
        "output_file": "mrcr_512K_1M.jsonl",
        "description": "512K–1M token range",
    },
}

# Token count field name in the openai/mrcr dataset.
# If the dataset uses a different field, adjust this constant.
TOKEN_COUNT_FIELD = "n_tokens_qwen"


def main():
    parser = argparse.ArgumentParser(description="Download and filter MRCR ultra-long data.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output JSONL file.")
    parser.add_argument("--range", required=True, choices=list(RANGE_CONFIG.keys()),
                        help="Token range to download: 128k_512k or 512k_1m")
    parser.add_argument("--dataset_id", default="openai/mrcr",
                        help="HuggingFace dataset ID (default: openai/mrcr)")
    parser.add_argument("--split", default="test",
                        help="Dataset split to use (default: test)")
    args = parser.parse_args()

    config = RANGE_CONFIG[args.range]
    output_path = os.path.join(args.output_dir, config["output_file"])
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path) as f:
            count = sum(1 for _ in f)
        print(f"[INFO] Output file already exists: {output_path} ({count} samples)")
        return

    print(f"[INFO] Downloading {args.dataset_id} (split={args.split}) from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("[ERROR] 'datasets' package not found. Install it with: pip install datasets")

    ds = load_dataset(args.dataset_id, split=args.split, trust_remote_code=True)
    print(f"[INFO] Loaded {len(ds)} total records.")

    min_t = config["min_tokens"]
    max_t = config["max_tokens"]
    print(f"[INFO] Filtering for {config['description']} ({min_t} ≤ {TOKEN_COUNT_FIELD} < {max_t})...")

    # Validate that the token count field exists
    sample = ds[0]
    if TOKEN_COUNT_FIELD not in sample:
        available = list(sample.keys())
        raise SystemExit(
            f"[ERROR] Field '{TOKEN_COUNT_FIELD}' not found in dataset.\n"
            f"Available fields: {available}\n"
            f"Adjust TOKEN_COUNT_FIELD in this script to match your dataset."
        )

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            n_tokens = item.get(TOKEN_COUNT_FIELD, 0) or 0
            if min_t <= n_tokens < max_t:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
                count += 1

    print(f"[INFO] Saved {count} samples to {output_path}")
    if count == 0:
        print(f"[WARN] No samples found in range [{min_t}, {max_t}). "
              f"Check the '{TOKEN_COUNT_FIELD}' field values in the dataset.")


if __name__ == "__main__":
    main()
