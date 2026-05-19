#!/usr/bin/env python3
"""
FRAMES Scorer — aligned with QwenLong-L1.5 paper (Section 5.1).

Paper method: max(CEM, LLM_judge)
  - CEM = Cover Exact Match: checks if normalized target appears (word-boundary)
    in the full model output (with <think> tags stripped).
  - LLM_judge = acc field from evalscope (DeepSeek-V3 semantic equivalence check).

Usage: python score_frames.py -i review.jsonl [-v]
"""

import argparse, json, re, string, unicodedata, sys


def normalize(text: str) -> str:
    """Normalize text: NFKD, lowercase, remove articles/punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)          # e.g. "1,000" -> "1000"
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split()).strip()


def cem(prediction: str, target: str) -> bool:
    """Cover Exact Match with word-boundary to prevent '2' matching '2019'."""
    np, nt = normalize(prediction), normalize(target)
    if not nt:
        return False
    return bool(re.search(r"\b" + re.escape(nt) + r"\b", np))


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:                               # unclosed tag fallback
        text = text.split("</think>")[-1]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    samples = []
    for line in open(args.input, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        sc = r.get("sample_score", {}).get("score", {})
        target = str(r.get("target", ""))

        # CEM on full output (think-stripped), as used in the paper's eval pipeline
        full_out = strip_think(str(sc.get("prediction", "")))
        cem_hit = cem(full_out, target)

        # LLM judge score (DeepSeek-V3 semantic equivalence)
        llm_acc = float((sc.get("value") or {}).get("acc", 0.0))

        # Paper: max(CEM, LLM_judge)
        correct = 1.0 if (cem_hit or llm_acc >= 0.5) else 0.0

        samples.append({
            "idx": r.get("index"),
            "target": target,
            "cem": cem_hit,
            "llm": llm_acc,
            "score": correct,
        })

    if not samples:
        print("[ERROR] No samples found.")
        sys.exit(1)

    n = len(samples)
    cem_n = sum(s["cem"] for s in samples)
    llm_n = sum(1 for s in samples if s["llm"] >= 0.5)
    final = sum(s["score"] for s in samples)

    print(f"Samples : {n}")
    print(f"Accuracy: {final/n*100:.2f}%  ({int(final)}/{n})  [max(CEM, LLM)]")
    print(f"  CEM: {cem_n/n*100:.2f}%  LLM: {llm_n/n*100:.2f}%")

    if args.verbose:
        miss = [s for s in samples if s["score"] == 0.0]
        if miss:
            print(f"\n[Misses] {len(miss)} samples:")
            for s in miss[:30]:
                print(f"  idx={s['idx']} target='{s['target']}' cem={s['cem']} llm={s['llm']}")


if __name__ == "__main__":
    main()