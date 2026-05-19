# QwenLong-Benchmarks

Evaluation scripts for the QwenLong model series, covering long-context understanding, reasoning, memory, and general capabilities.

[中文](README_zh.md)

---

## Benchmarks Overview

### Long-Context Benchmarks

| Benchmark | Script | Description |
|-----------|--------|-------------|
| **LongBench-V2** | via evalscope | General long-context comprehension |
| **MRCR** (≤128K) | `eval_mrcr.128K.sh` | Retrieval under long contexts |
| **MRCR** (128K–512K) | `eval_mrcr_4B.128K_512K.sh` | Ultra-long retrieval with YaRN extension |
| **MRCR** (512K–1M) | `eval_mrcr.512K_1M.sh` | Ultra-long retrieval with YaRN extension |
| **Frames** | `eval_frames.new_api.sh` | Multi-hop reasoning over long documents |
| **LongBench** (QA subsets) | `eval_lbv1qa.sh` | Long-context QA (2WikiMultihopQA, HotpotQA, MuSiQue, NarrativeQA, Qasper) |
| **DocMath** | `run_docmath_eval.new_api.sh` | Numerical reasoning over documents |
| **CorpusQA** (≤128K) | `eval_corpusqa.128K.new_api.sh` | Corpus-level reasoning over multi-document corpora |
| **CorpusQA** (≤1M) | `eval_corpusqa.1M.new_api.sh` | Ultra-long corpus reasoning with YaRN extension |

### General Capability Benchmarks

| Benchmark | Script | Description |
|-----------|--------|-------------|
| **MMLU-Pro** | `eval_mmlu_pro.sh` | General knowledge and reasoning |
| **AIME 2024** | `eval_aime24.sh` | Math competition problems |
| **AIME 2025** | `eval_aime25.sh` | Math competition problems |
| **GPQA-Diamond** | `eval_gpqa_diamond.sh` | Graduate-level science QA |

### Memory Benchmarks

| Benchmark | Script | Description |
|-----------|--------|-------------|
| **BFCL-V4** (memory subset) | `eval_bfcl_v4_sglang.sh` | Agentic memory via function calling |
| **LongMemEval** | `eval_longmemeval.sh` | Dialogue memory evaluation |

---

## Prerequisites

### Environment Setup

```bash
# Create and activate the conda environment
conda create -n evalscope python=3.10
conda activate evalscope

# Install evalscope
cd evalscope
pip install -e .

# Install vLLM (required by most scripts)
pip install vllm

# For BFCL-V4 only: install SGLang and bfcl-eval
pip install sglang
pip install bfcl-eval==2025.10.27.1
```

### Common Environment Variables

All scripts require at minimum:

```bash
export MODEL_PATH=/path/to/your/model
export MODEL_NAME=your-model-name
export CONDA_ENV=evalscope        # optional, defaults to "evalscope"
export CONDA_SH=~/miniconda3/etc/profile.d/conda.sh  # optional
```

Scripts with LLM-as-judge scoring additionally require judge API credentials. Note that different scripts use different variable names:

| Scripts | Variables required |
|---------|-------------------|
| CorpusQA, LongBench QA | `ARK_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL` |
| Frames, DocMath, LongMemEval | `ARK_API_KEY`, `JUDGE_MODEL_ID`, `JUDGE_API_URL` |

See each benchmark's section below for the exact variable names.

---

## Running Evaluations

All scripts are located in the `evalscope/` directory. Run them from the repo root or from within that directory.

### AIME 2024 / 2025

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export OUTPUT_DIR=./results/aime24
export DATA_PATH=/path/to/aime24_data
export GPUS=0,1
export TP=2

bash evalscope/eval_aime24.sh
bash evalscope/eval_aime25.sh
```

Uses `inference_aime.py` with 32 samples per problem (pass@32). Results are written to `OUTPUT_DIR`.

### MMLU-Pro

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export GPUS=0,1,2,3,4,5,6,7
export TP_SIZE=8

bash evalscope/eval_mmlu_pro.sh
```

### GPQA-Diamond

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export GPUS=0,1,2,3,4,5,6,7
export TP_SIZE=8

bash evalscope/eval_gpqa_diamond.sh
```

### MRCR (≤128K)

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export DATA_FILE=./MRCR_eval/data/mrcr_0_128K.jsonl  # optional, has default

bash evalscope/eval_mrcr.128K.sh
```

Starts a vLLM server on port 8000 with TP=4, runs `mrcr_eval.py`, and saves results under `MRCR_eval/runs/`.

### MRCR Ultra-Long (128K–512K and 512K–1M)

These scripts extend context with YaRN RoPE scaling. Data is downloaded automatically if not present.

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-4B

# 128K–512K (YaRN factor=2.0)
bash evalscope/eval_mrcr_4B.128K_512K.sh

# 512K–1M (YaRN factor=4.0)
bash evalscope/eval_mrcr.512K_1M.sh
```

> **Note:** Ultra-long evaluations require significant GPU memory. The 1M script uses `max-num-seqs 4` and may take considerable time to initialize.

### Frames

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_MODEL_ID=your-judge-model
export JUDGE_API_URL=https://your-judge-api/v1

bash evalscope/eval_frames.new_api.sh
```

### LongBench (QA subsets)

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_lbv1qa.sh
```

Evaluates 2WikiMultihopQA, HotpotQA, MuSiQue, NarrativeQA, and Qasper. Results under `LBv1QA_eval/`.

### DocMath

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export DATASET_PATH=/path/to/docmath_dataset
export ARK_API_KEY=your_api_key
export JUDGE_MODEL_ID=your-judge-model
export JUDGE_API_URL=https://your-judge-api/v1

bash evalscope/run_docmath_eval.new_api.sh
```

### CorpusQA (≤128K)

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_corpusqa.128K.new_api.sh
```

Dataset (`128k_4domains.jsonl`) is downloaded automatically from `Tongyi-Zhiwen/CorpusQA` on HuggingFace.

### CorpusQA Ultra-Long (≤1M)

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-4B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_corpusqa.1M.new_api.sh
```

Uses YaRN (factor=4.0) to extend context to 1M tokens. Dataset is downloaded automatically.

### BFCL-V4 (Memory Subset)

Requires SGLang instead of vLLM, and a separate conda environment.

```bash
conda create -n sglang056 python=3.10
conda activate sglang056
pip install sglang bfcl-eval==2025.10.27.1

export CONDA_ENV=sglang056
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B

bash evalscope/eval_bfcl_v4_sglang.sh
```

Evaluates `memory_kv`, `memory_vector`, and `memory_rec_sum` subsets.

### LongMemEval

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_MODEL_ID=your-judge-model
export JUDGE_API_URL=https://your-judge-api/v1

bash evalscope/eval_longmemeval.sh
```

Dataset (`longmemeval_s_cleaned.json`) is downloaded automatically from `xiaowu0162/longmemeval-cleaned`.

---

## Output Structure

Each benchmark writes results under its own directory (configurable via `WORK_DIR` or `OUTPUT_PATH`):

```
evalscope/
├── MRCR_eval/runs/          # MRCR inference outputs (.jsonl)
├── CorpusQA_eval/runs/      # CorpusQA inference outputs
├── CorpusQA_eval/evals/     # CorpusQA judge-scored results
├── LBv1QA_eval/             # LongBench QA results
├── LongMemEval_eval/        # LongMemEval results
└── results/
    ├── outputs_frames/      # Frames results
    ├── outputs_gpqa_diamond/ # GPQA results
    ├── outputs_mmlu_pro/    # MMLU-Pro results
    ├── outputs_docmath/     # DocMath results
    └── outputs_bfcl_v4/    # BFCL-V4 results
```

---

## Citation

If you use these evaluation scripts, please cite the corresponding benchmark papers:

- LongBench-V2: Bai et al., 2025
- MRCR (Michelangelo): Vodrahalli et al., 2024
- Frames: Krishna et al., 2025
- LongBench: Bai et al., 2024
- DocMath: Zhao et al., 2024
- CorpusQA: Lu et al., 2026
- MMLU-Pro: Wang et al., 2024
- GPQA: Rein et al., 2023
- BFCL-V4: Patil et al., 2025
- LongMemEval: Wu et al., 2024
