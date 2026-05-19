# QwenLong-Benchmarks

QwenLong 系列模型的评测脚本集合，涵盖长文本理解、推理、记忆及通用能力等多个维度。

[English](README.md)

---

## 评测基准总览

### 长文本评测

| 基准 | 脚本 | 说明 |
|------|------|------|
| **LongBench-V2** | 通过 evalscope | 通用长文本理解 |
| **MRCR**（≤128K）| `eval_mrcr.128K.sh` | 长上下文检索 |
| **MRCR**（128K–512K）| `eval_mrcr_4B.128K_512K.sh` | 超长文检索（YaRN 扩展） |
| **MRCR**（512K–1M）| `eval_mrcr.512K_1M.sh` | 超长文检索（YaRN 扩展） |
| **Frames** | `eval_frames.new_api.sh` | 长文档多跳推理 |
| **LongBench** QA 子集 | `eval_lbv1qa.sh` | 长文本问答（2WikiMultihopQA、HotpotQA、MuSiQue、NarrativeQA、Qasper） |
| **DocMath** | `run_docmath_eval.new_api.sh` | 文档数值推理 |
| **CorpusQA**（≤128K）| `eval_corpusqa.128K.new_api.sh` | 多文档语料库推理 |
| **CorpusQA**（≤1M）| `eval_corpusqa.1M.new_api.sh` | 超长文语料库推理（YaRN 扩展） |

### 通用能力评测

| 基准 | 脚本 | 说明 |
|------|------|------|
| **MMLU-Pro** | `eval_mmlu_pro.sh` | 通用知识与推理 |
| **AIME 2024** | `eval_aime24.sh` | 数学竞赛题 |
| **AIME 2025** | `eval_aime25.sh` | 数学竞赛题 |
| **GPQA-Diamond** | `eval_gpqa_diamond.sh` | 研究生级科学问答 |

### 记忆能力评测

| 基准 | 脚本 | 说明 |
|------|------|------|
| **BFCL-V4** 记忆子集 | `eval_bfcl_v4_sglang.sh` | 基于函数调用的 Agent 记忆 |
| **LongMemEval** | `eval_longmemeval.sh` | 对话记忆评测 |

---

## 环境准备

### 基础环境

```bash
# 创建并激活 conda 环境
conda create -n evalscope python=3.10
conda activate evalscope

# 安装 evalscope
cd evalscope
pip install -e .

# 安装 vLLM（大多数脚本需要）
pip install vllm

# 仅 BFCL-V4 需要：安装 SGLang 和 bfcl-eval
pip install sglang
pip install bfcl-eval==2025.10.27.1
```

### 通用环境变量

所有脚本至少需要以下变量：

```bash
export MODEL_PATH=/path/to/your/model   # 模型权重路径
export MODEL_NAME=your-model-name       # 服务模型名
export CONDA_ENV=evalscope              # 可选，默认 "evalscope"
export CONDA_SH=~/miniconda3/etc/profile.d/conda.sh  # 可选
```

使用 LLM-as-Judge 打分的脚本还需要 Judge API 凭证。注意不同脚本使用的变量名不同：

| 脚本 | 所需变量 |
|------|----------|
| CorpusQA、LongBench QA | `ARK_API_KEY`、`JUDGE_BASE_URL`、`JUDGE_MODEL` |
| Frames、DocMath、LongMemEval | `ARK_API_KEY`、`JUDGE_MODEL_ID`、`JUDGE_API_URL` |

详细变量名见各基准的专属说明章节。

---

## 运行评测

所有脚本位于 `evalscope/` 目录下，可从仓库根目录或该目录内执行。

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

使用 `inference_aime.py`，每题采样 32 次（pass@32），结果写入 `OUTPUT_DIR`。

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

### MRCR（≤128K）

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export DATA_FILE=./MRCR_eval/data/mrcr_0_128K.jsonl  # 可选，有默认值

bash evalscope/eval_mrcr.128K.sh
```

在端口 8000 启动 vLLM（TP=4），执行 `mrcr_eval.py`，结果保存在 `MRCR_eval/runs/`。

### MRCR 超长文（128K–512K 与 512K–1M）

这两个脚本使用 YaRN RoPE 扩展上下文长度，数据集不存在时会自动下载。

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-4B

# 128K–512K（YaRN factor=2.0）
bash evalscope/eval_mrcr_4B.128K_512K.sh

# 512K–1M（YaRN factor=4.0）
bash evalscope/eval_mrcr.512K_1M.sh
```

> **注意：** 超长文评测对显存要求极高。1M 脚本设置 `max-num-seqs 4`，初始化时间较长，请耐心等待。

### Frames

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_MODEL_ID=your-judge-model
export JUDGE_API_URL=https://your-judge-api/v1

bash evalscope/eval_frames.new_api.sh
```

### LongBench QA 子集

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_lbv1qa.sh
```

评测 2WikiMultihopQA、HotpotQA、MuSiQue、NarrativeQA、Qasper 五个子集，结果保存在 `LBv1QA_eval/`。

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

### CorpusQA（≤128K）

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_corpusqa.128K.new_api.sh
```

数据集（`128k_4domains.jsonl`）自动从 HuggingFace `Tongyi-Zhiwen/CorpusQA` 下载。

### CorpusQA 超长文（≤1M）

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-4B
export ARK_API_KEY=your_api_key
export JUDGE_BASE_URL=https://your-judge-api/v1
export JUDGE_MODEL=your-judge-model

bash evalscope/eval_corpusqa.1M.new_api.sh
```

使用 YaRN（factor=4.0）扩展到 1M token 上下文，数据集自动下载。

### BFCL-V4（记忆子集）

该脚本使用 SGLang 而非 vLLM，需要独立的 conda 环境。

```bash
conda create -n sglang056 python=3.10
conda activate sglang056
pip install sglang bfcl-eval==2025.10.27.1

export CONDA_ENV=sglang056
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B

bash evalscope/eval_bfcl_v4_sglang.sh
```

评测 `memory_kv`、`memory_vector`、`memory_rec_sum` 三个子集。

### LongMemEval

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=Qwen3-8B
export ARK_API_KEY=your_api_key
export JUDGE_MODEL_ID=your-judge-model
export JUDGE_API_URL=https://your-judge-api/v1

bash evalscope/eval_longmemeval.sh
```

数据集（`longmemeval_s_cleaned.json`）自动从 `xiaowu0162/longmemeval-cleaned` 下载。

---

## 输出目录结构

各基准结果写入独立目录（可通过 `WORK_DIR` 或 `OUTPUT_PATH` 配置）：

```
evalscope/
├── MRCR_eval/runs/           # MRCR 推理输出（.jsonl）
├── CorpusQA_eval/runs/       # CorpusQA 推理输出
├── CorpusQA_eval/evals/      # CorpusQA Judge 打分结果
├── LBv1QA_eval/              # LongBench QA 结果
├── LongMemEval_eval/         # LongMemEval 结果
└── results/
    ├── outputs_frames/       # Frames 结果
    ├── outputs_gpqa_diamond/ # GPQA 结果
    ├── outputs_mmlu_pro/     # MMLU-Pro 结果
    ├── outputs_docmath/      # DocMath 结果
    └── outputs_bfcl_v4/     # BFCL-V4 结果
```

---

## 引用

如果您使用了这些评测脚本，请引用对应基准的原始论文：

- LongBench-V2：Bai et al., 2025
- MRCR（Michelangelo）：Vodrahalli et al., 2024
- Frames：Krishna et al., 2025
- LongBench：Bai et al., 2024
- DocMath：Zhao et al., 2024
- CorpusQA：Lu et al., 2026
- MMLU-Pro：Wang et al., 2024
- GPQA：Rein et al., 2023
- BFCL-V4：Patil et al., 2025
- LongMemEval：Wu et al., 2024
