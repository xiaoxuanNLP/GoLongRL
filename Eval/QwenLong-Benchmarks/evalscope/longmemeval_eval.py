"""
LongMemEval Evaluation Pipeline (Full-context Long-context Reading)
- Step 1: Inference via local vLLM (OpenAI-compatible API)
- Step 2: LLM-as-judge evaluation via DeepSeek-V3
Aligned with QwenLong-L1.5 paper Section 5.3 (Table 8) and official LongMemEval repo.

Dataset: LongMemEval-S (Wu et al., 2024)
  - 500 questions, ~115K tokens per instance, ~40 history sessions
  - 5 core abilities: Information Extraction, Multi-Session Reasoning,
    Knowledge Updates, Temporal Reasoning, Abstention
  - Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Paper evaluation config (QwenLong-L1.5, Section 5.1):
  - Max input length: 128K tokens
  - Max generation length: 50K tokens
  - Temperature: 0.7, top_p: 0.95
  - Judge: DeepSeek-V3 for semantic equivalence
"""

import json, os, re, argparse, asyncio, threading
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= Judge Prompt (LongMemEval official evaluate_qa.py) =========================
# The official LongMemEval evaluation uses GPT-4o as judge to check answer correctness.
# We replicate the same semantic equivalence check, consistent with the QwenLong-L1.5 paper's
# approach of using LLM-as-a-judge for multi-hop QA benchmarks.

JUDGE_SYSTEM = """You are an expert evaluator. Given a question, a reference answer, and a model's hypothesis, determine whether the hypothesis correctly answers the question based on the reference answer.

Guidelines:
1. The hypothesis does not need to be an exact match. It should capture the key information in the reference answer.
2. If the reference answer contains multiple key points, the hypothesis should address all of them.
3. Minor differences in phrasing, formatting, or additional correct details are acceptable.
4. If the question is unanswerable (abstention), the hypothesis should indicate that the information is not available or unknown.
5. Be lenient with partial matches if the core information is correct.

Your output must follow the following format:
1) Provide a brief explanation for your judgment.
2) Then provide your final verdict in the form of: [[CORRECT]] or [[INCORRECT]]"""

JUDGE_USER = """Question: {question}
Reference Answer: {answer}
Model's Response: {hypothesis}"""

# ========================= Reading Prompt (Chain-of-Note style, aligned with LongMemEval official) =====
# The official LongMemEval recommends the "con" (chain-of-note) reading method:
#   First extract useful information from the history, then reason to produce the answer.
# The QwenLong-L1.5 paper uses temperature=0.7, top_p=0.95 for all evaluations.

READING_SYSTEM = """You are a helpful assistant with access to the user's past conversation history. Based on the conversation history provided below, answer the user's question.

Instructions:
1. First, carefully review the conversation history to identify all relevant information.
2. Extract and list the key pieces of evidence from the history that are relevant to the question.
3. Based on the extracted evidence, reason step by step to produce your final answer.
4. If the question asks about information that was never mentioned in the conversation history, clearly state that the information is not available.
5. Be concise and precise in your final answer."""

HISTORY_TEMPLATE_JSON = """Below is the user's conversation history with a chat assistant, presented in chronological order. Each session is timestamped.

{history_text}

Based on the above conversation history, answer the following question:
{question}"""


# ========================= Helpers =========================
def load_json(path):
    """Load a JSON file (LongMemEval uses .json, not .jsonl)."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):
    """Load a JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(path):
    """Load already-processed question IDs for resumability."""
    ids = set()
    if os.path.exists(path):
        for item in load_jsonl(path):
            if 'question_id' in item:
                ids.add(item['question_id'])
    return ids


def strip_think(text):
    """Strip <think>...</think> blocks from reasoning model outputs."""
    return text.split('</think>')[-1].strip() if '</think>' in text else text.strip()


def format_history_json(item):
    """
    Format the chat history sessions into a JSON-style string.
    Each session is presented with its timestamp and turns.
    This matches the LongMemEval official 'json' history format.
    """
    sessions = item['haystack_sessions']
    dates = item['haystack_dates']

    history_parts = []
    for i, (session, date) in enumerate(zip(sessions, dates)):
        session_text = f"=== Session {i+1} (Date: {date}) ===\n"
        for turn in session:
            role = turn['role'].capitalize()
            content = turn['content']
            session_text += f"{role}: {content}\n"
        history_parts.append(session_text.strip())

    return "\n\n".join(history_parts)


def get_question_category(qtype, qid):
    """
    Map question_type to the 5 core LongMemEval ability categories.
    Matches the official LongMemEval taxonomy:
      - Information Extraction (IE): single-session-user, single-session-assistant, single-session-preference
      - Multi-Session Reasoning (MR): multi-session
      - Knowledge Updates (KU): knowledge-update
      - Temporal Reasoning (TR): temporal-reasoning
      - Abstention (ABS): any question_id ending with '_abs'
    """
    if qid.endswith('_abs'):
        return 'abstention'
    mapping = {
        'single-session-user': 'information_extraction',
        'single-session-assistant': 'information_extraction',
        'single-session-preference': 'information_extraction',
        'multi-session': 'multi_session_reasoning',
        'knowledge-update': 'knowledge_update',
        'temporal-reasoning': 'temporal_reasoning',
    }
    return mapping.get(qtype, 'other')


# ========================= Step 1: Inference =========================
async def run_inference(args):
    """
    Full-context long-context reading inference.
    The entire chat history (~115K tokens for LongMemEval-S) is provided
    as context, and the model is asked to answer the question.

    Paper config: temperature=0.7, top_p=0.95, max_tokens=51200 (50K).
    """
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_url)
    data = load_json(args.data_file)
    done = load_done_ids(args.infer_output)
    todo = [x for x in data if x['question_id'] not in done]

    if not todo:
        print(f"Inference: all {len(data)} samples done, skipping.")
        return

    print(f"Inference: {len(done)} done, {len(todo)} remaining")
    sem = asyncio.Semaphore(args.concurrency)

    async def infer_one(item):
        async with sem:
            history_text = format_history_json(item)
            user_content = HISTORY_TEMPLATE_JSON.format(
                history_text=history_text,
                question=item['question']
            )
            msgs = [
                {"role": "system", "content": READING_SYSTEM},
                {"role": "user", "content": user_content}
            ]
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=msgs,
                        stream=False,
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=51200
                    )
                    raw = resp.choices[0].message.content.strip()
                    return {**item, "response": strip_think(raw), "raw_response": raw}
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(10 * (attempt + 1))
                    else:
                        print(f"  [FAIL] {item['question_id']}: {e}")
                        return None

    tasks = [infer_one(x) for x in todo]
    os.makedirs(os.path.dirname(args.infer_output) or '.', exist_ok=True)
    with open(args.infer_output, 'a', encoding='utf-8') as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
            r = await fut
            if r:
                out = {
                    "question_id": r["question_id"],
                    "question_type": r["question_type"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "hypothesis": r["response"],
                    "model": args.model,
                }
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
                f.flush()


# ========================= Step 2: Evaluation =========================
def run_eval(args):
    """
    LLM-as-a-judge evaluation.
    Uses DeepSeek-V3 (default) to judge semantic equivalence between
    the model's hypothesis and the reference answer, consistent with the
    QwenLong-L1.5 paper's evaluation methodology for QA benchmarks.

    The official LongMemEval repo uses GPT-4o; we support both via --judge_model.
    """
    client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    data = load_jsonl(args.infer_output)
    done = load_done_ids(args.eval_output)
    todo = [x for x in data if x['question_id'] not in done]

    if not todo:
        print(f"Eval: all {len(data)} samples done.")
    else:
        print(f"Eval: {len(done)} done, {len(todo)} remaining (judge: {args.judge_model})")

    # 5 core ability categories + overall
    CATEGORIES = [
        'information_extraction', 'multi_session_reasoning',
        'knowledge_update', 'temporal_reasoning', 'abstention'
    ]
    stats = {'correct': 0, 'total': 0}
    cat_stats = {c: {'correct': 0, 'total': 0} for c in CATEGORIES}
    type_stats = {}  # fine-grained per question_type

    # Restore stats from existing results
    if os.path.exists(args.eval_output):
        for item in load_jsonl(args.eval_output):
            stats['total'] += 1
            if item.get('correct'):
                stats['correct'] += 1
            cat = get_question_category(item.get('question_type', ''), item.get('question_id', ''))
            if cat in cat_stats:
                cat_stats[cat]['total'] += 1
                if item.get('correct'):
                    cat_stats[cat]['correct'] += 1
            qtype = item.get('question_type', 'unknown')
            if qtype not in type_stats:
                type_stats[qtype] = {'correct': 0, 'total': 0}
            type_stats[qtype]['total'] += 1
            if item.get('correct'):
                type_stats[qtype]['correct'] += 1

    def judge_one(item):
        hypothesis = item.get('hypothesis', '')
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                question=item['question'],
                answer=item['answer'],
                hypothesis=hypothesis
            )}
        ]
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=msgs,
                    temperature=0.0
                )
                content = resp.choices[0].message.content
                correct = "[[CORRECT]]" in content and "[[INCORRECT]]" not in content
                return {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type", ""),
                    "question": item["question"],
                    "gold": item["answer"],
                    "pred": hypothesis,
                    "correct": correct,
                    "reason": content.strip(),
                }
            except Exception as e:
                if attempt == 2:
                    return {
                        "question_id": item["question_id"],
                        "question_type": item.get("question_type", ""),
                        "question": item["question"],
                        "gold": item["answer"],
                        "pred": hypothesis,
                        "correct": False,
                        "reason": f"Error: {e}",
                    }
                import time
                time.sleep(5 * (attempt + 1))

    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.eval_output) or '.', exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool, \
         open(args.eval_output, 'a', encoding='utf-8') as f:
        futs = {pool.submit(judge_one, x): x['question_id'] for x in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    f.flush()
                    stats['total'] += 1
                    if r['correct']:
                        stats['correct'] += 1
                    cat = get_question_category(r.get('question_type', ''), r.get('question_id', ''))
                    if cat in cat_stats:
                        cat_stats[cat]['total'] += 1
                        if r['correct']:
                            cat_stats[cat]['correct'] += 1
                    qtype = r.get('question_type', 'unknown')
                    if qtype not in type_stats:
                        type_stats[qtype] = {'correct': 0, 'total': 0}
                    type_stats[qtype]['total'] += 1
                    if r['correct']:
                        type_stats[qtype]['correct'] += 1
                    if stats['total'] % 20 == 0:
                        print(f"  [{stats['total']}] acc={stats['correct']/stats['total']:.2%}")

    # Report
    print("\n" + "=" * 60)
    print("LONGMEMEVAL RESULTS")
    print("=" * 60)
    if stats['total']:
        print(f"Overall: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.2%}")
        print()
        print("--- By Core Ability Category ---")
        CATEGORY_NAMES = {
            'information_extraction': 'Information Extraction (IE)',
            'multi_session_reasoning': 'Multi-Session Reasoning (MR)',
            'knowledge_update': 'Knowledge Update (KU)',
            'temporal_reasoning': 'Temporal Reasoning (TR)',
            'abstention': 'Abstention (ABS)',
        }
        for cat in CATEGORIES:
            s = cat_stats[cat]
            name = CATEGORY_NAMES.get(cat, cat)
            if s['total']:
                print(f"  {name:40s}: {s['correct']:3d}/{s['total']:3d} = {s['correct']/s['total']:.2%}")
            else:
                print(f"  {name:40s}: N/A")
        print()
        print("--- By Question Type (fine-grained) ---")
        for qtype in sorted(type_stats.keys()):
            s = type_stats[qtype]
            if s['total']:
                print(f"  {qtype:40s}: {s['correct']:3d}/{s['total']:3d} = {s['correct']/s['total']:.2%}")
    print("=" * 60)


# ========================= Main =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LongMemEval Evaluation Pipeline")

    # Data
    p.add_argument("--data_file", required=True,
                    help="Path to longmemeval_s_cleaned.json (or longmemeval_oracle.json)")

    # Inference settings (aligned with QwenLong-L1.5 paper Section 5.1)
    p.add_argument("--model", default="QwenLong-L1.5-30B-A3B",
                    help="Model name served by vLLM")
    p.add_argument("--api_url", default=os.getenv("LLM_URL", "http://<LLM_HOST>:<PORT>/v1"),
                    help="vLLM OpenAI-compatible API endpoint")
    p.add_argument("--api_key", default=os.getenv("API_KEY", "<API_KEY>"),
                    help="API key for vLLM (usually EMPTY)")
    p.add_argument("--concurrency", type=int, default=4,
                    help="Inference concurrency (lower than CorpusQA due to ~115K input)")

    # Output paths
    p.add_argument("--infer_output", default="runs/longmemeval_infer.jsonl",
                    help="Path to save inference results")
    p.add_argument("--eval_output", default="evals/longmemeval_eval.jsonl",
                    help="Path to save evaluation results")

    # Judge settings (DeepSeek-V3, same as QwenLong-L1.5 paper Table 6)
    p.add_argument("--judge_model", default="deepseek-chat",
                    help="Judge model (deepseek-chat for DeepSeek-V3, or gpt-4o)")
    p.add_argument("--judge_api_key", required=True,
                    help="API key for judge model")
    p.add_argument("--judge_base_url", default="https://api.deepseek.com",
                    help="Base URL for judge API (https://api.deepseek.com or https://api.openai.com/v1)")

    # Control flags
    p.add_argument("--skip_infer", action="store_true",
                    help="Skip inference step (use existing infer_output)")
    p.add_argument("--skip_eval", action="store_true",
                    help="Skip evaluation step (use existing eval_output)")

    args = p.parse_args()

    if not args.skip_infer:
        asyncio.run(run_inference(args))
    if not args.skip_eval:
        run_eval(args)
