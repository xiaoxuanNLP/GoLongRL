from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser
import json
import os
import time
from tqdm import tqdm
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vLLM")

def create_parser():
    parser = FlexibleArgumentParser()
    engine_group = parser.add_argument_group("Engine Initialization Parameters")
    engine_group.add_argument("--model", type=str, required=True, help="Path to the model directory")
    engine_group.add_argument("--max-model-len", type=int, default=81920)
    engine_group.add_argument("--rope-scaling", type=str,
                            default='{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}')
    engine_group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    engine_group.add_argument("--tensor-parallel-size", type=int, default=2)
    engine_group.add_argument("--max-num-seqs", type=int, default=128)
    engine_group.add_argument("--seed", type=int, default=0)

    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=65536)
    sampling_group.add_argument("--temperature", type=float, default=0.6)
    sampling_group.add_argument("--top-p", type=float, default=0.95)
    sampling_group.add_argument("--top-k", type=int, default=-1)
    sampling_group.add_argument("--n", type=int, default=1)

    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--dataset-path", type=str, required=True)
    data_group.add_argument("--output-data", type=str, default="./output")
    data_group.add_argument("--cache-dir", type=str, default="./cache")
    data_group.add_argument("--batch-size", type=int, default=1)

    return parser


def main(args: dict):
    data_path = args.pop("dataset_path")
    output_dir = args.pop("output_data")
    batch_size = args.pop("batch_size")
    data_list = load_dataset(data_path)
    n = args.pop("n")

    os.makedirs(output_dir, exist_ok=True)

    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    logger.info("=" * 50)
    logger.info("Initializing vLLM Engine")
    logger.info(f"Model: {args.get('model', 'unknown')}")
    logger.info(f"Tensor Parallel Size: {args.get('tensor_parallel_size', 1)}")
    logger.info(f"GPU Memory Utilization: {args.get('gpu_memory_utilization', 0.9)}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info("=" * 50)

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    model_path = args.get("model")
    llm = LLM(
        model=args.get("model"),
        tensor_parallel_size=args.get("tensor_parallel_size", 8),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.9),
        enable_prefix_caching=True,
        max_num_seqs=args.get("max_num_seqs", 128),
        enforce_eager=False,
        seed=args.get("seed", 0),
    )

    logger.info("vLLM engine initialization completed")

    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.05
    )
    print(sampling_params)

    # Use only the final path components to keep the filename concise
    model_path_basename = os.path.basename(model_path.rstrip("/"))
    data_path_basename = os.path.basename(data_path)
    result_path = os.path.join(output_dir, f"{model_path_basename}_{data_path_basename}.jsonl")
    process_data_batch(llm, data_list, sampling_params, result_path, batch_size)


def process_data_batch(llm, data_list, sampling_params, result_path, batch_size):
    results_file = result_path

    processed_count = 0

    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f) // sampling_params.n
        logger.info(f"Found existing results file, {processed_count} entries already processed")

    remaining_data = data_list[processed_count:]
    if not remaining_data:
        logger.info("All data processing completed")
        return

    logger.info(f"Starting to process remaining {len(remaining_data)} entries")

    with open(results_file, 'a', encoding='utf-8') as f:
        for batch_start in range(0, len(remaining_data), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_data))
            batch = remaining_data[batch_start:batch_end]

            try:
                batch_messages = []
                original_items = []

                for item in batch:
                    original_items.append(item)
                    if "messages" in item:
                        batch_messages.append(item["messages"])
                    else:
                        logger.warning(f"Data item missing 'messages' field: {item}")
                        batch_messages.append([{"role": "user", "content": item["problem"]}])

                logger.debug(f"Starting batch processing {batch_start} to {batch_end-1}")
                outputs = llm.chat(batch_messages, sampling_params, use_tqdm=True)

                for i, output in enumerate(outputs):
                    item_idx = processed_count + batch_start + i
                    original_item = original_items[i]

                    for j, candidate in enumerate(output.outputs):
                        complete_messages = batch_messages[i].copy()
                        complete_messages.append({"role": "assistant", "content": candidate.text})

                        result = {
                            "id": f"{item_idx}_{j}",
                            "meta": original_item,
                            "messages": complete_messages
                        }

                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                    f.flush()
                logger.info(f"Processed {processed_count + batch_end}/{len(data_list)} entries")

            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end-1}: {str(e)}")
                for i in range(len(batch)):
                    item_idx = processed_count + batch_start + i
                    original_item = batch[i]

                    error_result = {
                        "id": item_idx,
                        "meta": original_item,
                        "messages": original_item.get("messages", []),
                        "error": str(e)
                    }
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()

    logger.info(f"Data processing completed, results saved to {results_file}")


def load_dataset(dataset_path: str) -> list:
    data_list = []
    logger.info(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Unable to parse JSON line: {line}")

    logger.info(f"Loaded {len(data_list)} entries")
    return data_list


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    main(args)