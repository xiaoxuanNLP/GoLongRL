import argparse

from evalscope import run_task
from evalscope.config import TaskConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--judge_api_key', type=str, required=True)
parser.add_argument('--judge_model_id', type=str, default='deepseek-chat')
parser.add_argument('--judge_api_url', type=str, default='https://api.deepseek.com/v1')
parser.add_argument('--subsets', type=str, default='complong_testmini,compshort_testmini,simplong_testmini,simpshort_testmini')
parser.add_argument('--max_input_tokens', type=int, default=131000)
parser.add_argument('--max_tokens', type=int, default=51200)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()

JUDGE_ARGS = {
    'model_id': args.judge_model_id,
    'api_url': args.judge_api_url,
    'api_key': args.judge_api_key,
}

task_cfg = TaskConfig(
    model=args.model_name,
    api_url=f'http://127.0.0.1:{args.port}/v1',
    api_key='EMPTY',
    datasets=['docmath'],
    dataset_args={
        'docmath': {
            'dataset_id': args.dataset_path,
            'subset_list': args.subsets.split(','),
            'filters': {'remove_until': '</think>'},
            'extra_params': {
                'tokenizer_path': args.model_path,
                'max_input_tokens': args.max_input_tokens,
            },
        }
    },
    generation_config={
        'max_tokens': args.max_tokens,
        'temperature': 0.7,
        'top_p': 0.95,
    },
    judge_model_args=JUDGE_ARGS,
    repeats=1,
    eval_batch_size=args.eval_batch_size,
    use_cache=args.output_path,
    work_dir=args.output_path,
    limit=args.limit,
)

run_task(task_cfg=task_cfg)
