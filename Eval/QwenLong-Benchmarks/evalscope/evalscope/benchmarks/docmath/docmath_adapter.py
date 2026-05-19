from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


@register_benchmark(
    BenchmarkMeta(
        name='docmath',
        pretty_name='DocMath',
        tags=[Tags.REASONING, Tags.MATH, Tags.LONG_CONTEXT],
        description="""
## Overview

DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires models to comprehend long and specialized documents and perform numerical reasoning to answer questions.

## Task Description

- **Task Type**: Document-based Mathematical Reasoning
- **Input**: Long document context + numerical reasoning question
- **Output**: Numerical answer with reasoning
- **Focus**: Long-context comprehension and quantitative reasoning

## Key Features

- Long specialized documents requiring comprehension
- Numerical reasoning within document context
- Multiple complexity levels (comp/simp, long/short)
- Tests real-world document understanding
- Requires both reading comprehension and math skills

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for answer evaluation
- Subsets: complong_testmini, compshort_testmini, simplong_testmini, simpshort_testmini
- Answer format: "Therefore, the answer is (answer)"
""",  # noqa: E501
        dataset_id='yale-nlp/DocMath-Eval',
        metric_list=['acc', 'llm_judge_acc', 'max_acc'],
        subset_list=['complong_testmini', 'compshort_testmini', 'simplong_testmini', 'simpshort_testmini'],
        eval_split='test',
        prompt_template=TEMPLATE_0SHOT,
        extra_params={
            'tokenizer_path': {
                'type': 'str',
                'description': 'Tokenizer path for token-level middle truncation.',
                'value': None,
            },
            'max_input_tokens': {
                'type': 'int',
                'description': 'Max input token length. If set, inputs exceeding this will be truncated using middle strategy.',
                'value': None,
            },
        },
    )
)
class DocMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = True  # Enable LLM judge for DocMath
        self.split_as_subset = True  # Use split as subset for DocMath
        self._tokenizer = None
        self.max_input_tokens: Optional[int] = self.extra_params.get('max_input_tokens', None)
        self.tokenizer_path: Optional[str] = self.extra_params.get('tokenizer_path', None)

    @property
    def tokenizer(self):
        """Lazily initialize tokenizer when first needed."""
        if self._tokenizer is None and self.tokenizer_path:
            from modelscope import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, fix_mistral_regex=True)
        return self._tokenizer

    def _truncate_middle(self, text: str, max_tokens: int) -> str:
        """Truncate text using middle strategy on token level: keep beginning and end tokens, remove middle."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        half = max_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
        return self.tokenizer.decode(truncated_tokens)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        ground_truth = record['ground_truth']
        context = '\n'.join(record['paragraphs'])
        question = record['question']
        message = self.prompt_template.format(context=context, question=question)
        return Sample(
            input=[ChatMessageUser(content=message)],
            target=str(ground_truth),
            metadata={
                'question_id': record.get('question_id', ''),
                'answer_type': type(ground_truth).__name__,
                'question': question,
            }
        )

    def _on_inference_start(self, model, sample: Sample) -> None:
        """Apply middle truncation on token level to the formatted input if max_input_tokens is set."""
        if self.max_input_tokens is None or self.tokenizer is None:
            return
        for i, msg in enumerate(sample.input):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                truncated = self._truncate_middle(msg.content, self.max_input_tokens)
                sample.input[i] = msg.model_copy(update={'content': truncated})

    def extract_answer(self, prediction: str, task_state: TaskState):
        """
        Extract the answer from the model prediction.
        """
        from .utils import extract_answer

        extracted_answer = extract_answer(prediction)
        return extracted_answer

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Calculate accuracy score by matching prediction with reference.
        """
        from .utils import get_acc

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        answer_type = task_state.metadata.get('answer_type', 'unknown')
        accuracy = get_acc(prediction=filtered_prediction, gt=reference, answer_type=answer_type)
        score.value = {'acc': accuracy}
        score.main_score_name = 'acc'

        return score

    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        """Calculate metrics: rule-based acc, LLM judge acc, and max of both."""
        assert task_state.completed, 'TaskState must be completed before calculating metrics.'

        prediction = task_state.output.completion if task_state.output is not None else ''
        filtered_prediction = self.filter_prediction(prediction, task_state)

        rule_score = self.match_score(
            original_prediction=prediction,
            filtered_prediction=filtered_prediction,
            reference=task_state.target,
            task_state=task_state,
        )
        rule_acc = float(rule_score.main_value)

        if self.use_llm_judge:
            llm_score = self.llm_match_score(
                original_prediction=prediction,
                filtered_prediction=filtered_prediction,
                reference=task_state.target,
                task_state=task_state,
            )
            llm_acc = float(llm_score.main_value)
        else:
            llm_acc = rule_acc

        final_score = Score(
            extracted_prediction=filtered_prediction,
            prediction=prediction,
            value={
                'acc': rule_acc,
                'llm_judge_acc': llm_acc,
                'max_acc': max(rule_acc, llm_acc),
            },
            main_score_name='max_acc',
            explanation=llm_score.explanation if self.use_llm_judge else None,
            metadata=llm_score.metadata if self.use_llm_judge else None,
        )

        return SampleScore(
            score=final_score,
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Use LLM judge to evaluate the prediction against the reference.
        """
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        question = task_state.metadata.get('question', '')

        # Get grading response
        prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=reference, answer_2=filtered_prediction)
        orm_response = self.llm_judge.judge(prompt, system_prompt=GENERAL_ORM_PROMPT)

        # Parse grading response
        if 'YES' in orm_response:
            accuracy = 1.0
        else:
            accuracy = 0.0

        score.value = {'acc': accuracy}
        score.explanation = f'LLM judge: {orm_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id
        }
        score.main_score_name = 'acc'

        return score
