"""Shared logic: load questions, build DeepEval test cases, define metrics."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase

try:  # newer DeepEval
    from deepeval.test_case import SingleTurnParams as LLMTestCaseParams
except ImportError:  # older DeepEval
    from deepeval.test_case import LLMTestCaseParams

from llm_client import ask_llm

load_dotenv()

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
EVAL_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")  # model that does the scoring
THRESHOLD = float(os.getenv("PASS_THRESHOLD", "0.7"))


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(questions: list[dict]) -> list[LLMTestCase]:
    """Ask OpenAI every question and wrap input/output into DeepEval test cases."""
    cases = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] Asking: {q['input']}")
        answer = ask_llm(q["input"])
        cases.append(
            LLMTestCase(
                input=q["input"],
                actual_output=answer,
                expected_output=q.get("expected_output"),
            )
        )
    return cases


def relevancy_metric() -> AnswerRelevancyMetric:
    """Does the answer actually address the question? (input vs output)"""
    return AnswerRelevancyMetric(threshold=THRESHOLD, model=EVAL_MODEL, include_reason=True)


def correctness_metric() -> GEval:
    """How close is the answer to the expected answer? (needs expected_output)"""
    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct and "
            "consistent with the expected output. Minor wording differences are fine."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=EVAL_MODEL,
    )
