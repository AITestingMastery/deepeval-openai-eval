"""Main entry point.

    python run_eval.py

1. Reads questions.json
2. Sends each question to OpenAI
3. Scores input vs output with DeepEval
4. Uploads the results to the Confident AI (DeepEval) dashboard if you are logged in
"""

from deepeval import evaluate

from eval_core import (
    build_test_cases,
    correctness_metric,
    load_questions,
    relevancy_metric,
)


def main() -> None:
    questions = load_questions()
    test_cases = build_test_cases(questions)

    with_expected = [tc for tc in test_cases if tc.expected_output]
    without_expected = [tc for tc in test_cases if not tc.expected_output]

    # Questions with an expected answer: relevancy + correctness
    if with_expected:
        print(f"\nScoring {len(with_expected)} question(s) with expected answers...")
        evaluate(
            test_cases=with_expected,
            metrics=[relevancy_metric(), correctness_metric()],
        )

    # Questions without an expected answer: relevancy only
    if without_expected:
        print(f"\nScoring {len(without_expected)} question(s) without expected answers...")
        evaluate(test_cases=without_expected, metrics=[relevancy_metric()])

    print("\nDone. If you ran `deepeval login`, open the Confident AI dashboard to see the results.")


if __name__ == "__main__":
    main()
