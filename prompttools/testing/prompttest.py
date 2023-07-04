from functools import wraps
import logging

from prompttools.testing.threshold_type import ThresholdType
from prompttools.testing.error.failure import PromptTestSetupException
from prompttools.testing.runner.prompt_template_runner import (
    run_prompt_template_test,
    run_prompt_template_test_from_files,
)
from prompttools.testing.runner.system_prompt_runner import (
    run_system_prompt_test,
    run_system_prompt_test_from_files,
)

TESTS_TO_RUN = []


def prompttest(
    model_name,
    metric_name,
    threshold,
    prompt_template_file=None,
    user_input_file=None,
    system_prompt_file=None,
    human_messages_file=None,
    prompt_template=None,
    user_input=None,
    system_prompt=None,
    human_messages=None,
    is_average=None,
    use_input_pairs=False,
    threshold_type=ThresholdType.MINIMUM,
    model_arguments={},
):
    """
    Creates a decorator for prompt tests, which can annotate evaluation functions.
    This enables developers to create a prompt test suite from their evaluations.
    """

    def prompttest_decorator(eval_fn):
        @wraps(eval_fn)
        def runs_test():
            if prompt_template_file and user_input_file:
                return run_prompt_template_test_from_files(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    threshold_type,
                    is_average,
                    prompt_template_file,
                    user_input_file,
                    use_input_pairs,
                    model_arguments,
                )
            elif prompt_template and user_input:
                return run_prompt_template_test(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    threshold_type,
                    is_average,
                    prompt_template,
                    user_input,
                    use_input_pairs,
                    model_arguments,
                )
            elif system_prompt_file and human_messages_file:
                return run_system_prompt_test_from_files(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    threshold_type,
                    is_average,
                    system_prompt_file,
                    human_messages_file,
                    use_input_pairs,
                    model_arguments,
                )
            elif system_prompt and human_messages:
                return run_system_prompt_test(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    threshold_type,
                    is_average,
                    system_prompt,
                    human_messages,
                    use_input_pairs,
                    model_arguments,
                )
            else:
                logging.error("Bad configuration for metric: " + metric_name)
                raise PromptTestSetupException

        TESTS_TO_RUN.append(runs_test)
        return runs_test

    return prompttest_decorator


def main():
    logging.getLogger().setLevel(logging.WARNING)
    failures = sum([test() for test in TESTS_TO_RUN])
    if failures == 0:
        print("Ok")
        exit(0)
    else:
        print("Tests failed: " + str(failures))
        exit(1)
