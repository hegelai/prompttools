from functools import wraps
import logging

from prompttools.testing.error.failure import PromptTestFailure
from prompttools.testing.runner.prompt_template_runner import (
    run_prompt_template_test,
    run_prompt_template_test_from_files,
)
from prompttools.testing.runner.system_prompt_runner import (
    run_system_prompt_test,
    run_system_prompt_test_from_files,
)


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
):
    def prompttest_decorator(eval_fn):
        @wraps(eval_fn)
        def runs_test():
            if prompt_template_file and user_input_file:
                run_prompt_template_test_from_files(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    is_average,
                    prompt_template_file,
                    user_input_file,
                )
            elif prompt_template and user_input:
                run_prompt_template_test(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    is_average,
                    prompt_template,
                    user_input,
                )
            elif system_prompt_file and human_messages_file:
                run_system_prompt_test_from_files(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    is_average,
                    system_prompt_file,
                    human_messages_file,
                )
            elif system_prompt and human_messages:
                run_system_prompt_test(
                    model_name,
                    metric_name,
                    eval_fn,
                    threshold,
                    is_average,
                    system_prompt,
                    human_messages,
                )
            else:
                logging.error("Bad configuration for metric: " + metric_name)
                raise PromptTestFailure

        return runs_test

    return prompttest_decorator
