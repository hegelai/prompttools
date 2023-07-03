from functools import wraps
import logging

from prompttools.testing.error.failure import PromptTestFailure
from prompttools.testing.runner.prompt_template_runner import (
    prompt_template_test_runner,
)
from prompttools.testing.runner.system_prompt_runner import system_prompt_test_runner


def prompttest(
    model_name,
    metric_name,
    threshold,
    prompt_template_file=None,
    user_input_file=None,
    system_prompt_file=None,
    human_messages_file=None,
    is_average=None,
):
    def prompttest_decorator(eval_fn):
        @wraps(eval_fn)
        def runs_test():
            if prompt_template_file and user_input_file:
                prompt_template, user_inputs = prompt_template_test_runner.read(
                    prompt_template_file, user_input_file
                )
                prompt_template_test_runner.run(
                    model_name, prompt_template, user_inputs
                )
                prompt_template_test_runner.evaluate(metric_name, eval_fn)
                scored_template = prompt_template_test_runner.rank(
                    metric_name, is_average
                )
                if scored_template[prompt_template] < threshold:
                    logging.error(
                        "Test failed for metric: "
                        + metric_name
                        + "\nThreshold: "
                        + str(threshold)
                        + "\nActual: "
                        + str(scored_template[prompt_template])
                    )
                    raise PromptTestFailure
                return
            elif system_prompt_file and human_messages_file:
                system_prompt, human_messages = system_prompt_test_runner.read(
                    system_prompt_file, human_messages_file
                )
                system_prompt_test_runner.run(model_name, system_prompt, human_messages)
                system_prompt_test_runner.evaluate(metric_name, eval_fn)
                scored_template = system_prompt_test_runner.rank(
                    metric_name, is_average
                )
                if scored_template[system_prompt] < threshold:
                    logging.error(
                        "Test failed for metric: "
                        + metric_name
                        + "\nThreshold: "
                        + str(threshold)
                        + "\nActual: "
                        + str(scored_template[system_prompt])
                    )
                    raise PromptTestFailure
                return
            else:
                logging.error("Bad configuration for metric: " + metric_name)
                raise PromptTestFailure

        return runs_test

    return prompttest_decorator
