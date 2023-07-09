# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
import jinja2
from .harness import ExperimentationHarness
from prompttools.experiment import OpenAICompletionExperiment


class PromptTemplateExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test various prompt templates.
    We use `jinja` templates, e.g. "Answer the following question: {{input}}".

    Args:
        model_name (str): The name of the model.
        prompt_templates (List[str]): A list of prompt ``jinja``-styled templates.
        user_inputs (List[Dict[str, str]]): A list of dictionaries representing user inputs.
        use_scribe (Optional[bool], optional): Whether to use ``HegelScribe`` for logging and analytics.
            Defaults to ``False``.
        scribe_name (Optional[str], optional): The experiment name passed to ``HegelScribe``.
            Defaults to ``f"Prompt Template Experiment {model_name}"``.
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``.
    """

    PIVOT_COLUMNS = ["prompt_template", "user_input"]

    def __init__(
        self,
        model_name: str,
        prompt_templates: List[str],
        user_inputs: List[Dict[str, str]],
        use_scribe: Optional[bool] = False,
        scribe_name: Optional[str] = None,
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.environment = jinja2.Environment()
        self.experiment_classname = OpenAICompletionExperiment
        self.model_name = model_name
        self.prompt_templates = prompt_templates
        self.user_inputs = user_inputs
        self.use_scribe = use_scribe
        self.scribe_name = (
            f"Prompt Template Experiment {model_name}"
            if scribe_name is None
            else scribe_name
        )
        self.model_arguments = {} if model_arguments is None else model_arguments
        super().__init__()

    def prepare(self) -> None:
        r"""
        Creates prompts from templates to use for the experiment, and then initializes and prepares the experiment.
        """
        self.input_pairs_dict = {}
        rendered_inputs = []
        for pt in self.prompt_templates:
            for user_input in self.user_inputs:
                template = self.environment.from_string(pt)
                prompt = template.render(**user_input)
                rendered_inputs.append(prompt)
                self.input_pairs_dict[prompt] = (pt, user_input)
        self.experiment = self.experiment_classname(
            [self.model_name],
            rendered_inputs,
            self.use_scribe,
            self.scribe_name,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self):
        if not self.experiment:
            self.prepare()
        super().run()
