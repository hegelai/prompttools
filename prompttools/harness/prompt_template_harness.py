# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Type
import jinja2
from .harness import ExperimentationHarness, Experiment
import logging


class PromptTemplateExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test various prompt templates.
    We use `jinja` templates, e.g. "Answer the following question: {{input}}".

    Args:
        experiment (Type[Experiment]): The experiment constructor that you would like to execute within the harness
            (e.g. ``prompttools.experiment.OpenAICompletionExperiment``)
        model_name (str): The name of the model.
        prompt_templates (List[str]): A list of prompt ``jinja``-styled templates.
        user_inputs (List[Dict[str, str]]): A list of dictionaries representing user inputs.
        model_arguments (Optional[Dict[str, object]], optional): Additional arguments for the model.
            Defaults to ``None``.
    """

    PIVOT_COLUMNS = ["prompt_template", "user_input"]

    def __init__(
        self,
        experiment: Type[Experiment],
        model_name: str,
        prompt_templates: List[str],
        user_inputs: List[Dict[str, str]],
        model_arguments: Optional[Dict[str, object]] = None,
    ):
        self.environment = jinja2.Environment()
        self.experiment_cls_constructor = experiment
        self.model_name = model_name
        self.prompt_templates = prompt_templates
        self.user_inputs = user_inputs
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
        self.experiment = self.experiment_cls_constructor(
            [self.model_name],
            rendered_inputs,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()

    def run(self):
        if not self.experiment:
            self.prepare()
        super().run()
