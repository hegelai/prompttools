from typing import Dict, List, Optional
import jinja2
from prompttools.harness.harness import ExperimentationHarness


class PromptTemplateExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for prompt templates.
    We use jinja templates, e.g. "Answer the following question: {{input}}".
    """

    def __init__(
        self,
        experiment_classname,
        model_name: str,
        prompt_templates: List[str],
        user_inputs: List[Dict[str, str]],
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.environment = jinja2.Environment()
        self.experiment_classname = experiment_classname
        self.model_name = model_name
        self.model_arguments = model_arguments
        self.prompt_templates = prompt_templates
        self.user_inputs = user_inputs

    def prepare(self):
        rendered_inputs = []
        for template in self.prompt_templates:
            for user_input in self.user_inputs:
                template = self.environment.from_string(template)
                rendered_inputs.append(template.render(**user_input))
        self.experiment = self.experiment_classname(
            [self.model_name],
            rendered_inputs,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()
