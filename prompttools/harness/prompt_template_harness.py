from typing import Dict, List, Optional
import jinja2
from prompttools.harness.harness import ExperimentationHarness
from prompttools.experiment.openai_completion_experiment import (
    OpenAICompletionExperiment,
)


class PromptTemplateExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used for prompt templates.
    We use jinja templates, e.g. "Answer the following question: {{input}}".
    """

    PIVOT_COLUMNS = ["prompt_template", "user_input"]

    def __init__(
        self,
        model_name: str,
        prompt_templates: List[str],
        user_inputs: List[Dict[str, str]],
        use_dialectic_scribe: bool = False,
        dialectic_scribe_name: str = "Prompt Template Experiment",
        model_arguments: Optional[Dict[str, object]] = {},
    ):
        self.environment = jinja2.Environment()
        self.experiment_classname = OpenAICompletionExperiment
        self.model_name = model_name
        self.prompt_templates = prompt_templates
        self.user_inputs = user_inputs
        self.use_dialectic_scribe = use_dialectic_scribe
        self.dialectic_scribe_name = dialectic_scribe_name
        self.model_arguments = model_arguments

    def prepare(self) -> None:
        """
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
            self.use_dialectic_scribe,
            self.dialectic_scribe_name,
            **self._prepare_arguments(self.model_arguments),
        )
        super().prepare()
