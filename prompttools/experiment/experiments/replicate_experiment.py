# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import logging
import itertools
from functools import partial

from prompttools.mock.mock import mock_replicate_stable_diffusion_completion_fn
from IPython.display import display, HTML
from tabulate import tabulate

from prompttools.selector.prompt_selector import PromptSelector
from ..widgets.utility import is_interactive


import os


try:
    import replicate

except ImportError:
    replicate = None

from .experiment import Experiment


class ReplicateExperiment(Experiment):
    r"""
    Perform an experiment with the Replicate API for both image models and LLMs.

    Note:
        Set your API token to ``os.environ["REPLICATE_API_TOKEN"]``.
        If you are using an image model, set ``use_image_model=True`` as input argument.

    Args:
        models (list[str]): "stability-ai/stable-diffusion:27b93a2413e"
        input_kwargs (dict[str, list]): keyword arguments that can be used across all models
        model_specific_kwargs (dict[str, dict[str, list]]): model-specific keyword arguments that will only be used
            by a specific model (e.g. ``stability-ai/stable-diffusion:27b93a2413``
        use_image_model (bool): Defaults to ``False``, must set to ``True`` to render output from image models.
    """

    def __init__(
        self,
        models: list[str],
        input_kwargs: dict[str, list],
        model_specific_kwargs: dict[str, dict[str, list]] = {},
        use_image_model: bool = False,
    ):
        if replicate is None:
            raise ModuleNotFoundError(
                "Package `replicate` is required to be installed to use this experiment."
                "Please use `pip install replicate` to install the package"
            )
        try:
            os.environ["REPLICATE_API_TOKEN"]
        except KeyError:
            raise RuntimeError('`os.environ["REPLICATE_API_TOKEN]` needs to be set.')
        
        # If we are using a prompt selector, we need to
        # render the prompts from the selector        
        if isinstance(input_kwargs['prompt'][0], PromptSelector):
            input_kwargs['prompt'] = [selector.for_llama() for selector in input_kwargs['prompt']]

        self.models = models
        self.input_kwargs = input_kwargs
        self.model_specific_kwargs = model_specific_kwargs
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_replicate_stable_diffusion_completion_fn
        else:
            self.completion_fn = self.replicate_completion_fn
        super().__init__()
        self.image_experiment = use_image_model

    def prepare(self):
        for model in self.models:
            for base_combo in itertools.product(*self.input_kwargs.values()):
                arg_dict = dict(zip(self.input_kwargs.keys(), base_combo))
                # arg_dict['model_version'] = model
                for model_combo in itertools.product(*self.model_specific_kwargs[model].values()):
                    model_arg_dict = dict(zip(self.model_specific_kwargs[model].keys(), model_combo))
                    for k, v in model_arg_dict.items():
                        arg_dict[k] = v
                    arg_dict["model_version"] = model
                    self.argument_combos.append(arg_dict)

    @staticmethod
    def replicate_completion_fn(model_version: str, **kwargs):
        return replicate.run(model_version, input=kwargs)

    def _extract_responses(self, output) -> str:
        if self.image_experiment:
            return output[0]  # Output should be a list of URIs
        else:  # Assume `output` is a generator of text
            res = ""
            for item in output:
                res += item
            return res

    @staticmethod
    def _image_tag(url, image_width):
        r"""
        Create the HTML code to render the image.
        """
        return f'<img src="{url}" width="{image_width}"/>'

    def visualize(
        self, get_all_cols: bool = False, pivot: bool = False, pivot_columns: list = [], image_width: int = 300
    ) -> None:
        if not self.image_experiment:
            super().visualize(get_all_cols, pivot, pivot_columns)
        else:
            if pivot:
                table = self.pivot_table(pivot_columns, get_all_cols=get_all_cols)
            else:
                table = self.get_table(get_all_cols)

            images = table["response"].apply(partial(self._image_tag, image_width=image_width))
            table["images"] = images

            if is_interactive():
                display(HTML(table.to_html(escape=False, columns=[col for col in table.columns if col != "response"])))
            else:
                logging.getLogger().setLevel(logging.INFO)
                logging.info(tabulate(table, headers="keys", tablefmt="psql"))
