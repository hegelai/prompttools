# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import itertools

from prompttools.mock.mock import mock_replicate_stable_diffusion_completion_fn

import os


try:
    import replicate

except ImportError:
    replicate = None

from .experiment import Experiment


class ReplicateExperiment(Experiment):
    r"""
    Perform an experiment with the Replicate API to test different embedding functions or retrieval arguments.
    You can query from an existing table, or create a new one (and insert documents into it) during
    the experiment.

    Note:
        Set your API token to ``os.environ["REPLICATE_API_TOKEN"]``.

    Args:
        models (list[str]): "stability-ai/stable-diffusion:27b93a2413e"
        input_kwargs (dict[str, list]): keyword arguments that can be used across all models

        model_specific_kwargs (dict[str, dict[str, list]]): model-specific keyword arguments that will only be used
            by a specific model (e.g. ``stability-ai/stable-diffusion:27b93a2413``
    """

    def __init__(
        self, models: list[str], input_kwargs: dict[str, list], model_specific_kwargs: dict[str, dict[str, list]] = {}
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
        self.models = models
        self.input_kwargs = input_kwargs
        self.model_specific_kwargs = model_specific_kwargs
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_replicate_stable_diffusion_completion_fn
        else:
            self.completion_fn = self.replicate_completion_fn
        super().__init__()

    def prepare(self):
        for model in self.models:
            for base_combo in itertools.product(*self.input_kwargs.values()):
                arg_dict = dict(zip(self.input_kwargs.keys(), base_combo))
                # arg_dict['model_version'] = model
                for model_combo in itertools.product(*self.model_specific_kwargs[model].values()):
                    model_arg_dict = dict(zip(self.model_specific_kwargs[model].keys(), model_combo))
                    for k, v in model_arg_dict.items():
                        arg_dict[k] = v
                    self.argument_combos.append({"model_version": model, "input": arg_dict})

    @staticmethod
    def replicate_completion_fn(model_version: str, **kwargs):
        return replicate.run(model_version, input=kwargs)

    @staticmethod
    def _extract_responses(output: dict) -> list[str]:
        # TODO: May need to extract image from the URI, perhaps with `cv2.imread`
        return output["data"]
