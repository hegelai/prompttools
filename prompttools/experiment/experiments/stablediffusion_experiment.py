# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List
import itertools

from time import perf_counter
import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
except ImportError:
    DiffusionPipeline = None

from prompttools.mock.mock import mock_stable_diffusion

from .experiment import Experiment
from .error import PromptExperimentException


class StableDiffusionExperiment(Experiment):
    r"""
    Experiment for experiment with the Stable Diffusion model.

    Args:
        hf_model_path (str): path to model on hugging face
        use_auth_token (bool): boolean to determine if hf login is necessary [needed without GPU]
        kwargs (dict): keyword arguments to call the model with
    """

    MODEL_PARAMETERS = ["hf_model_path", "prompt"]

    CALL_PARAMETERS: List[str] = ["prompt"]

    def __init__(
        self,
        hf_model_path: List[str],
        prompt: List[str],
        compare_images_folder: str,
        use_auth_token: bool = False,
        **kwargs: Dict[str, object],
    ):
        if DiffusionPipeline is None:
            raise ModuleNotFoundError(
                "Package `diffusers` is required to be installed to use this experiment."
                "Please use `pip install diffusers` and \
                `pip install invisible_watermark transformers accelerate safetensors` to install the package"
            )
        if cv2 is None:
            raise ModuleNotFoundError(
                "Package `cv2` is required to be installed to use this experiment."
                "Please use `pip opencv-python` to install the package"
            )
        self.use_auth_token = use_auth_token
        self.completion_fn = self.sd_completion_fn
        self.compare_images_folder = compare_images_folder
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_stable_diffusion
        self.model_params = dict(hf_model_path=hf_model_path)

        self.call_params = dict(prompt=prompt)
        for k, v in kwargs.items():
            self.CALL_PARAMETERS.append(k)
            self.call_params[k] = v

        self.all_args = self.model_params | self.call_params
        super().__init__()

    def prepare(self) -> None:
        r"""
        Combo builder.
        """
        self.model_argument_combos = [
            dict(zip(self.model_params, val, strict=False)) for val in itertools.product(*self.model_params.values())
        ]
        self.call_argument_combos = [
            dict(zip(self.call_params, val, strict=False)) for val in itertools.product(*self.call_params.values())
        ]

    def sd_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request.
        """
        client = params["client"]
        image_folder = params["image_folder"]
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        response = client(params["prompt"])
        img = response["images"][0]
        img_path = params["image_folder"] + "_".join(params["prompt"].split(" ")) + ".png"
        img.save(img_path)
        main_img = cv2.imread(img_path)
        logging.info("Resizing comparison images to match Stable Diffusion response image size for comparison.")
        for fil in os.listdir(self.compare_images_folder):
            compare_img = cv2.imread(self.compare_images_folder + fil)
            compare_img = cv2.resize(compare_img, (main_img.shape[1], main_img.shape[0]))
            cv2.imwrite(self.compare_images_folder + fil, compare_img)
        return main_img

    def run(
        self,
        runs: int = 1,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For each combination, it will execute `runs` times, default to 1.
        # TODO This can be done with an async queue.
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        results = []
        latencies = []
        for model_combo in self.model_argument_combos:
            for call_combo in self.call_argument_combos:
                if self.use_auth_token:
                    client = StableDiffusionPipeline.from_pretrained(
                        model_combo["hf_model_path"], use_auth_token=self.use_auth_token
                    )
                else:
                    client = DiffusionPipeline.from_pretrained(
                        model_combo["hf_model_path"], **{k: call_combo[k] for k in call_combo if k != "prompt"}
                    )
                    client.to("cuda")
                for _ in range(runs):
                    call_combo["client"] = client
                    start = perf_counter()
                    res = self.completion_fn(**call_combo)
                    latencies.append(perf_counter() - start)
                    results.append(res)
                    self.argument_combos.append(model_combo | call_combo)
        if len(results) == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException
        self._construct_result_dfs(self.argument_combos, results, latencies, extract_response_equal_full_result=True)

    @staticmethod
    def _extract_responses(output: object) -> object:
        return cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
