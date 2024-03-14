# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Union
import itertools

from time import perf_counter
import logging

try:
    import librosa
except ImportError:
    librosa = None

try:
    from audiocraft.models import MusicGen
    music_gen = MusicGen.get_pretrained
    from audiocraft.data.audio import audio_write
except ImportError:
    music_gen = None

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.mock.mock import mock_music_gen_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException


class MusicGenExperiment(Experiment):
    r"""
    Experiment for MusicGen's API.
    It accepts lists for each argument passed into MusicGen's API,
    then creates a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments. For example, ``kwargs`` should have string keys,
          with ``list``s being the values.

    Args:
        repo_id (List[str]): IDs of repository (e.g. [`facebook/musicgen-small`]).
        prompt (List[str] | List[PromptSelector]): list of prompts to test
        task (List[str]): List of tasks in strings. Determines whether to force a task instead of using task
            specified in the repository.
        **kwargs (Dict[str, list[object]]): Keyword parameters used in the call to ``MusicGen``.
            The values should be ``list``s.
    """

    MODEL_PARAMETERS = ["repo_id", "task"]

    CALL_PARAMETERS = ["prompt"]

    def __init__(
        self,
        repo_id: List[str],
        prompt: Union[List[str], List[PromptSelector]],
        duration: List[int] = [5],
        **kwargs: Dict[str, list[object]],
    ):
        if music_gen is None:
            raise ModuleNotFoundError(
                "Package `audiocraft` is required to be installed to use this experiment."
                "Please use `pip install audiocraft` to install the package"
            )
        if librosa is None:
            raise ModuleNotFoundError(
                "Package `librosa` is required to be installed to use this experiment."
                "Please use `pip install librosa` to install the package"
            )
        if "generated_audio_files" not in os.listdir():
            os.mkdir("generated_audio_files")
        self.duration = duration
        self.completion_fn = self.music_gen_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_music_gen_completion_fn
        self.model_params = dict(repo_id=repo_id, duration=self.duration)

        # If we are using a prompt selector, we need to render
        # messages, as well as create prompt_keys to map the messages
        # to corresponding prompts in other models.
        if isinstance(prompt[0], PromptSelector):
            self.prompt_keys = {selector.for_music_gen(): selector.for_music_gen() for selector in prompt}
            prompt = [selector.for_music_gen() for selector in prompt]
        else:
            self.prompt_keys = prompt

        self.call_params = dict(prompt=prompt)
        for k, v in kwargs.items():
            self.CALL_PARAMETERS.append(k)
            self.call_params[k] = v

        self.all_args = self.model_params | self.call_params
        super().__init__()

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.model_argument_combos = [
            dict(zip(self.model_params, val)) for val in itertools.product(*self.model_params.values())
        ]
        self.call_argument_combos = [
            dict(zip(self.call_params, val)) for val in itertools.product(*self.call_params.values())
        ]

    def music_gen_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        r"""
        Local model helper function to make request
        """
        signal, sr = librosa.load(f'generated_audio_files/{params["prompt"]}.wav')
        # Extract relevant features, for example, Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr)
        return mfccs.flatten()

    def run(
        self,
        runs: int = 1,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.
        For each combination, it will execute `runs` times, default to 1.
        # TODO This can be done with an async queue
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        results = []
        latencies = []
        for model_combo in self.model_argument_combos:
            client = music_gen(
                name=model_combo["repo_id"],
            )
            client.set_generation_params(duration=8)
            for call_combo in self.call_argument_combos:
                wav = client.generate(call_combo["prompt"])
                for _, one_wav in enumerate(wav):
                    audio_write(f'generated_audio_files/{call_combo["prompt"]}', one_wav.cpu(), client.sample_rate, strategy="loudness")
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
    def _extract_responses(output: List[Dict[str, object]]) -> List[float]:
        return output
