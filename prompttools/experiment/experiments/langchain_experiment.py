# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List
import itertools

from pydantic.main import ModelMetaclass

from time import perf_counter
import logging

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.mock.mock import mock_lc_completion_fn

from .experiment import Experiment
from .error import PromptExperimentException

VALID_TASKS = ("text2text-generation", "text-generation", "summarization")


class SequentialChainExperiment(Experiment):
    """
    Experiment for LangChain sequential chains.
    """

    MODEL_PARAMETERS = ["repo_id", "task"]

    CALL_PARAMETERS = ["prompt"]

    def __init__(
        self,
        llms: List[ModelMetaclass],
        **kwargs: Dict[str, object],
    ):
        self.completion_fn = self.lc_completion_fn
        if os.getenv("DEBUG", default=False):
            self.completion_fn = mock_lc_completion_fn

    def lc_completion_fn(
        self,
        **params: Dict[str, Any],
    ):
        pass


class RouterChainExperiment(Experiment):
    """
    Experiment for LangChain router chains.
    """
    # TODO: functionality for router chains
