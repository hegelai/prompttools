# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, List


class Benchmark:
    r"""
    Benchmark models using defined data sets.
    Find example under benchmarks/examples/benchmarking.ipynb.

    Args:
    ----
        experiments (list(experiment types)): list of experiments
        eval_methods (list(eval methods)): list of evaluation methods to measure response similarity
        prompts (list(str)): list of queries, questions, prompts for LLMs to respond to
        response_options (list(str)): possible responses to measure against
        correct_response_index (list(int)): list of index of correct response in response_options
    """

    def __init__(
        self,
        experiments: List[Any],
        eval_methods: List[Any],
        prompts: List[str],
        response_options: List[str],
        correct_response_index: List[int]
    ):
        self.experiments = experiments
        self.eval_methods = eval_methods
        self.prompts = prompts
        self.response_options = response_options
        self.correct_response_index = correct_response_index
        pass

    def run(
        self,
        early_stopping
    ):
        r"""
        Run model experiments to measure response quality.

        Args:
        ----
        early_stopping: maximum time to allow benchmark to run
        """
        pass
