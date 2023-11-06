# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from .ModelEvaluator import ModelEvaluator
from .GptEvaluator import GptEvaluator
from .AnthropicEvaluator import AnthropicEvaluator

Evaluators = [GptEvaluator(), AnthropicEvaluator()]


def get_evaluator_for_model(model: str) -> ModelEvaluator:
    for evaluator in Evaluators:
        if evaluator.supports_model(model):
            return evaluator
