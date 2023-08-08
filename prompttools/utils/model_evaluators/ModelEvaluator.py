# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: str, prompt: str, response: str):
        pass

    @abstractmethod
    def supports_model(self, model: str):
        pass

    @abstractmethod
    def evaluate_and_score(self, model: str, fact: str, answer: str):
        pass
