from typing import Callable, Dict, Optional
from prompttools.template import Template


# list of parameters, prompts, endpoints to run over
# way to create specs using cartesian product, but 
# must be aware that not all combos are valid bc of API difference 

# Defines a single experiment to run
class Experiment:
    def __init__(self, completion_fn: Callable,
                       prompt: Optional[str] = None,
                       messages: Optional[Dict] = None,
                       template: Optional[Template] = None,
                       model_args: Optional[Dict] = None,
                       inputs: Optional[Dict] = None):
        if prompt:
            assert messages is None and template is None
        
        if messages:
            assert prompt is None and template is None

        if template:
            assert prompt is None and messages is None

    def run(self):
        pass