# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Type, Callable
import jinja2
from .harness import ExperimentationHarness, Experiment
import copy


DOC_PROMPT_TEMPLATE = r"""
Given these documents:{{documents}}

{{prompt}}
"""


def _doc_list_to_str(documents: list[str]) -> str:
    res = ""
    for d in documents:
        res += "\n"
        res += d
    return res


def _generate_doc_prompt(documents: list[str], prompt: str):
    environment = jinja2.Environment()
    template = environment.from_string(DOC_PROMPT_TEMPLATE)
    doc_str = _doc_list_to_str(documents)

    doc_prompt = template.render(
        {
            "documents": doc_str,
            "prompt": prompt,
        }
    )
    return doc_prompt


class RetrievalAugmentedGenerationExperimentationHarness(ExperimentationHarness):
    r"""
    An experimentation harness used to test the Retrieval-Augmented Generation process, which
    involves a vector DB and a LLM at the same time.

    Args:
        vector_db_experiment (Experiment): An initialized vector DB experiment.
        llm_experiment_cls (Type[Experiment]): The experiment constructor that you would like to execute
            within the harness (e.g. ``prompttools.experiment.OpenAICompletionExperiment``)
        llm_arguments (dict[str, list]): Dictionary of arguments for the LLM.
        extract_document_fn (Callable): A function, when given a row of results from the vector DB experiment,
            extract the relevant documents (``list[str]``) that will be inserted into the template.
        prompt_template (str): A ``jinja``-styled templates, where documents and prompt will be inserted.
    """

    def __init__(
        self,
        vector_db_experiment: Experiment,
        llm_experiment_cls: Type[Experiment],
        llm_arguments: dict,
        extract_document_fn: Callable,
        prompt_template: str = DOC_PROMPT_TEMPLATE,
    ):
        self.vector_db_experiment = vector_db_experiment
        self.llm_experiment_cls: Type[Experiment] = llm_experiment_cls
        self.experiment: Optional[Experiment] = None
        self.llm_arguments = copy.copy(llm_arguments)
        self.extract_document_fn = extract_document_fn
        self.prompt_templates = prompt_template

    def run(self) -> None:
        self.vector_db_experiment.run()
        document_lists: list[list[str]] = []
        # latencies = []  # TODO: Include latency results
        # Extract documents from the result of
        for i, row in self.vector_db_experiment.full_df.iterrows():
            document_lists.append(self.extract_document_fn(row))
            # latencies.append(row["latencies"])
        # Put documents into prompt template
        augmented_prompts = []
        for doc in document_lists:
            for prompt in self.llm_arguments["prompt"]:  # TODO: Make this work for chat, check `experiment._is_chat()`
                augmented_prompts.append(_generate_doc_prompt(doc, prompt))

        # Pass documents into LLM
        self.llm_arguments["prompt"]: list[str] = augmented_prompts
        self.experiment = self.llm_experiment_cls(**self.llm_arguments)

        # Run the LLM experiment
        self.experiment.run()
        # TODO: Need to clean up results, ideally display these columns ("top doc ids", "prompt")
        #       Right now it is displaying the fully augmented prompt for every row, which take up too much space
        #       Permutation ordering (doc_list1, prompt, then other arguments)
        self.partial_df = self.experiment.partial_df
        self.full_df = self.experiment.full_df

    def visualize(self) -> None:
        if self.experiment is None:
            self.run()
        self.experiment.visualize()
