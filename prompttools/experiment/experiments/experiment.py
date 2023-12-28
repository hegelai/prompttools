# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Union
from operator import itemgetter
import base64
from collections import defaultdict
import itertools
import logging
from IPython import display
from tabulate import tabulate
import pandas as pd
import sentry_sdk
import os

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pymongo
except ImportError:
    pymongo = None

from prompttools.requests.request_queue import RequestQueue

# from ..widgets.feedback import FeedbackWidgetProvider
# from ..widgets.comparison import ComparisonWidgetProvider
from ..widgets.utility import is_interactive
from .error import PromptExperimentException
from ._utils import _get_dynamic_columns

pd.set_option("display.max_colwidth", 0)


class Experiment:
    r"""
    Base class for experiment. This should not be used directly, please use the subclasses instead.
    """

    completion_fn: Callable
    all_args: Dict

    def __init__(self):
        self.queue = RequestQueue()
        self.argument_combos: list[dict] = []
        self.full_df, self.partial_df, self.score_df = None, None, None
        self.image_experiment = False
        self._experiment_id = None
        self._revision_id = None
        try:
            if "SENTRY_OPT_OUT" not in os.environ:
                sentry_sdk.capture_message(f"Initializing {self.__class__.__name__}", "info")
        except Exception:
            pass
        # self.feedback_widget_provider = FeedbackWidgetProvider(
        #     self.completion_fn, self._aggregate_metric, self._get_human_eval_listener
        # )
        # self.comparison_widget_provider = ComparisonWidgetProvider(
        #     self.completion_fn,
        #     self._aggregate_comparison,
        #     self._get_comparison_listener,
        # )

    @classmethod
    def initialize(cls, test_parameters: dict[str, list], frozen_parameters: dict):
        r"""
        An alternate way to initialize an experiment by specifying which parameters should be tested
        and which ones should be frozen. If a parameter is not specified, the default value (if exists)
        for the parameter will be used.

        This allows you to easily initialize an experiment **without** wrapping every parameter in a list.

        Note:
            - For a given experiment, some parameters must be specified (e.g. the ``model`` parameter
              for OpenAI Chat Experiment). See the experiment's ``__init__`` method.
            - Each of ``test_parameters``'s values should be a ``list``, but not for ``frozen_parameters``.

        Args:
            test_parameters (dict[str, list]): parameters that are being tested. A list of multiple test values
                should be the value (e.g. ``{model: ["gpt-3.5-turbo", "gpt-4"], temperature: [0,0. 1.0]}``)
            frozen_parameters (dict): parameters that are intended to be frozen across different configuration.
                There is no need to wrap the value in a list. (e.g. ``{top_p: 1.0, presence_penalty: 0.0}``)

        Example:
            >>> from prompttools.experiment import OpenAIChatExperiment
            >>> test_parameters = {"model": ["gpt-3.5-turbo", "gpt-4"]}
            >>> messages = [{"role": "user", "content": "Who was the first president?"}]
            >>> frozen_parameters = {"top_p": 1.0, "messages": messages}
            >>> experiment = OpenAIChatExperiment.initialize(test_parameters, frozen_parameters)
        """
        frozen_parameters = {k: [v] for k, v in frozen_parameters.items()}
        return cls(**test_parameters, **frozen_parameters)

    @staticmethod
    def _is_chat():
        return False

    # def _get_human_eval_listener(self, i: int) -> Callable:
    #     def listener(change):
    #         self.score_df["feedback"][i] = change["new"]
    #
    #     return listener

    # def _get_comparison_listener(self, index: int) -> Callable:
    #     def listener(change):
    #         new_index = self.comparison_index_translation(index)
    #         self.score_df["comparison"][new_index] = change["new"]
    #
    #     return listener

    # def _aggregate_comparison(
    #     self,
    #     table: pd.DataFrame,
    #     agg_column: int = 0,
    #     is_average: bool = False,
    # ) -> Dict[str, int]:
    #     # TODO: This could be a group by
    #     prompt_scores = defaultdict(int)
    #     prompt_counts = defaultdict(int)
    #     for index, row in enumerate(table.iterrows()):
    #         key = str(row[agg_column])
    #         new_index = self.comparison_index_translation(index)
    #         prompt_scores[key] += self.score_df["comparison"][new_index]
    #         prompt_counts[key] += 1
    #     if is_average:
    #         for k, v in prompt_scores.items():
    #             prompt_scores[k] = v / prompt_counts[k]
    #     sorted_scores = dict(sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True))
    #     return sorted_scores

    def _aggregate_metric(
        self,
        table: pd.DataFrame,
        metric_name: str,
        agg_column: str,
        is_average: bool = False,
    ) -> Dict[str, int]:
        # TODO: This could be a group by

        prompt_scores = defaultdict(int)
        prompt_counts = defaultdict(int)
        for index, row in table.iterrows():
            key = str(row[agg_column])
            prompt_scores[key] += self.score_df[metric_name][index]
            prompt_counts[key] += 1
        if is_average:
            for k, v in prompt_scores.items():
                prompt_scores[k] = v / prompt_counts[k]
        sorted_scores = dict(sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores

    def prepare(self) -> None:
        r"""
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos = [dict(zip(self.all_args, val)) for val in itertools.product(*self.all_args.values())]

    def run(
        self,
        runs: int = 1,
        clear_previous_results: bool = False,
    ) -> None:
        r"""
        Create tuples of input and output for every possible combination of arguments.

        Note:
            If you overwrite this method in a subclass, make sure your method calls ``_construct_result_dfs``
            in order to save the results from your run as DataFrames. Then, they can later be used
            for evaluation, aggregation, and persistence.

        Args:
            runs (int): number of times to execute each possible combination of arguments, defaults to 1.
            clear_previous_results (bool): clear previous results before running
        """
        if not self.argument_combos:
            logging.info("Preparing first...")
            self.prepare()
        if clear_previous_results:
            self.queue = RequestQueue()
            self.full_df, self.partial_df, self.score_df = None, None, None
        original_n_results = len(self.queue.get_results()) if self.queue else 0
        for combo in self.argument_combos:
            for _ in range(runs):
                self.queue.enqueue(
                    self.completion_fn,
                    # We need to filter out defaults that are invalid JSON from the request
                    {k: v for k, v in combo.items() if (v is not None) and (v != float("inf"))},
                )
        number_of_new_results = len(self.queue.get_results()) - original_n_results
        if number_of_new_results == 0:
            logging.error("No results. Something went wrong.")
            raise PromptExperimentException
        results = self.queue.get_results()[-number_of_new_results:]
        input_args = self.queue.get_input_args()[-number_of_new_results:]
        latencies = self.queue.get_latencies()[-number_of_new_results:]
        self._construct_result_dfs(input_args, results, latencies)

    def _construct_result_dfs(
        self,
        input_args: list[dict[str, object]],
        results: list[dict[str, object]],
        latencies: list[float],
        extract_response_equal_full_result: bool = False,
        response_extractors: Optional[dict[str, Callable]] = None,
    ):
        r"""
        Takes in the input, results, and other metrics from the experiment's run, and construct a few DataFrames that
        contain all relevant data (i.e. input arguments, results, evaluation metrics).

        These DataFrames can later be used for evaluation, aggregation, or storing them for persistence.

        Note:
            - If your subclass of ``Experiment`` has a custom ``run`` method. You should consider overwriting this
              method. In particular, you likely would want to define how to extract the response from LLM's result
              and save that into ``response_df`` below. ChromaDBExperiment provides an example of this.
            - The inputs should all share the same length.

        Args:
             input_args (list[dict[str, object]]): list of dictionaries, where each of them is a set of
                 input argument that was passed into the model
             results (list[dict[str, object]]): list of responses from the model
             latencies (list[float]): list of latency measurements
             extract_response_equal_full_result (bool): if ``True``, ``result_df`` will only contain
                 the extracted response, lead to a simpler (but incomplete) columns of results.
             response_extractors (Optional[dict[str, Callable]]): Optional dictionary of response extractors,
                 defaults to ``None``, which will use the ``_extract_responses`` defined by the class.
                 The key of the dictionary will be the name of the resulting column, the value of the dictionary
                 will be an extractor function that accepts the response from the model and returns a value.
        """
        # `input_arg_df` contains all all input args
        input_arg_df = pd.DataFrame(input_args)
        # `dynamic_input_arg_df` contains input args that has more than one unique values
        dynamic_input_arg_df = _get_dynamic_columns(input_arg_df)

        # `response_df` contains the extracted response (often being the text response)
        if response_extractors is None:
            response_df = pd.DataFrame({"response": [self._extract_responses(result) for result in results]})
        else:
            res_dict = {}
            for col_name, extractor in response_extractors.items():
                res_dict[col_name] = [extractor(result) for result in results]
            response_df = pd.DataFrame(res_dict)
        # `result_df` contains everything returned by the completion function
        if extract_response_equal_full_result:
            result_df = response_df
        else:
            # Handle the case where `input_arg_df` has the same column names as `result_df`
            try:
                results = [r.model_dump() for r in results]  # For turing OpenAI response in to dict
            except Exception:
                pass
            result_df = pd.DataFrame(results)
            common_columns = set(input_arg_df.columns) & set(result_df.columns)
            result_df = result_df.add_prefix("response_") if common_columns else result_df
            result_df = pd.concat([response_df, result_df], axis=1)

        # `score_df` contains computed metrics (e.g. latency, evaluation metrics)
        new_score_df = pd.DataFrame({"latency": latencies})
        self.score_df = new_score_df if self.score_df is None else pd.concat([self.score_df, new_score_df])

        # `partial_df` contains some input arguments, extracted responses, and score
        new_partial_df = pd.concat([dynamic_input_arg_df, response_df, new_score_df], axis=1)
        self.partial_df = new_partial_df if self.partial_df is None else pd.concat([self.partial_df, new_partial_df])

        # `full_df` contains all input arguments, responses, and score
        new_full_df = pd.concat([input_arg_df, result_df, new_score_df], axis=1)
        self.full_df = new_full_df if self.full_df is None else pd.concat([self.full_df, new_full_df])

    def get_table(self, get_all_cols: bool = False) -> pd.DataFrame:
        r"""
        Get the DataFrame in one of two versions:
        1. ``get_all_cols = False`` - good for visualization. This contains dynamic (non-frozen) input arguments,
            the text response, and scores (e.g. latency and metrics generated from evaluation).
        2. ``get_all_cols = True`` - good for full result. This contains full data with all
            input arguments (including frozen ones), full model response (not just the text response), and scores.

        Args:
            get_all_cols (bool): defaults to ``False``. If ``True``, it will return the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
        """
        if self.full_df is None:
            logging.info("Running first...")
            self.run()
        if get_all_cols:
            return self.full_df
        else:
            return self.partial_df

    def visualize(self, get_all_cols: bool = False, pivot: bool = False, pivot_columns: list = []) -> None:
        r"""
        Visualize the DataFrame in one of two versions:
        1. ``get_all_cols = False`` - good for visualization. This contains dynamic (non-frozen) input arguments,
            the text response, and scores (e.g. latency and metrics generated from evaluation).
        2. ``get_all_cols = True`` - good for full result. This contains full data with all
            input arguments (including frozen ones), full model response (not just the text response), and scores.

        Args:
            get_all_cols (bool): defaults to ``False``. If ``True``, it will visualize the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
        """
        if pivot:
            table = self.pivot_table(pivot_columns, get_all_cols=get_all_cols)
        else:
            table = self.get_table(get_all_cols)
        if is_interactive() and self.image_experiment:
            # revert to color, display as image in notebook cells
            # for user friendly experience
            table["response"] = table["response"].map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR))
            table["response"] = table["response"].map(lambda x: self.cv2_image_to_base64(x))
            table["response"] = table["response"].apply(self.display_image_html)
            display.display(display.HTML(table.to_html(escape=False)))
        elif is_interactive():
            display.display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))

    def cv2_image_to_base64(self, image):
        if cv2 is None:
            raise ModuleNotFoundError(
                "Package `cv2` is required to be installed to use this experiment."
                "Please use `pip opencv-python` to install the package"
            )
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")

    def display_image_html(self, base64_string):
        return f'<img src="data:image/png;base64,{base64_string}" width="150" height="150"/>'

    def evaluate(
        self,
        metric_name: str,
        eval_fn: Callable,
        static_eval_fn_kwargs: dict = {},
        image_experiment: bool = False,
        **eval_fn_kwargs,
    ) -> None:
        """
        Using the given evaluation function that accepts a row of data, compute a new column with the evaluation
        result. Each row of data generally contain inputs, model response, and other previously computed metrics.

        Args:
            metric_name (str): name of the metric being computed
            eval_fn (Callable): an evaluation function that takes in a row from pd.DataFrame
                and optional keyword arguments
            static_eval_fn_kwargs (dict): keyword args for ``eval_fn`` that are consistent for all rows
            eval_fn_kwargs (Optional[list]): keyword args for ``eval_fn`` that may be different for each row.
                Each value entered here should be a list, and the length of the list should be
                the same as the number of responses in the experiment's result. The ``i``th element of the list will be
                passed to the evaluation function to evaluate the ``i``th row.

        Example:
            >>> from prompttools.utils import validate_json_response
            >>> experiment.evaluate("is_json", validate_json_response,
            >>>                     static_eval_fn_kwargs={"response_column_name": "response"})
        """
        self.image_experiment = image_experiment
        if metric_name in self.score_df.columns:
            logging.warning(metric_name + " is already present, skipping.")
            return
        res = []
        table = self.get_table(get_all_cols=True)
        for i, row in table.iterrows():
            curr_kwargs = static_eval_fn_kwargs.copy()
            for k, v in eval_fn_kwargs.items():
                curr_kwargs[k] = v[i]
            res.append(eval_fn(row, **curr_kwargs))
        self._update_score(metric_name, res)

    def _update_score(self, metric_name: str, res) -> None:
        self.score_df[metric_name] = res
        self.partial_df[metric_name] = res
        self.full_df[metric_name] = res

    def pivot_table(
        self, pivot_columns: List[str], response_value_name: Optional[str] = None, get_all_cols: bool = False
    ) -> pd.DataFrame:
        """
        Returns a pivoted DataFrame.

        Args:
            pivot_columns (List[str]): two column names (first for pivot row, second for pivot column)
                that serve as indices the pivot table
            response_value_name (Optional[str]): name of the column to aggregate.
            get_all_cols (bool): defaults to ``False``. If ``True``, it will visualize the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
        """
        df = self.get_table(get_all_cols)
        pivot_df = pd.pivot_table(
            df,
            values=response_value_name,
            index=[pivot_columns[1]],
            columns=[pivot_columns[0]],
            aggfunc=lambda x: x.iloc[0],
        )
        return pivot_df

    # def gather_feedback(self, pivot_data: Dict[str, object], pivot_columns: List[str]) -> None:
    #     """
    #     This method creates a table to gather human feedback from a notebook interface.
    #
    #     Args:
    #         pivot_data (Dict[str, object]): dictionary that contains additional data or metadata related to the input
    #         pivot_columns (List[str]): two column names (first for pivot row, second for pivot column)
    #             that serve as indices the pivot table
    #     """
    #     if not is_interactive():
    #         logging.warning("This method only works in notebooks.")
    #         return
    #     table = self.get_table(get_all_cols=True)
    #     self.score_df["feedback"] = [1] * len(table.index))
    #     self.feedback_widget_provider.set_pivot_columns(pivot_columns)
    #     items = self.feedback_widget_provider.get_header_widgets()
    #     for row in table.iterrows():
    #         items += self.feedback_widget_provider.get_row_widgets(*row)
    #     items += self.feedback_widget_provider.get_footer_widgets(table)
    #     self.feedback_widget_provider.display(items)

    # def compare(self, primary_model: str, pivot_columns: List[str]) -> None:
    #     """
    #     This method creates a table to gather human feedback from a notebook interface.
    #     """
    #     if not is_interactive():
    #         logging.warning("This method only works in notebooks.")
    #         return
    #     table = self.get_table(pivot_data={}, pivot_columns=pivot_columns, pivot=True)
    #     self.score_df["comparison"] = [1] * len(table.index))
    #     self.comparison_index_translation = lambda i: i * len(table.columns)
    #     self.comparison_widget_provider.set_models(table.columns)
    #     items = self.comparison_widget_provider.get_header_widgets()
    #     for index, row in enumerate(table.iterrows()):
    #         items += self.comparison_widget_provider.get_row_widgets(index, row[1])
    #     items += self.comparison_widget_provider.get_footer_widgets(table)
    #     self.comparison_widget_provider.display(items)
    def aggregate(self, metric_name, column_name, is_average=False):
        r"""
        Aggregates a metric for a given column and displays to the user.

         Args:
            metric_name (str): metric to aggregate
            column_name (str): column to base the aggregation on
            is_average (bool): if ``True``, compute the average for the metric, else compute the total
        """
        if self.score_df is None or metric_name not in self.score_df.columns:
            logging.warning("Can't find " + metric_name + " in scores. Did you run `evaluate`?")
            return
        table = self.get_table(get_all_cols=False)
        sorted_scores = self._aggregate_metric(table, metric_name, column_name, is_average)
        if is_interactive():
            import matplotlib.pyplot as plt
            import os

            # Import style file, assumes same dir as experiment.py
            style_path = os.path.join(os.path.dirname(__file__), "style.mplstyle")
            plt.style.use(style_path)

            # Define the custom colors
            custom_colors = [
                "black", "#771541", "#EB8F4C", "#594F3B", "#A8B7AB", "#9C92A3"
            ]

            plt.ylabel("Latency (s)")

            # Cycle through the custom colors when creating the bars
            for i, (label, value) in enumerate(sorted_scores.items()):
                plt.bar(i, value, align="center", color=custom_colors[i % len(custom_colors)])

            plt.xticks(range(len(sorted_scores)), list(sorted_scores.keys()))
            plt.show()

    def rank(
        self,
        metric_name: str,
        is_average: bool,
        agg_column: str,
        get_all_cols: bool = False,
    ) -> Dict[str, int]:
        """
        Using pivot data, groups the data by the first pivot column to
        get scores, and sorts descending. For example, using pivot data of
        (prompt_template, user_input), a metric of latency, and is_average=True,
        we rank prompt templates by their average latency in the test set.

        Args:
            metric_name (str): metric to aggregate over
            is_average (bool): if ``True``, compute the average for the metric, else compute the total
            agg_column (str): column to aggregate over
            get_all_cols (bool): defaults to ``False``. If ``True``, it will return the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
        """
        if self.score_df is None or metric_name not in self.score_df.columns:
            logging.warning("Can't find " + metric_name + " in scores. Did you run `evaluate`?")
            return
        table = self.get_table(get_all_cols=get_all_cols)
        sorted_scores = self._aggregate_metric(table, metric_name, agg_column, is_average)
        return sorted_scores

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        raise NotImplementedError("This should be implemented by a subclass of `Experiment`.")

    def to_csv(
        self,
        path: str,
        get_all_cols: bool = True,
        **kwargs,
    ):
        r"""
        Export the results to a CSV file. If the experiment has not been executed, it will run.

        Args:
            path (str): path/buffer to write the CSV output
            get_all_cols (bool): defaults to ``False``. If ``True``, it will return the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
            **kwargs: optional arguments passed to ``pd.DataFrame.to_csv()``
        """
        table = self.get_table(get_all_cols=get_all_cols)
        table.to_csv(path, **kwargs)

    def to_pandas_df(self, get_all_cols: bool = True, from_streamlit: bool = False):
        r"""
        Return the results as a ``pandas.DataFrame``. If the experiment has not been executed, it will run.

        Args:
            get_all_cols (bool): defaults to ``False``. If ``True``, it will return the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
        """
        if from_streamlit:
            self.run()
        return self.get_table(get_all_cols=get_all_cols)

    def to_json(
        self,
        path: Optional[str] = None,
        get_all_cols: bool = True,
        **kwargs,
    ):
        r"""
        Export the results to a JSON file. If the experiment has not been executed, it will run.

        Args:
            path (Optional[str]): path/buffer to write the JSON output, defaults to ``None`` which returns
                the JSON as a `dict`
            get_all_cols (bool): defaults to ``False``. If ``True``, it will return the full data with all
                input arguments (including frozen ones), full model response (not just the text response), and scores.
            **kwargs: optional arguments passed to ``pd.DataFrame.to_json()``
        """
        table = self.get_table(get_all_cols=get_all_cols)
        if path is None:
            return table.to_json(**kwargs)
        else:
            return table.to_json(path, **kwargs)

    def to_lora_json(
        self,
        instruction_extract: Union[str, Callable],
        input_extract: Union[str, Callable],
        output_extract: Union[str, Callable],
        path: Optional[str] = None,
        **kwargs,
    ):
        r"""
        Export the results to a LoRA-format JSON file for fine-tuning.
        If the experiment has not been executed, it will run.

        Args:
            instruction_extract (Union[str, Callable]): column name, or an extractor function that will accept a row
                of the result table and return a value assigned to ``"instruction"`` entry in the JSON file
            input_extract (Union[str, Callable]): column name, or an extractor function that will accept a row
                of the result table and return a value assigned to ``"input"`` entry in the JSON file
            output_extract (Union[str, Callable]): column name, or an extractor function that will accept a row
                of the result table and return a value assigned to ``"output"`` entry in the JSON file
            path (Optional[str]): path/buffer to write the JSON output, defaults to ``None`` which returns
                the JSON as a `dict`
            **kwargs: optional arguments passed to ``pd.DataFrame.to_json()``
        """
        if isinstance(instruction_extract, str):
            instruction_extract = itemgetter(instruction_extract)
        if isinstance(input_extract, str):
            input_extract = itemgetter(input_extract)
        if isinstance(output_extract, str):
            output_extract = itemgetter(output_extract)
        df = self.to_pandas_df(get_all_cols=True)
        extracted_data = df.apply(
            lambda row: {
                "instruction": instruction_extract(row),
                "input": input_extract(row),
                "output": output_extract(row),
            },
            axis=1,
        )
        if "orient" not in kwargs:
            kwargs["orient"] = "records"
        if "indent" not in kwargs:
            kwargs["indent"] = 2

        if path:
            extracted_data.to_json(path, **kwargs)
        else:
            return extracted_data.to_json(**kwargs)

    # TODO: Add MongoDB local instruction (maybe include docker)
    def to_mongo_db(self, mongo_uri: str, database_name: str, collection_name: str) -> None:
        r"""
        Insert the results of the experiment into MongoDB for persistence.

        Note:
            - You need to install the ``pymongo`` package to use this method.
            - You need to run a local or remote instance of MongoDB in order to store the data.

        Args:
            mongo_uri (str): a connection string to the target MongoDB
            database_name (str): name of the MongoDB database
            collection_name (str): name of the MongoDB collection
        """
        if pymongo is None:
            raise ModuleNotFoundError(
                "Package `pymongo` is required to be installed to use this method."
                "Please use `pip install pymongo` to install the package"
            )
        if self.full_df is None:
            logging.info("Running first...")
            self.run()
        client = pymongo.MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]
        collection.insert_many(self.full_df.to_dict("records"))
        logging.info(f"Inserted results in {database_name}'s collection {collection_name}.")
        client.close()

    def to_markdown(self):
        markdown = self.to_pandas_df().to_markdown()
        return markdown

    def _get_model_names(self):
        pass

    def _get_prompts(self):
        pass

    def _get_state(self):
        raise NotImplementedError("Should be implemented by specific harness class.")

    @classmethod
    def _load_state(cls, state, experiment_id: str, revision_id: str, experiment_type_str: str):
        raise NotImplementedError("Should be implemented by specific harness class.")
