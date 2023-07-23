# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List
import pandas as pd
from IPython import display
import ipywidgets as widgets


class ComparisonWidgetProvider:
    r"""
    Provides functionality for widgets to compare models. This includes
    displaying widgets, and recording evaluations in the experiment.
    """

    def __init__(self, completion_fn, agg_fn, eval_listener_fn):
        self.completion_fn = completion_fn
        self.agg_fn = agg_fn
        self.eval_listener_fn = eval_listener_fn

    def _get_comparison_submission_listener(self, table: pd.DataFrame, models: List[str]) -> Callable:
        def on_click(b):
            sorted_scores = self.agg_fn(table, 0)
            data = {
                models[0]: sorted_scores.keys(),
                "feedback": sorted_scores.values(),
            }
            df = pd.DataFrame(data)
            display.display(df)

        return on_click

    def set_models(self, models: List[str]) -> None:
        self.models = models
        self.row_len = 2 + len(self.models)

    def get_header_widgets(self) -> List[object]:
        return [widgets.Label("Input")] + [widgets.Label(model) for model in self.models] + [widgets.Label("Feedback")]

    def get_row_widgets(self, index, row):
        items = [widgets.HTML(value="<style>p{word-wrap: break-word}</style> <p>" + row.name + " </p>")]
        items += [
            widgets.HTML(value="<style>p{word-wrap: break-word}</style> <p>" + row[model] + " </p>")
            for model in self.models
        ]
        feedback_dropdown = widgets.Dropdown(
            options=[("\U0001F44D", 1), ("\U0001F44E", 0)],
            value=1,
            layout={"width": "50px"},
        )
        feedback_dropdown.observe(self.eval_listener_fn(index), names="value")
        items += [feedback_dropdown]
        return items

    def get_footer_widgets(self, table):
        submit_button = widgets.Button(
            description="Submit",
            disabled=False,
            button_style="success",
            tooltip="Submit",
        )
        submit_button.on_click(self._get_comparison_submission_listener(table, self.models))
        return [widgets.Label("")] * (self.row_len - 1) + [submit_button]

    def display(self, items):
        row_len = 2 + len(self.models)
        grid = widgets.GridBox(
            items,
            layout=widgets.Layout(grid_template_columns="repeat(" + str(row_len) + ", 230px)"),
        )
        display.display(grid)
