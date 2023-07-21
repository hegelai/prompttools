# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List
import pandas as pd
from IPython import display
import ipywidgets as widgets


class FeedbackWidgetProvider:
    r"""
    Provides functionality for widgets to evaluate models. This includes
    displaying widgets, and recording evaluations in the experiment.
    """

    def __init__(self, completion_fn, agg_fn, eval_listener_fn):
        self.completion_fn = completion_fn
        self.agg_fn = agg_fn
        self.eval_listener_fn = eval_listener_fn

    def _get_feedback_submission_listener(self, table: pd.DataFrame, pivot_columns: List[str]) -> Callable:
        def on_click(b):
            sorted_scores = self.agg_fn(table, "feedback", pivot_columns[0])
            data = {
                pivot_columns[0]: sorted_scores.keys(),
                "feedback": sorted_scores.values(),
            }
            df = pd.DataFrame(data)
            display.display(df)

        return on_click

    def set_pivot_columns(self, pivot_columns: List[str]) -> None:
        self.pivot_columns = pivot_columns

    def get_header_widgets(self) -> List[object]:
        return [
            widgets.Label(self.pivot_columns[0]),
            widgets.Label(self.pivot_columns[1]),
            widgets.Label("response(s)"),
            widgets.Label("Feedback"),
        ]

    def get_row_widgets(self, index, row):
        items = [
            widgets.HTML(value="<style>p{word-wrap: break-word}</style> <p>" + row[self.pivot_columns[0]] + " </p>"),
            widgets.HTML(value="<style>p{word-wrap: break-word}</style> <p>" + row[self.pivot_columns[1]] + " </p>"),
            widgets.HTML(value="<style>p{word-wrap: break-word}</style> <p>" + ", ".join(row["response(s)"]) + " </p>"),
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
        submit_button.on_click(self._get_feedback_submission_listener(table, self.pivot_columns))
        return [
            widgets.Label(""),
            widgets.Label(""),
            widgets.Label(""),
            submit_button,
        ]

    def display(self, items):
        grid = widgets.GridBox(
            items,
            layout=widgets.Layout(grid_template_columns="repeat(4, 230px)"),
        )
        display.display(grid)
