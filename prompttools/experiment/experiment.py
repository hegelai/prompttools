import itertools
import logging
from utils import is_interactive
from IPython import display
from tabulate import tabulate


class Experiment:
    def prepare(self):
        """
        Creates argument combinations by taking the cartesian product of all inputs.
        """
        self.argument_combos = []
        for combo in itertools.product(*self.all_args):
            self.argument_combos.append(combo)

    def visualize(self):
        """
        Creates and shows a table using the results produced.
        """
        if not self.scores:
            logging.warning("Please run `evaluate` first.")
        table = self.get_table()
        if is_interactive():
            display(table)
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(tabulate(table, headers="keys", tablefmt="psql"))
