class ExperimentationHarness:
    @staticmethod
    def _prepare_arguments(arguments):
        return {name: [arg] for name, arg in arguments}

    def prepare(self):
        self.experiment.prepare()

    def run(self):
        self.experiment.run()

    def evaluate(self, metric_name, eval_fn):
        self.experiment.evaluate(metric_name, eval_fn)

    def gather_feedback(self):
        self.experiment.gather_feedback(self.input_pairs_dict, self.PIVOT_COLUMNS)

    def visualize(self, pivot=False):
        if pivot:
            self.experiment.visualize(self.input_pairs_dict, self.PIVOT_COLUMNS)
        else:
            self.experiment.visualize()

    def rank(self, metric_name, is_average=False):
        return self.experiment.rank(
            self.input_pairs_dict, self.PIVOT_COLUMNS, metric_name, is_average
        )
