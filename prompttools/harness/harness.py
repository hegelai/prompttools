class ExperimentationHarness:
    @staticmethod
    def _prepare_arguments(arguments):
        return {name: [arg] for name, arg in arguments}

    def prepare(self):
        self.experiment.prepare()

    def run(self):
        self.experiment.run()

    def evaluate(self, eval_fn):
        self.experiment.evaluate(eval_fn)

    def visualize(self):
        self.experiment.visualize()
