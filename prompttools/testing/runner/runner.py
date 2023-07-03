class PromptTestRunner:
    def __init__(self):
        self.ran = False

    def run(self, *args, **kwargs):
        if self.ran:
            return
        self.harness = self._get_harness(*args, **kwargs)
        self.harness.prepare()
        self.harness.run()
        self.ran = True

    def evaluate(self, metric_name, eval_fn):
        self.harness.evaluate(metric_name, eval_fn)

    def rank(self, metric_name, is_average):
        return self.harness.rank(metric_name, is_average)
