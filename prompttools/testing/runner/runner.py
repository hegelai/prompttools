from collections import defaultdict


class PromptTestRunner:
    def __init__(self):
        self.ran = defaultdict(bool)
        self.harnesses = dict()

    def run(self, *args, **kwargs):
        key = str(args)
        if self.ran[key]:
            return key
        self.harnesses[key] = self._get_harness(*args, **kwargs)
        self.harnesses[key].prepare()
        self.harnesses[key].run()
        self.ran[key] = True
        return key

    def evaluate(self, key, metric_name, eval_fn, use_input_pairs):
        self.harnesses[key].evaluate(metric_name, eval_fn, use_input_pairs)

    def rank(self, key, metric_name, is_average):
        return self.harnesses[key].rank(metric_name, is_average)
