import boto3


class EvaluationBase:
    def __init__(self):
        pass


class RuleEvaluation(EvaluationBase):
    def __init__(self):
        pass

    def run(self):
        pass


class SemanticEvaluation(EvaluationBase):
    def __init__(self):
        pass

    def run(self):
        pass


class AutoEvaluation(EvaluationBase):
    def __init__(self):
        pass

    def run(self):
        pass


# https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkRequester/GetStartedMturk.html
class MTurkEvaluation(EvaluationBase):
    def __init__(self):
        self.client: boto3.MTurk.Client = boto3.client("mturk")

    def run(self):
        response = self.client.create_hit()

    def check(self):
        pass

    def get(self):
        pass
