from prompttools.experiment import HuggingFaceHubExperiment
from prompttools.utils import similarity


def test_top_n_responses():
    hf_repo_ids = ["google/flan-t5-xxl", "bigscience/bloom"]
    max_lengths = [17, 32]
    temperatures = [0.1, 0.5]
    input_variables = ["question"]
    template = """Question: {question}
            Answer: """

    question = "Who was the very first president of America"
    expected = "President George Washington"

    experiment = HuggingFaceHubExperiment(
        repo_id=hf_repo_ids,
        max_length=max_lengths,
        temperature=temperatures,
        input_variables=input_variables,
        template=template,
        question=question,
        expected=expected,
    )

    experiment.run()

    responses = experiment.top_n_responses(eval_fn=similarity.compute, n=3)
    for resp in responses:
        assert resp["score"] > 0.50