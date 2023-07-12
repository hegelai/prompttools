# Example comparing LLM responses

from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from prompttools.utils import similarity


def test_response_hugging_face() -> None:
    """
    Model repos are via Hugging Face. So pytorch_cos_sim is
    used for computing distance.
    """
    # TODO: refactor once HuggingFaceHubExperiment is done
    # Principle still works

    hf_repo_ids = ["google/flan-t5-xxl", "bigscience/bloom"]
    temperatures = [0.1, 0.5]
    max_lengths = [17, 32]
    template = """Question: {question}
            Answer: """
    question = "Who was the first president?"
    expected = "George Washington"
    prompt = PromptTemplate(template=template, input_variables=["question"])

    tests = []

    for repo_id in hf_repo_ids:
        for temperature in temperatures:
            for max_length in max_lengths:
                tests.append(
                    {
                        "repo_id": repo_id,
                        "temperature": temperature,
                        "max_length": max_length,
                    }
                )
    
    for test in tests:
        llm = HuggingFaceHub(
            repo_id=test["repo_id"],
            model_kwargs={"temperature": test["temperature"],
                        "max_length": test["max_length"]}
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run(question)
        score = similarity.compute(doc1=response, doc2=expected, use_chroma=False)
        assert score > 0.41

    
