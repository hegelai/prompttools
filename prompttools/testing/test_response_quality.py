from prompttools.structures.prompt import PromptTest
from prompttools.structures.config import get_model_params_config

from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from prompttools.utils import similarity


def test_response_hugging_face() -> None:
    """
    Model repos are via Hugging Face. So pytorch_cos_sim is
    used for computing distance.
    """

    model_params = get_model_params_config()

    question = model_params.question
    template = """Question: {question}

    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    expected = model_params.expected
    hf_repo_ids = model_params.hf_repo_ids
    temperatures = model_params.temperatures
    max_lengths = model_params.max_lengths # 17 for deterministic response: George Washington

    tests = []

    for repo_id in hf_repo_ids:
        for temperature in temperatures:
            for max_length in max_lengths:
                tests.append(
                    PromptTest(
                        repo_id=repo_id,
                        temperature=temperature,
                        max_length=max_length
                    )
                )
    
    for test in tests:
        llm = HuggingFaceHub(
            repo_id=test.repo_id,
            model_kwargs={"temperature": test.temperature,
                        "max_length": test.max_length}
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run(question)
        score = similarity.compute(doc1=response, doc2=expected, use_chroma=False)
        assert score > 0.41

    
