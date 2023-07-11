from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from prompttools.structures.prompt import Response
from prompttools.utils import similarity
from typing import Any, List


class MaxLLM:
    """
    Get best response from set of LLMs and params.
    """
    def __init__(
        self,
        hf_repo_ids: List[str] = [],
        temperatures: List[float] = [],
        max_lengths: List[int] = [],
        template: str = """Question: {question}
        Answer: """,
        question: str = "",
        expected: str = "",
    ):
        self.hf_repo_ids = hf_repo_ids
        self.temperatures = temperatures
        self.max_lengths = max_lengths
        self.template = template
        self.question = question
        self.expected = expected
        self.responses: List[Any] = []
    
    def best_response(self):
        if not self.responses:
            return None
        return self.responses[0]

    def top_n_responses(self, n=1):
        if not self.responses:
            return None
        return self.responses[:n]

    def run(self) -> None:
        prompt = PromptTemplate(template=self.template, 
                                input_variables=["question"])
        for repo_id in self.hf_repo_ids:
            for temperature in self.temperatures:
                for max_length in self.max_lengths:
                    llm = HuggingFaceHub(
                            repo_id=repo_id, 
                            model_kwargs={"temperature": temperature, "max_length": max_length}
                        )
                    llm_chain = LLMChain(prompt=prompt, llm=llm)
                    response = llm_chain.run(self.question)
                    score = similarity.compute(doc1=response, doc2=self.expected, use_chroma=False)
                    self.responses.append(
                        Response(
                            repo_id=repo_id,
                            temperature=temperature,
                            max_length=max_length,
                            response=response,
                            score=score,
                        )
                    )
        self.responses = sorted(self.responses, key=lambda x: x.score, reverse=True)

