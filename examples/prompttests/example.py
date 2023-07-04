import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import chromadb
from typing import Dict, Tuple
import prompttools.testing.prompttest as prompttest
from prompttools.testing.threshold_type import ThresholdType


def extract_responses(output) -> str:
    return [choice["text"] for choice in output["choices"]]


@prompttest.prompttest(
    model_name="gpt-3.5-turbo",
    metric_name="did_echo",
    threshold=2,
    threshold_type=ThresholdType.MAXIMUM,
    prompt_template="Echo the following input: {{input}}",
    user_input=[{"input": "This is a test"}],
    use_input_pairs=True,
    model_arguments={"temperature": 0.9},
)
def check_similarity(
    input_pair: Tuple[str, Dict[str, str]], results: Dict, metadata: Dict
) -> float:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_collection")
    collection.add(
        documents=[dict(input_pair[1])["input"]], metadatas=[metadata], ids=["id1"]
    )
    query_results = collection.query(
        query_texts=extract_responses(results), n_results=1
    )
    chroma_client.delete_collection("test_collection")
    return min(query_results["distances"])[0]


if __name__ == "__main__":
    prompttest.main()
