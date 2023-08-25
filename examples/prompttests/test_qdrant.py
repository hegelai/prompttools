import hashlib
from typing import List

from qdrant_client import QdrantClient

from prompttools.experiment import QdrantExperiment


def embedding_function(text: str) -> List[float]:
    r"""
    Create vector embedding from text. This is a dummy function for testing purposes
    and returns a vector of 16 floats.

    Args:
        text (str): Text to be vectorized
    Returns:
        List[float]: Vector embedding of the text
    """
    import numpy as np
    import struct

    vectorized_text = np.abs(
        np.array(struct.unpack(">ffffffffffffffff", hashlib.sha512(text.encode("utf-8")).digest()))
    )
    normalized_vector = vectorized_text / np.linalg.norm(vectorized_text)
    return normalized_vector.tolist()


test_parameters = {
    "collection_params": {
        "vectors_config__distance": ["Cosine", "Euclid", "Dot"],
        "hnsw_config__m": [16, 32, 64, 128],
    },
    "query_params": {
        "search_params__hnsw_ef": [1, 16, 32, 64, 128],
        "search_params__exact": [True, False],
    },
}
frozen_parameters = {
    # Run Qdrant server locally with:
    # docker run -p "6333:6333" -p "6334:6334" qdrant/qdrant:v1.4.0
    "client": QdrantClient("http://localhost:6333"),
    "collection_name": "test_collection",
    "embedding_fn": embedding_function,
    "vector_size": 16,
    "documents": ["test document 1", "test document 2"],
    "queries": ["test query 1", "test query 2"],
}
experiment = QdrantExperiment.initialize(test_parameters=test_parameters, frozen_parameters=frozen_parameters)
experiment.run()

print(experiment.get_table(True))
