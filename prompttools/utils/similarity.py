# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

r"""
Use a list to optionally hold a reference to the embedding model and client,
allowing for lazy initialization.
"""
from typing import Dict, List

EMBEDDING_MODEL = []  #
CHROMA_CLIENT = []


def _get_embedding_model():
    if len(EMBEDDING_MODEL) == 0:
        from sentence_transformers import SentenceTransformer

        EMBEDDING_MODEL.append(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    return EMBEDDING_MODEL[0]


def _get_chroma_client():
    if len(CHROMA_CLIENT) == 0:
        import chromadb

        CHROMA_CLIENT.append(chromadb.Client())
    return CHROMA_CLIENT[0]


def _from_huggingface(doc1, doc2):
    model = _get_embedding_model()
    embedding_1 = model.encode(doc1, convert_to_tensor=True)
    embedding_2 = model.encode(doc2, convert_to_tensor=True)
    from sentence_transformers.util import pytorch_cos_sim

    return pytorch_cos_sim(embedding_1, embedding_2).item()


def _from_chroma(doc1, doc2):
    chroma_client = _get_chroma_client()
    collection = chroma_client.create_collection(name="test_collection")
    collection.add(documents=[doc1], ids=["id1"])
    query_results = collection.query(query_texts=doc2, n_results=1)
    chroma_client.delete_collection("test_collection")
    return query_results["distances"][0][0] / 2


def compute(doc1, doc2, use_chroma=True):
    r"""
    Computes the semantic similarity between two documents, using either ChromaDB
    or HuggingFace sentence_transformers.

    Args:
        doc1 (str): The first document.
        doc2 (str): The second document.
        use_chroma (bool): Indicates whether or not to use Chroma.
            If false, uses HuggingFace sentence_transformers.
    """
    if use_chroma:
        return _from_chroma(doc1, doc2)
    else:
        return _from_huggingface(doc1, doc2)


def evaluate(prompt: str, response: str, metadata: Dict, expected: str) -> float:
    r"""
    A simple test that checks semantic similarity between the user input
    and the model's text responses.
    """
    return compute(expected, response)
