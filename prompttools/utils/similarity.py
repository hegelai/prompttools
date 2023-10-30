# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

r"""
Use a list to optionally hold a reference to the embedding model and client,
allowing for lazy initialization.
"""
from typing import Dict
import pandas.core.series
import logging

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from skimage.metrics import structural_similarity as skimage_structural_similarity
except ImportError:
    skimage_structural_similarity = None


EMBEDDING_MODEL = []
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


def compute(doc1, doc2, use_chroma=False):
    r"""
    Computes the semantic similarity between two documents, using either ChromaDB
    or HuggingFace sentence_transformers.

    Args:
        doc1 (str): The first document.
        doc2 (str): The second document.
        use_chroma (bool): Indicates whether or not to use Chroma.
            If ``False``, uses HuggingFace ``sentence_transformers``.
    """
    if use_chroma:
        return _from_chroma(doc1, doc2)
    else:
        return _from_huggingface(doc1, doc2)


def evaluate(prompt: str, response: str, metadata: Dict, expected: str) -> float:
    r"""
    A simple test that checks semantic similarity between the expected response (provided by the user)
    and the model's text responses.

    Args:
        prompt (str): Not used.
        response (str): the response string that will be compared against
        metadata (dict): Not used.
        expected (str): the expected response
    """
    return compute(expected, response)


def structural_similarity(
    row: pandas.core.series.Series, expected: str, response_column_name: str = "response"
) -> float:
    r"""
    Compute the structural similarity index measure (SSIM) between two images.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        expected (str): the column name of the expected image responses in each row
        response_column_name (str): the column name that contains the model's response, defaults to ``"response"``
    """
    if cv2 is None:
        raise ModuleNotFoundError(
            "Package `cv2` is required to be installed to use this evaluation method."
            "Please use `pip install opencv-python` to install the package"
        )
    if skimage_structural_similarity is None:
        raise ModuleNotFoundError(
            "Package `skimage` is required to be installed to use this evaluation method."
            "Please use `pip install scikit-image` to install the package"
        )
    if len(expected) == 1:
        logging.warning("Expected should be a list of strings. You may have passed in a single string.")
    expected_img = cv2.imread(expected)
    expected_img = cv2.cvtColor(expected_img, cv2.COLOR_BGR2GRAY)
    score, _ = skimage_structural_similarity(row[response_column_name], expected_img, full=True)
    return score


def semantic_similarity(row: pandas.core.series.Series, expected: str, response_column_name: str = "response") -> float:
    r"""
    A simple test that checks semantic similarity between the expected response (provided by the user)
    and the model's text responses.

    Args:
        row (pandas.core.series.Series): A row of data from the full DataFrame (including input, model response, other
            metrics, etc).
        expected (str): the expected responses for each row in the column
        response_column_name (str): name of the column that contains the model's response, defaults to ``"response"``
    """
    if len(expected) == 1:
        logging.warn("Expected should be a list of strings." + "You may have passed in a single string")
    return compute(expected, row[response_column_name])
