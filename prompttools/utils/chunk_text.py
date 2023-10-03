# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


def chunk_text(text: str, max_chunk_length: int) -> list[str]:
    r"""
    Given a long string paragraph of text and a chunk max length, returns chunks of texts where each chunk's
    length is smaller than the max length, without breaking up individual words (separated by space).

    Args:
        text (str): source text to be chunked
        max_chunk_length (int): maximum length of a chunk
    """

    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chunk_length:
            if current_chunk:
                current_chunk += " "
            current_chunk += word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
