# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Tuple
import prompttools.prompttest as prompttest
from prompttools.utils import similarity
from prompttools.experiment import ChromaDBExperiment

EXPECTED = {"Who was the first president of the USA?": "George Washington"}

if not (("CHROMADB_API_TOKEN" in os.environ) or ("DEBUG" in os.environ)):  # placeholder api naming
    print("Error: This example requires you to set either your CHROMADB_API_TOKEN or DEBUG=1")
    exit(1)


def extract_chromadb_dists(output: Dict[str, object]) -> list[str]:
    return output


def measure_fn():  # TODO: Do we want to build a separate framework from prompttest that handles vectors?
    pass
