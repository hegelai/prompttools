# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
from os.path import join, dirname

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    dotenv_path = join(dirname(dirname(__file__)), ".env")
    load_dotenv(dotenv_path)


ENV = os.environ.get("ENV", "prod")
if ENV == "development":
    HEGEL_BACKEND_URL = """http://127.0.0.1:5000"""
else:
    HEGEL_BACKEND_URL = """https://api.hegel-ai.com"""
