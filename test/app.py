# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


r"""
App for local testing of logger
"""

from flask import Flask, request
import time

app = Flask(__name__)


@app.route("/", methods=["POST"])
def process_request():
    time.sleep(0.1)
    data = request.json
    print(f"Request received and processed {data}.")
    return "", 200


if __name__ == "__main__":
    app.run(debug=True)
