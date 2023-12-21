# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import json

import requests
import threading
import queue
from functools import partial
import openai
import os
from dotenv import load_dotenv
from os.path import join, dirname
from time import perf_counter
from prompttools.common import HEGEL_BACKEND_URL


# Load "OPENAI_API_KEY" into `os.environ["OPENAI_API_KEY"]`
# See `.env.example`
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)


class Logger:
    def __init__(self):
        self.backend_url = f"{HEGEL_BACKEND_URL}/sdk/logger"
        self.data_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker)

        # When the main thread is joining, put `None` into queue to signal worker thread to end
        threading.Thread(target=lambda: threading.main_thread().join() or self.data_queue.put(None)).start()

        self.worker_thread.start()

    def execute_and_add_to_queue(self, callable_func, **kwargs):
        if "hegel_model" in kwargs:
            hegel_model = kwargs["hegel_model"]
            del kwargs["hegel_model"]
        else:
            hegel_model = None
        start = perf_counter()
        result = callable_func(**kwargs)
        latency = perf_counter() - start
        self.data_queue.put(
            {
                "hegel_model": hegel_model,
                "result": result.model_dump_json(),
                "input_parameters": json.dumps(kwargs),
                "latency": latency,
            }
        )
        return result

    def wrap(self, callable_func):
        return partial(self.execute_and_add_to_queue, callable_func)

    def worker(self):
        while True:
            if not self.data_queue.empty():
                result = self.data_queue.get()
                if result is None:
                    return
                self.log_data_to_remote(result)
                self.data_queue.task_done()

    def log_data_to_remote(self, data):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": os.environ["HEGELAI_API_KEY"],
            }

            response = requests.post(self.backend_url, json=data, headers=headers)
            if response.status_code != 200:
                print(f"Failed to send data to Flask API. Status code: {response.status_code} for {data}.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to Flask API: {e}")


sender = Logger()
# Monkey-patching
try:
    openai.chat.completions.create = sender.wrap(openai.chat.completions.create)
except Exception:
    print("You may need to add `OPENAI_API_KEY=''` to your `.env` file.")
    raise
