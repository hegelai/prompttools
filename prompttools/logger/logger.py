# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.
import json
import uuid

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
        self.feedback_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker)

        # When the main thread is joining, put `None` into queue to signal worker thread to end
        threading.Thread(target=lambda: threading.main_thread().join() or self.data_queue.put(None)).start()

        self.worker_thread.start()

    def add_feedback(self, log_id, metric_name, value):
        self.feedback_queue.put({"log_id": log_id, "key": metric_name, "value": value})

    def add_to_queue(
        self,
        hegel_model: str,
        result: dict,
        input_parameters: dict,
        latency: float,
        log_id: str,
    ):
        self.data_queue.put(
            {
                "hegel_model": hegel_model,
                "result": result,
                "input_parameters": input_parameters,
                "latency": latency,
                "log_id": log_id,
            }
        )

    def execute_and_add_to_queue(self, callable_func, **kwargs):
        if "hegel_model" in kwargs:
            hegel_model = kwargs["hegel_model"]
            del kwargs["hegel_model"]
        else:
            hegel_model = None
        start = perf_counter()
        result = callable_func(**kwargs)
        latency = perf_counter() - start
        log_id = str(uuid.uuid4())
        self.add_to_queue(hegel_model, result.model_dump_json(), json.dumps(kwargs), latency, log_id)
        result.log_id = log_id
        return result

    def wrap(self, callable_func):
        return partial(self.execute_and_add_to_queue, callable_func)

    def worker(self):
        while True:
            # Process logging data
            if not self.data_queue.empty():
                data = self.data_queue.get()
                if data is None:  # Shutdown signal
                    return
                self.log_data_to_remote(data)
                self.data_queue.task_done()

            # Process feedback data
            if not self.feedback_queue.empty():
                feedback_data = self.feedback_queue.get()
                if feedback_data is None:  # Shutdown signal
                    return
                self.send_feedback_to_remote(feedback_data)
                self.feedback_queue.task_done()

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

    def send_feedback_to_remote(self, feedback_data):
        feedback_url = f"{HEGEL_BACKEND_URL}/sdk/add_feedback/"
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": os.environ["HEGELAI_API_KEY"],
            }

            response = requests.post(feedback_url, json=feedback_data, headers=headers)
            if response.status_code != 200:
                print(f"Failed to send feedback to Flask API. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending feedback to Flask API: {e}")


sender = Logger()


def logging_wrapper(original_fn):
    def wrapped_function(**kwargs):
        # Call the original function with the provided arguments

        if "hegel_model" in kwargs:
            hegel_model = kwargs["hegel_model"]
            del kwargs["hegel_model"]
        else:
            hegel_model = None
        start = perf_counter()
        result = original_fn(**kwargs)
        latency = perf_counter() - start
        log_id = str(uuid.uuid4())
        sender.add_to_queue(hegel_model, result.model_dump_json(), json.dumps(kwargs), latency, log_id)
        result.log_id = log_id
        return result

    return wrapped_function


# Monkey-patching main client
try:
    openai.chat.completions.create = sender.wrap(openai.chat.completions.create)
except Exception:
    print("Error monkey-patching main client")
    print("You may need to add `OPENAI_API_KEY=''` to your `.env` file.")
    raise

# Monkey-patching client instance
try:
    # This is working as of openai SDK version 1.11.1
    openai.resources.chat.completions.Completions.create = logging_wrapper(
        openai.resources.chat.completions.Completions.create
    )
except Exception:
    print("Error monkey-patch individual client.")
    raise


def add_feedback(*args):
    sender.add_feedback(*args)
