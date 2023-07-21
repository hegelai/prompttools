# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Tuple
from queue import Queue, Empty
from time import perf_counter
import threading
import openai
import logging

from prompttools.requests.retries import retry_decorator


class RequestQueue:
    r"""
    A generic queue for processing requests in the `prompttools` library.
    It can be used to handle and time requests to any LLM asynchronously.
    """

    def __init__(self):
        self.data_queue = Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.request_results = []
        self.request_latencies = []

    def _process_queue(self) -> None:
        while self.is_running:
            try:
                fn, args = self.data_queue.get(timeout=0.2)
                self._do_task(fn, args)
                self.data_queue.task_done()
            except Empty:
                continue

    def _do_task(self, fn: Callable, args: Dict[str, object]) -> None:
        try:
            res = self._run(fn, args)
            self.request_results.append(res[0])
            self.request_latencies.append(res[1])
        # TODO: If we get an unexpected error here, the queue will hang
        except openai.error.AuthenticationError:
            logging.error("Authentication error. Skipping request.")

    @retry_decorator
    def _run(self, fn: Callable, args: Dict[str, object]) -> Tuple[Dict[str, object], float]:
        start = perf_counter()
        result = fn(**args)
        return result, perf_counter() - start

    def shutdown(self) -> None:
        r"""
        Stops the worker thread from executed and joins it.
        """
        self.data_queue.join()
        self.is_running = False
        # TODO: If we are hanging and interrupt, this line will
        #       have the following error: TypeError: 'NoneType' object is not callable
        self.worker_thread.join()

    def __del__(self) -> None:
        self.shutdown()

    def enqueue(self, callable: Callable, args: Dict[str, object]) -> None:
        r"""
        Adds another request to the queue.
        """
        self.data_queue.put((callable, args))

    def results(self) -> List[Dict[str, object]]:
        r"""
        Joins the queue and gets results.
        """
        self.data_queue.join()
        return self.request_results

    def latencies(self) -> List[float]:
        r"""
        Joins the queue and gets latencies.
        """
        self.data_queue.join()
        return self.request_latencies
