from queue import Queue, Empty
from time import perf_counter
import threading
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import openai
import logging

class RequestQueue:
    def __init__(self):
        self.data_queue = Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.request_results = []
        self.request_latencies = []

    # TODO: Use this to add retry decorators to OpenAI API calls
    def _create_retry_decorator():
        min_seconds = 4
        max_seconds = 10
        max_attempts = 5
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        )

    def _process_queue(self) -> None:
        while self.is_running:
            try:
                fn, args = self.data_queue.get(timeout=0.2)
                try:
                    res = self._run(fn, args)
                    self.request_results.append(res[0])
                    self.request_latencies.append(res[1])
                except openai.error.AuthenticationError as e:
                    logging.error("Authentication error. Skipping request.")
                self.data_queue.task_done()
            except Empty:
                continue

    def _run(self, fn, args, retry = False):
        start = perf_counter()
        print(args)
        result = fn(**args)
        return result, perf_counter() - start
    
    def shutdown(self) -> None:
        r"""
        Stops the worker thread from executed and joins it.
        """
        self.data_queue.join()
        self.is_running = False
        self.worker_thread.join()

    def __del__(self) -> None:
        self.shutdown()

    def enqueue(self, callable, args):
        self.data_queue.put((callable, args))

    def results(self):
        self.data_queue.join()
        return self.request_results
    
    def latencies(self):
        self.data_queue.join()
        return self.request_latencies
