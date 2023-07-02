from queue import Queue, Empty
from time import perf_counter
import threading
import openai
import logging

from prompttools.requests.retries import retry_decorator


class RequestQueue:
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

    def _do_task(self, fn, args):
        try:
            res = self._run(fn, args)
            self.request_results.append(res[0])
            self.request_latencies.append(res[1])
        # TODO: If we get an unexpected error here, the queue will hang
        except openai.error.AuthenticationError as e:
            logging.error("Authentication error. Skipping request.")

    @retry_decorator
    def _run(self, fn, args):
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
        # have the following error: TypeError: 'NoneType' object is not callable
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
