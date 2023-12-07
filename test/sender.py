import requests
import threading
import queue
from functools import partial


class Sender:
    def __init__(self):
        self.flask_api_url = "http://localhost:5000/"
        self.data_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker)

        # When the main thread is joining, put `None` into queue to signal worker thread to end
        threading.Thread(target=lambda: threading.main_thread().join() or self.data_queue.put(None)).start()

        self.worker_thread.start()

    def execute_and_add(self, callable_func):
        result = callable_func()
        self.data_queue.put(result)

    def worker(self):
        while True:
            if not self.data_queue.empty():
                result = self.data_queue.get()
                if result is None:
                    return
                self.send_data_to_flask(result)
                self.data_queue.task_done()

    def send_data_to_flask(self, data):
        try:
            response = requests.post(self.flask_api_url, json=data)
            if response.status_code == 200:
                print(f"Data sent to Flask API: {data}")
            else:
                print(f"Failed to send data to Flask API. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to Flask API: {e}")


if __name__ == "__main__":
    sender = Sender()

    # Example usage:
    def example_callable(i: int):
        print(f"Executing callable {i}")
        return {"message": f"Hello, {i}!"}

    for i in range(3):
        sender.execute_and_add(partial(example_callable, i))
    print("end")
