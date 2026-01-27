# from Zaheen
from __future__ import annotations
from multiprocessing import set_start_method, Process, Queue
import os
import time


def actor_process(queue: Queue[int]) -> None:
    queue.put(os.getpid())
    time.sleep(6)
    print(f"Finishing: {os.getpid()}")


def evaluate_queue(queue: Queue[int]) -> None:
    while not queue.empty():
        print(f"Evaluating: {queue.get()}")
    print("Finished evaluating.")


if __name__ == "__main__":
    set_start_method("spawn")
    num_actors: int = 4
    queue: Queue[int] = Queue()
    actors: list[Process] = [Process(target=actor_process, args=(queue,)) for _ in range(num_actors)]

    for actor in actors:
        actor.start()

    time.sleep(3)
    evaluate_queue(queue)

    for actor in actors:
        actor.join()