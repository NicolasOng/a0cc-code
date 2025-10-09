import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import os
import time
import random

def _new_thread(input: int) -> int:
    # Simulate computation time
    sleep_time = random.uniform(1, 5)  # Random sleep between 1-5 seconds
    time.sleep(sleep_time)
    print(f"Thread with input {input} finished after {sleep_time:.2f} seconds")
    return input * 2

def processes_test() -> None:
    input = 10
    full_list = 30
    items: list[int] = []
    num_cores = os.cpu_count() or 4
    print(f"Using {num_cores} cores for self-play.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # create a list to hold the futures
        futures: list[Future[int]] = []

        # create a function to start a thread
        def start_thread() -> None:
            future = executor.submit(
                _new_thread,
                input=input
            )
            futures.append(future)
        
        # start a thread for each core
        for _ in range(num_cores):
            start_thread()
        
        while True:
            print(f"Current list size: {len(items)}")
            # when a thread finishes,
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            # for each finished thread,
            for future in done:
                # remove it from the list of futures
                futures.remove(future)
                # get the result from the thread
                result = future.result()
                # add the result to the list of items
                items.append(result)
                # if the training set is not full yet, start a new thread
                if len(items) < full_list:
                    start_thread()

            # stop when the list is full
            if len(items) >= full_list:
                print("List is full, cancelling all threads")
                # need to wait for remaining threads to finish
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True)
                break
            print(f"Waiting for {len(futures)} threads to finish...")

    print(f"Generated list of size: {len(items)}/{full_list}")

if __name__ == "__main__":
    processes_test()
