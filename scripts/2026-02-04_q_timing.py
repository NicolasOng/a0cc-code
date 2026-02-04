import time
import random

from multiprocessing import Queue, Process, shared_memory
from typing import Any

import numpy as np
from numpy.typing import NDArray

def print_mean_and_ci(data: list[float], description: str) -> None:
    times_array = np.array(data)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array, ddof=1)  # sample std
    n = len(times_array)
    ci_half_width = 1.96 * std_time / np.sqrt(n)
    ci_lower = mean_time - ci_half_width
    ci_upper = mean_time + ci_half_width
    
    print(f"{description} ({len(data)}):")
    print(f"  Average time: {mean_time:.4f} seconds")
    print(f"  95% CI: {mean_time:.4f} +- {ci_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] seconds")

class Req:
    def __init__(self,
                 states: NDArray[np.float32],
                 qid: int,
                 nonce: int,
                 shutdown: bool=False
                ):
        self.states = states
        self.qid = qid
        self.nonce = nonce
        self.shutdown = shutdown

class Res:
    def __init__(self,
                 values: NDArray[np.float32],
                 policies: NDArray[np.float32],
                 nonce: int,
                 data: Any
                ):
        self.values = values
        self.policies = policies
        self.nonce = nonce
        self.data = data

def q_server(req_q, res_q) -> list[float]:
    req_processing_times: list[float] = []
    while True:
        req: Req = req_q.get() 
        start_time = time.time()
        if req.shutdown:
            res: Res = Res(
                values=np.array([], dtype=np.float32),
                policies=np.array([], dtype=np.float32),
                nonce=req.nonce,
                data=req_processing_times
            )
            res_q.put(res)
            break # Shutdown signal

        res: Res = Res(
            values=np.sum(req.states, axis=(1,2,3)), # dummy processing
            policies=np.mean(req.states, axis=(1,2,3)), # dummy processing
            nonce=req.nonce,
            data=None
        )
        end_time = time.time()
        req_processing_times.append(end_time - start_time)
        res_q.put(res)
    return req_processing_times

def q_main(batch_size: int, req_num: int, req_q, res_q) -> tuple[list[float], list[float]]:
    reqres_times: list[float] = []
    for _ in range(req_num):
        states = np.random.rand(batch_size, 5, 5, 2).astype(np.float32) # single state
        nonce = random.randint(0, 2**31 - 1)
        req = Req(states, qid=0, nonce=nonce)
        start_time = time.time()
        req_q.put(req)

        res: Res = res_q.get()
        end_time = time.time()
        reqres_times.append(end_time - start_time)
        if res.nonce != nonce:
            raise ValueError(f"Received response with unexpected nonce {res.nonce}, expected {nonce}")
    req_q.put(Req(states=np.empty((0,5,5,2), dtype=np.float32), qid=0, nonce=0, shutdown=True))
    final_res = res_q.get() # get shutdown response
    return reqres_times, final_res.data

def q_test(batch_size: int = 1, req_num: int = 100) -> None:
    start_time = time.time()

    req_q = Queue(maxsize=10)
    res_q = Queue(maxsize=10)

    server_process = Process(target=q_server, args=(req_q, res_q))
    server_process.start()

    reqres_times, processing_times = q_main(batch_size, req_num, req_q, res_q)

    server_process.join()

    end_time = time.time()
    print(f'Q Test: Total time for {req_num} requests with batch size {batch_size}: {end_time - start_time:.4f} seconds.')
    avg_reqres_time = sum(reqres_times) / len(reqres_times)
    avg_processing_time = sum(processing_times) / len(processing_times)
    print(f'Average time per request-response: {avg_reqres_time:.6f} seconds.')
    print(f'Average processing time per request: {avg_processing_time:.6f} seconds.')
    print_mean_and_ci(reqres_times, "Request-Response times")
    print_mean_and_ci(processing_times, "Processing times")
    print(f'Average overhead time per request-response: {avg_reqres_time - avg_processing_time:.6f} seconds.')

def q_tests() -> None:
    for batch_size in [1, 8, 64, 512, 1024, 4096]:
        print(f'\nRunning Q test with batch size {batch_size}')
        q_test(batch_size=batch_size, req_num=1000)

if __name__ == "__main__":
    q_tests()
