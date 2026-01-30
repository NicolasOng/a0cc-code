'''
Comparing CPU vs GPU performance for models.
CPU strategy: give each process its own model instance.
GPU strategy: use dynamic batching to batch requests from multiple processes.
'''
from __future__ import annotations

from a0_new.cc.models.nn import CCNNModel
from a0_new.models.dynamic_batching import DynamicBatchingModelClient, DynamicBatchingModelServer, InferenceRequest, InferenceResponse

import dill
import multiprocessing
from multiprocessing import Process, Queue
import time

from numpy.typing import NDArray
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This class exists specifically for type hinting
    from multiprocessing.queues import Queue

def create_ccnn_model() -> CCNNModel:
    # can load a model, but for now just create a new one
    return CCNNModel()

def cpu_process(serialized_model: bytes, states: NDArray[np.float32]) -> None:
    model: CCNNModel = dill.loads(serialized_model)
    max_batch_size = 1

    batch_size = states.shape[0]
    num_batches = (batch_size + max_batch_size - 1) // max_batch_size
    start_time = time.time()
    for i in range(num_batches):
        batch_states = states[i * max_batch_size : (i + 1) * max_batch_size]
        values, policies = model.evaluate(batch_states)
    end_time = time.time()
    print(f'CPU Process: Evaluated {batch_size} states in {end_time - start_time:.4f} seconds.')

def cpu_test(n: int = 100, num_processes: int = 4) -> None:
    start_time = time.time()
    model = create_ccnn_model()
    serialized_model = dill.dumps(model)

    num_states = 100
    board_size = 5
    states = np.random.rand(num_states, board_size, board_size, 2).astype(np.float32)

    processes: list[Process] = []
    for _ in range(num_processes): # 4 CPU processes
        p = Process(target=cpu_process, args=(serialized_model, states))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f'CPU Test: Total time for {num_processes} processes: {end_time - start_time:.4f} seconds.')

def gpu_server_process(serialized_model: bytes, inference_queue: Queue[InferenceRequest], response_queues: dict[int, Queue[InferenceResponse]], num_clients: int) -> None:
    model: CCNNModel = dill.loads(serialized_model)
    server = DynamicBatchingModelServer(model, inference_queue, response_queues, max_batch_size=num_clients, timeout=1)
    server.serve()

def gpu_client_process(inference_queue: Queue[InferenceRequest], response_queue: Queue[InferenceResponse], qid: int, states: NDArray[np.float32]) -> None:
    client = DynamicBatchingModelClient(inference_queue, response_queue, qid, req_timeout=5, res_timeout=5)

    # can only send requests of size up to 1
    start_time = time.time()
    for i in range(states.shape[0]):
        state = states[i : i + 1] # single state
        values, policies = client.evaluate(state)
    end_time = time.time()
    print(f'GPU Client Process: Evaluated {states.shape[0]} states in {end_time - start_time:.4f} seconds.')

def gpu_test(n: int = 100, num_clients: int = 4) -> None:
    start_time = time.time()
    model = create_ccnn_model()
    serialized_model = dill.dumps(model)

    inference_queue: Queue[InferenceRequest] = Queue(maxsize=num_clients)
    response_queues: dict[int, Queue[InferenceResponse]] = {i: Queue(maxsize=1) for i in range(num_clients)}

    server_process = Process(target=gpu_server_process, args=(serialized_model, inference_queue, response_queues, num_clients))
    server_process.start()

    num_states = 100
    board_size = 5
    states = np.random.rand(num_states, board_size, board_size, 2).astype(np.float32)

    processes: list[Process] = []
    for i in range(num_clients): # 4 GPU client processes
        p = Process(target=gpu_client_process, args=(inference_queue, response_queues[i], i, states))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Send shutdown signal to server
    shutdown_request = InferenceRequest(states=np.empty((0, board_size, board_size, 2), dtype=np.float32), qid=0, nonce=0, shutdown=True)
    inference_queue.put(shutdown_request)

    server_process.join()

    end_time = time.time()
    print(f'GPU Test: Total time for {num_clients} clients: {end_time - start_time:.4f} seconds.')

if __name__ == "__main__":
    run_gpu_test = False

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    if not run_gpu_test:
        print("Starting CPU Test")
        cpu_test(100000, 32)

    if run_gpu_test:
        print("Starting GPU Test")
        gpu_test(100000, 32)