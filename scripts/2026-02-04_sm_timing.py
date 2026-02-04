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

def shm_server(req_q, res_q, req_shm_name, res_p_shm_name, res_v_shm_name, req_shape, res_p_shape, res_v_shape, dtype) -> list[float]:
    # Attach to the existing shared memory blocks
    existing_req_shm = shared_memory.SharedMemory(name=req_shm_name)
    existing_res_p_shm = shared_memory.SharedMemory(name=res_p_shm_name)
    existing_res_v_shm = shared_memory.SharedMemory(name=res_v_shm_name)
    # Create numpy arrays backed by the shared memory
    req_shm_array = np.ndarray(req_shape, dtype=dtype, buffer=existing_req_shm.buf)
    res_p_shm_array = np.ndarray(res_p_shape, dtype=dtype, buffer=existing_res_p_shm.buf)
    res_v_shm_array = np.ndarray(res_v_shape, dtype=dtype, buffer=existing_res_v_shm.buf)
    
    req_processing_times: list[float] = []
    
    while True:
        req_info = req_q.get() # req_info could just be the nonce or metadata
        if req_info is None: # Shutdown signal
            res_q.put(req_processing_times)
            break
        
        start_time = time.time()
        nonce = req_info['nonce']

        # Data is already in shm_array thanks to q_main copying it there
        # We perform dummy processing on the shared buffer
        values = np.random.rand(*res_v_shape).astype(dtype)
        policies = np.random.rand(*res_p_shape).astype(dtype)

        end_time = time.time()
        req_processing_times.append(end_time - start_time)

        # Write results back to the response shared memory
        res_p_shm_array[:] = policies
        res_v_shm_array[:] = values

        # Signal completion
        res_q.put({'nonce': nonce})
        
    existing_req_shm.close()
    existing_res_p_shm.close()
    existing_res_v_shm.close()

    return req_processing_times

def shm_main(batch_size: int, req_num: int, req_q, res_q, req_shm_name, res_p_shm_name, res_v_shm_name, req_shape, res_p_shape, res_v_shape, dtype):
    # Attach to the existing shared memory blocks
    existing_req_shm = shared_memory.SharedMemory(name=req_shm_name)
    existing_res_p_shm = shared_memory.SharedMemory(name=res_p_shm_name)
    existing_res_v_shm = shared_memory.SharedMemory(name=res_v_shm_name)
    # Create numpy arrays backed by the shared memory
    req_shm_array = np.ndarray(req_shape, dtype=dtype, buffer=existing_req_shm.buf)
    res_p_shm_array = np.ndarray(res_p_shape, dtype=dtype, buffer=existing_res_p_shm.buf)
    res_v_shm_array = np.ndarray(res_v_shape, dtype=dtype, buffer=existing_res_v_shm.buf)
    
    reqres_times: list[float] = []
    
    for _ in range(req_num):
        # Generate data locally
        states = np.random.rand(*req_shape).astype(dtype)
        nonce = random.randint(0, 2**31 - 1)
        
        start_time = time.time()
        
        # Copy data INTO shared memory
        # This is where the "overhead" shift happens: from pickling to memory copying
        np.copyto(req_shm_array, states)
        
        req_q.put({'nonce': nonce})
        
        # Wait for server to finish processing the SHM
        res = res_q.get()

        # don't care about the results for timing purposes, but could read them from shared memory if needed
        
        end_time = time.time()
        reqres_times.append(end_time - start_time)
        
        if res['nonce'] != nonce:
            raise ValueError("Nonce mismatch")

    req_q.put(None) # Shutdown
    processing_times = res_q.get()
    
    existing_req_shm.close()
    existing_res_p_shm.close()
    existing_res_v_shm.close()
    return reqres_times, processing_times

def shm_test(batch_size: int = 1, req_num: int = 100) -> None:
    start_time = time.time()
    req_shape = (batch_size, 5, 5, 2)
    res_p_shape = (batch_size, 5**4)
    res_v_shape = (batch_size, 1)
    dtype1 = np.float32
    # Calculate size: (elements * bytes per float32)
    req_size = int(np.prod(req_shape) * np.dtype(dtype1).itemsize)
    res_p_size = int(np.prod(res_p_shape) * np.dtype(dtype1).itemsize)
    res_v_size = int(np.prod(res_v_shape) * np.dtype(dtype1).itemsize)
    
    # Create the shared memory blocks
    req_shm = shared_memory.SharedMemory(create=True, size=req_size)
    res_p_shm = shared_memory.SharedMemory(create=True, size=res_p_size)
    res_v_shm = shared_memory.SharedMemory(create=True, size=res_v_size)
    
    req_q = Queue()
    res_q = Queue()

    try:
        server_process = Process(target=shm_server, args=(req_q, res_q, req_shm.name, res_p_shm.name, res_v_shm.name, req_shape, res_p_shape, res_v_shape, dtype1))
        server_process.start()

        reqres_times, processing_times = shm_main(batch_size, req_num, req_q, res_q, req_shm.name, res_p_shm.name, res_v_shm.name, req_shape, res_p_shape, res_v_shape, dtype1)
        
        server_process.join()

        end_time = time.time()
        
        print(f"--- Batch Size: {batch_size} ---")
        print(f'Q Test: Total time for {req_num} requests with batch size {batch_size}: {end_time - start_time:.4f} seconds.')
        avg_reqres_time = sum(reqres_times) / len(reqres_times)
        avg_processing_time = sum(processing_times) / len(processing_times)
        print(f'Average time per request-response: {avg_reqres_time:.6f} seconds.')
        print(f'Average processing time per request: {avg_processing_time:.6f} seconds.')
        print_mean_and_ci(reqres_times, "Request-Response times")
        print_mean_and_ci(processing_times, "Processing times")
        print(f'Average overhead time per request-response: {avg_reqres_time - avg_processing_time:.6f} seconds.')

        
    finally:
        # Crucial: clean up the memory block
        req_shm.close()
        req_shm.unlink()
        res_p_shm.close()
        res_p_shm.unlink()
        res_v_shm.close()
        res_v_shm.unlink()

def shm_tests() -> None:
    for batch_size in [1, 8, 64, 512, 1024, 4096]:
        print(f'\nRunning SHM test with batch size {batch_size}')
        shm_test(batch_size=batch_size, req_num=1000)

if __name__ == "__main__":
    shm_tests()