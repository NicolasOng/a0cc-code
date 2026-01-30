from a0_new.protocols.model import RawModel

import time
import random
from multiprocessing import Queue

import numpy as np
from numpy.typing import NDArray

class InferenceResponse:
    def __init__(self,
                 values: NDArray[np.float32],
                 policies: NDArray[np.float32],
                 nonce: int
                ):
        self.values = values
        self.policies = policies
        self.nonce = nonce

class InferenceRequest:
    def __init__(self,
                 states: NDArray[np.float32],
                 response_queue: Queue[InferenceResponse],
                 nonce: int,
                 shutdown: bool=False
                ):
        # assumes the first dimension is the batch dimension
        self.states = states
        self.response_queue = response_queue
        self.nonce = nonce
        self.shutdown = shutdown

class DynamicBatchingModelClient(RawModel):
    '''
    DynamicBatchingModelClient should only be created after
    the DynamicBatchingModelServer is running in another process.
    Max size of the response queue should be 1.
    '''
    def __init__(self,
                 inference_queue: Queue[InferenceRequest],
                 response_queue: Queue[InferenceResponse],
                 req_timeout: float,
                 res_timeout: float
                ):
        self.inference_queue = inference_queue
        self.response_queue = response_queue
        self.req_timeout = req_timeout
        self.res_timeout = res_timeout
    
    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # create an inference request and submit it to the inference queue
        # Queue.Full exception if the queue is full for too long
        nonce = random.randint(0, 2**31 - 1)
        request = InferenceRequest(states, self.response_queue, nonce)
        self.inference_queue.put(request, block=True, timeout=self.req_timeout)

        # wait for the response with matching nonce
        # Queue.Empty exception if no response in time
        response = self.response_queue.get(block=True, timeout=self.res_timeout)
        # return the response if the nonce matches,
        # else raise an error
        if response.nonce == nonce:
            return response.values, response.policies
        else:
            raise ValueError(f"Received response with unexpected nonce {response.nonce}, expected {nonce}")

class DynamicBatchingModelServer():
    '''
    Should be created before any DynamicBatchingModelClient,
    and shut down after all clients are done.
    Max size of inference queue should be the number of clients.
    '''
    def __init__(self, model: RawModel, inference_queue: Queue[InferenceRequest], max_batch_size: int = 32, timeout: float = 0.01):
        self.model = model
        self.inference_queue = inference_queue
        self.max_batch_size = max_batch_size
        self.timeout = timeout
    
    def serve(self) -> None:
        while True:
            batch_requests: list[InferenceRequest] = []
            
            # 1. Wait for the first request (blocking)
            req = self.inference_queue.get() 
            if req.shutdown: break # Shutdown signal
            batch_requests.append(req)
            
            # 2. Try to fill the rest of the batch (non-blocking)
            # TODO: since clients can submit variable batch sizes,
            # this may exceed max_batch_size if a large request is submitted - need to handle this case
            start_time = time.time()
            while self.req_list_batch_size(batch_requests) < self.max_batch_size:
                try:
                    # Check for more items until the timeout expires
                    remaining_time = self.timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                    req = self.inference_queue.get(timeout=remaining_time)
                    batch_requests.append(req)
                except: # Queue.Empty
                    break
            
            # 3. Prepare and run inference
            inputs = np.concatenate([r.states for r in batch_requests])
            results = self.model.evaluate(inputs) 
            
            # 4. Dispatch results back to specific clients
            for i, req in enumerate(batch_requests):
                starting_index = sum(r.states.shape[0] for r in batch_requests[:i])
                ending_index = starting_index + req.states.shape[0]
                res = InferenceResponse(
                    values=results[0][starting_index:ending_index],
                    policies=results[1][starting_index:ending_index],
                    nonce=req.nonce
                )
                req.response_queue.put(res)
    
    @staticmethod
    def req_list_batch_size(req_list: list[InferenceRequest]) -> int:
        return sum(req.states.shape[0] for req in req_list)
    