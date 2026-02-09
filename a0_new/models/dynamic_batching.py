from __future__ import annotations

from a0_new.protocols.model import RawModel

import time
import random

from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    QueueRes = Queue["InferenceResponse"]
    QueueReq = Queue["InferenceRequest"]
    SynchronizedInt = Synchronized[int]
else:
    QueueRes = Queue
    QueueReq = Queue
    SynchronizedInt = Synchronized

import numpy as np
from numpy.typing import NDArray

from utils.log import get_logger
logger = get_logger(__name__)

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
                 qid: int,
                 nonce: int,
                 shutdown: bool=False
                ):
        # assumes the first dimension is the batch dimension
        self.states = states
        self.qid = qid
        self.nonce = nonce
        self.shutdown = shutdown

class DynamicBatchingModelClient(RawModel):
    '''
    DynamicBatchingModelClient should only be created after
    the DynamicBatchingModelServer is running in another process.
    Max size of the response queue should be 1.
    '''
    def __init__(self,
                 inference_queue: QueueReq,
                 response_queue: QueueRes,
                 qid: int,
                 req_timeout: float,
                 res_timeout: float
                ):
        self.inference_queue = inference_queue
        self.response_queue = response_queue
        self.qid = qid
        self.req_timeout = req_timeout
        self.res_timeout = res_timeout
    
    def evaluate(self, states: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # create an inference request
        nonce = random.randint(0, 2**31 - 1)
        request = InferenceRequest(states, self.qid, nonce)
        try:
            # submit the request to the inference queue
            # Queue.Full exception if the queue is full for too long
            self.inference_queue.put(request, block=True, timeout=self.req_timeout)
            # wait for the response with matching nonce
            # Queue.Empty exception if no response in time
            response = self.response_queue.get(block=True, timeout=self.res_timeout)
        except Exception as e:
            logger.error(f'''{time.time()}
Error in client {self.qid} with nonce {nonce}: {type(e).__name__} {e}.
States shape: {states.shape}, req_timeout: {self.req_timeout}, res_timeout: {self.res_timeout}
size of inference queue: {self.inference_queue.qsize()}, size of response queue: {self.response_queue.qsize()}
''')
            raise
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
    def __init__(
            self,
            model: RawModel,
            inference_queue: QueueReq,
            response_queues: dict[int, QueueRes],
            num_active_clients: SynchronizedInt,
            max_batch_size: int,
            timeout: float = 0.05
        ):
        self.model = model
        self.inference_queue = inference_queue
        self.response_queues = response_queues
        self.num_active_clients = num_active_clients
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.total_wait_time = 0.0
        self.num_timeouts = 0
        self.num_runs = 0
        self.total_clients_on_timeout = 0
    
    def serve(self) -> None:
        while True:
            #logger.log(10, "Server waiting for requests...")
            batch_requests: list[InferenceRequest] = []
            
            # 1. Wait for the first request (with timeout)
            try:
                req = self.inference_queue.get(timeout=60.0)
            except:
                logger.warning(f'''{time.time()}
Server waited 60 seconds for first request, no request received.
Inference queue size: {self.inference_queue.qsize()}, Number of active clients: {self.num_active_clients.value}
batch_requests len: {len(batch_requests)}
''')
                continue
            if req.shutdown: break # Shutdown signal
            batch_requests.append(req)
            #logger.log(10, f"Server received first request from client {req.qid} with nonce {req.nonce}, starting batch collection...")
            
            # 2. Try to fill the rest of the batch (non-blocking)
            # stop when any of the following conditions are met:
            #   - max batch size is reached
            #   - all clients have submitted requests
            #   - timeout is reached
            # TODOs when I get around to variable batch sizes.
            # for now, assume num_clients == max_batch_size and that all clients submit batch sizes of 1
            # TODO: since clients can submit variable batch sizes,
            # this may exceed max_batch_size if a large request is submitted - need to handle this case
            # TODO: also need to handle the case where requests recieved don't fill the batch before timeout.
            # might be best to pad with dummy requests if always running the same batch size is important for performance.
            start_time = time.time()
            remaining_time = self.timeout
            while not (
                self.req_list_batch_size(batch_requests) >= self.max_batch_size or
                len(batch_requests) >= self.num_active_clients.value
            ):
                try:
                    # Check for more items until the timeout expires
                    remaining_time = self.timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                    req = self.inference_queue.get(timeout=remaining_time)
                    if req.shutdown:
                        break
                    else:
                        batch_requests.append(req)
                    #logger.log(10, f"Server received additional request from client {req.qid} with nonce {req.nonce}, batch size now {self.req_list_batch_size(batch_requests)}")
                except: # Queue.Empty
                    # logger.log(15, f"Server batch collection timeout reached, proceeding with batch of size {self.req_list_batch_size(batch_requests)}")
                    # logger.log(15, f"Batch requests from clients {[r.qid for r in batch_requests]}")
                    # logger.log(15, f"Number of active clients: {self.num_active_clients.value}")
                    break
            
            wait_time = time.time() - start_time
            self.total_wait_time += wait_time
            if remaining_time <= 0:
                self.num_timeouts += 1
                self.total_clients_on_timeout += len(batch_requests)
            self.num_runs += 1
            
            # 3. Prepare and run inference
            #logger.log(10, f"Server starting inference for batch of size {self.req_list_batch_size(batch_requests)}")
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
                self.response_queues[req.qid].put(res)
                #logger.log(10, f"Server sent response to client {req.qid} with nonce {req.nonce}")
        
        logger.info("Server shutting down.")
        logger.info(f"Average wait time: {self.total_wait_time / self.num_runs if self.num_runs > 0 else 0:.4f}, Timeouts: {self.num_timeouts}, Runs: {self.num_runs}, Average clients on timeout: {self.total_clients_on_timeout / self.num_timeouts if self.num_timeouts > 0 else 0:.2f}")
    
    @staticmethod
    def req_list_batch_size(req_list: list[InferenceRequest]) -> int:
        return sum(req.states.shape[0] for req in req_list)
    