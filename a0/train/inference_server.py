'''
Single-model inference server for AlphaZero self-play.

Instead of every self-play worker holding its own copy of the model (and paying a
fresh JAX compile/JIT per process), one dedicated *server process* owns the only
model on the device. Workers hold a lightweight `InferenceClient` that proxies
`model.inference(x)` calls over multiprocessing queues. The server does dynamic
batching: it gathers requests across workers (up to `max_batch_size` or a short
timeout) and runs a single batched forward pass, then scatters the results back
to each worker by `qid`/`nonce`.

Ported from a0_new/models/dynamic_batching.py, adapted to a0's model interface:
the client exposes `.inference(x) -> (values, policies)` so it is a drop-in
replacement for `AlphaZeroModel` at every self-play call site (MCTS_NN,
model_utils policy helpers, and the value-target builders in alphazero.py).
'''
from __future__ import annotations

import time
import random

from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # mp.Queue / Value are only generic for type-checkers
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
    def __init__(self, values: NDArray[np.float32], policies: NDArray[np.float32], nonce: int):
        self.values = values
        self.policies = policies
        self.nonce = nonce


class InferenceRequest:
    def __init__(self, states: NDArray[np.float32], qid: int, nonce: int, shutdown: bool = False):
        # assumes the first dimension is the batch dimension
        self.states = states
        self.qid = qid
        self.nonce = nonce
        self.shutdown = shutdown


class InferenceClient:
    '''
    Drop-in replacement for `AlphaZeroModel` on the self-play path: exposes
    `.inference(x) -> (values, policies)`. Should only be created after the
    `InferenceServer` is running in another process. Its response queue must
    have maxsize 1 (one in-flight request per client).

    Inputs may be NumPy or JAX arrays; they are converted to NumPy before being
    shipped over the queue (device arrays don't pickle cleanly across `spawn`).
    Outputs are NumPy arrays. Every self-play consumer of `model.inference`
    treats the results as array-likes — `float(value[0][0])`, `np.array(policy[0])`,
    `jnp.concatenate([...])` — all of which accept NumPy.
    '''
    def __init__(self, inference_queue: QueueReq, response_queue: QueueRes, qid: int,
                 req_timeout: float, res_timeout: float):
        self.inference_queue = inference_queue
        self.response_queue = response_queue
        self.qid = qid
        self.req_timeout = req_timeout
        self.res_timeout = res_timeout

    def inference(self, x: Any) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        # accept jnp or np; ship np over the queue
        states = np.asarray(x, dtype=np.float32)
        nonce = random.randint(0, 2**31 - 1)
        request = InferenceRequest(states, self.qid, nonce)
        try:
            # Queue.Full if the server can't accept the request in time
            self.inference_queue.put(request, block=True, timeout=self.req_timeout)
            # Queue.Empty if no response comes back in time
            response = self.response_queue.get(block=True, timeout=self.res_timeout)
        except Exception as e:
            logger.error(
                f"{time.time()}\n"
                f"Error in inference client {self.qid} with nonce {nonce}: {type(e).__name__} {e}.\n"
                f"States shape: {states.shape}, req_timeout: {self.req_timeout}, res_timeout: {self.res_timeout}\n"
                f"inference queue size: {self.inference_queue.qsize()}, "
                f"response queue size: {self.response_queue.qsize()}"
            )
            raise
        if response.nonce != nonce:
            raise ValueError(f"Client {self.qid} received response with unexpected nonce "
                             f"{response.nonce}, expected {nonce}")
        return response.values, response.policies


class InferenceServer:
    '''
    Owns the only model on the device and serves batched inference to clients.
    Create it before any `InferenceClient`, and shut it down (via a `shutdown=True`
    request) after all clients are done. The inference queue's maxsize should be
    >= the number of concurrent clients.
    '''
    def __init__(self, model: Any, inference_queue: QueueReq,
                 response_queues: dict[int, QueueRes], num_active_clients: SynchronizedInt,
                 max_batch_size: int, timeout: float = 0.05):
        self.model = model
        self.inference_queue = inference_queue
        self.response_queues = response_queues
        self.num_active_clients = num_active_clients
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        # batching stats, logged on shutdown
        self.total_wait_time = 0.0
        self.num_timeouts = 0
        self.num_runs = 0
        self.total_clients_on_timeout = 0

    @staticmethod
    def _batch_size(req_list: list[InferenceRequest]) -> int:
        return sum(req.states.shape[0] for req in req_list)

    def serve(self) -> None:
        while True:
            batch_requests: list[InferenceRequest] = []

            # 1. Wait for the first request (long timeout so the server doesn't spin).
            try:
                req = self.inference_queue.get(timeout=60.0)
            except Exception:
                logger.warning(
                    f"{time.time()}\n"
                    f"Inference server waited 60s for a request, none received. "
                    f"queue size: {self.inference_queue.qsize()}, "
                    f"active clients: {self.num_active_clients.value}"
                )
                continue
            if req.shutdown:
                break
            batch_requests.append(req)

            # 2. Fill the rest of the batch until: max_batch_size reached, all active
            #    clients have submitted, or the batch-collection timeout expires.
            start_time = time.time()
            remaining_time = self.timeout
            while not (
                self._batch_size(batch_requests) >= self.max_batch_size
                or len(batch_requests) >= self.num_active_clients.value
            ):
                remaining_time = self.timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                try:
                    req = self.inference_queue.get(timeout=remaining_time)
                except Exception:  # Queue.Empty
                    break
                if req.shutdown:
                    # a shutdown arriving mid-batch: serve what we have, then stop
                    self._run_and_dispatch(batch_requests)
                    logger.info("Inference server received shutdown mid-batch.")
                    self._log_stats()
                    return
                batch_requests.append(req)

            wait_time = time.time() - start_time
            self.total_wait_time += wait_time
            if remaining_time <= 0:
                self.num_timeouts += 1
                self.total_clients_on_timeout += len(batch_requests)
            self.num_runs += 1

            # 3 & 4. Run inference and scatter results back to clients.
            self._run_and_dispatch(batch_requests)

        logger.info("Inference server shutting down.")
        self._log_stats()

    def _run_and_dispatch(self, batch_requests: list[InferenceRequest]) -> None:
        if not batch_requests:
            return
        inputs = np.concatenate([r.states for r in batch_requests], axis=0)
        values, policies = self.model.inference(inputs)
        # to numpy for shipping back over the queue
        values = np.asarray(values, dtype=np.float32)
        policies = np.asarray(policies, dtype=np.float32)

        start = 0
        for req in batch_requests:
            end = start + req.states.shape[0]
            res = InferenceResponse(
                values=values[start:end],
                policies=policies[start:end],
                nonce=req.nonce,
            )
            self.response_queues[req.qid].put(res)
            start = end

    def _log_stats(self) -> None:
        avg_wait = self.total_wait_time / self.num_runs if self.num_runs > 0 else 0.0
        avg_on_timeout = (
            self.total_clients_on_timeout / self.num_timeouts if self.num_timeouts > 0 else 0.0
        )
        logger.info(
            f"Inference server stats: runs={self.num_runs}, avg_wait={avg_wait:.4f}s, "
            f"timeouts={self.num_timeouts}, avg_batch_on_timeout={avg_on_timeout:.2f}"
        )
