from __future__ import annotations

from typing import Any

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from a0_new.protocols.model import A0Model, RawModel
from a0_new.protocols.game import A0Game
from a0_new.protocols.player import A0Player

from multiprocessing import Process, Queue, Array, Event
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.synchronize import Event as EventType

import pickle
import dill

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This class exists specifically for type hinting
    from multiprocessing.queues import Queue

from a0_new.models.dynamic_batching import DynamicBatchingModelClient, InferenceRequest, InferenceResponse, DynamicBatchingModelServer
from a0_new.experience_buffer import ExperienceBuffer, ExperienceData
from a0_new.play import play, GameData

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def _gpu_server_process(
        serialized_model: bytes,
        inference_queue: Queue[InferenceRequest],
        response_queues: dict[int, Queue[InferenceResponse]],
        num_clients: int
    ) -> None:
    # deserialize the model the server will use for inference
    model: RawModel = dill.loads(serialized_model)
    # create and start the dynamic batching server
    server = DynamicBatchingModelServer(
        model, inference_queue, response_queues, max_batch_size=num_clients, timeout=1
    )
    server.serve()

def _play_process(
        game: A0Game[Any, Any],
        player: A0Player[A0Model[Any, Any, Any], Any, Any],
        state_counter: SynchronizedArray[int],
        result_queue: Queue[Any],
        shutdown_event: EventType
    ) -> None:
    # TODO: implement self-play logic
    pass

def self_play(
        game: A0Game[Any, Any],
        player: A0Player[A0Model[Any, Any, Any], Any, Any],
        experience_buffer: ExperienceBuffer,
        iteration: int
    ) -> None:
    # gamedata list
    gamedata_list: list[GameData] = []

    # get and serialize the model the player is using
    player_model = player.get_model()
    serialized_player_model = dill.dumps(player_model)

    # get the number of clients to use, create the queues and shared variables
    num_workers = 4 # TODO: use config
    inference_queue: Queue[InferenceRequest] = Queue(maxsize=num_workers)
    response_queues: dict[int, Queue[InferenceResponse]] = {
        i: Queue(maxsize=1) for i in range(num_workers)
    }
    result_queue: Queue[tuple[GameData, list[ExperienceData]]] = Queue(maxsize=num_workers)
    state_counter: SynchronizedArray[int] = Array('i', [0] * num_workers)
    shutdown_event = Event()

    # start the GPU server process,
    # which will use the given player model for inference
    server_process = Process(
        target=_gpu_server_process,
        args=(serialized_player_model, inference_queue, response_queues, num_workers)
    )
    server_process.start()

    # start the play processes,
    # which will send inference requests to the server
    # and game results back to the main process
    processes: list[Process] = []
    for i in range(num_workers):
        res_q = response_queues[i]
        # first, create the raw model client for the player
        client = DynamicBatchingModelClient(
            inference_queue, res_q, i, req_timeout=5, res_timeout=5
        )
        # replace the raw model in the player's a0 model with the client
        player_model.set_raw_model(client)
        # set the updated model back to the player
        player.set_model(player_model)
        # clone the player for the process
        new_player = player.clone()
        # start the play process with the player using the client model
        p = Process(
            target=_play_process,
            args=(game, new_player, state_counter, result_queue, shutdown_event)
        )
        p.start()
        processes.append(p)
    
    # process results and coordinate shutdown
    while True:
        # process self-play results as they come in,
        # add to gamedata list and experience buffer
        gamedata, experience_data_list = result_queue.get()
        gamedata_list.append(gamedata)
        for experience_data in experience_data_list:
            experience_buffer.add(experience_data)
        
        # check for shutdown condition
        with state_counter.get_lock():
            # get the total number of states played so far
            total_states = sum(list(state_counter))
            # if enough, set the shutdown event
            if total_states >= config.training_samples:
                shutdown_event.set()
            # log state counts
            for count in state_counter:
                # TODO: log each process's state count
                pass
            # break the loop if shutdown event is set
            if shutdown_event.is_set():
                break
    
    # wait for all play processes to finish
    for p in processes:
        p.join()
    
    # Send shutdown signal to server
    shutdown_request = InferenceRequest(
        states=np.empty((0, config.board_size, config.board_size, 2), dtype=np.float32),
        qid=0, nonce=0, shutdown=True
    )
    inference_queue.put(shutdown_request)

    # save the game data to disk
    if config.training_dir:
        with open(config.training_dir + f"gamedata_{iteration + 1}.pkl", 'wb') as f:
            pickle.dump(gamedata_list, f)
    
    # put the original model back to the player
    original_model: A0Model[Any, Any, Any] = dill.loads(serialized_player_model)
    player.set_model(original_model)

    # wait for the server process to finish
    server_process.join()
