from __future__ import annotations

from typing import Any, cast

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from a0_new.protocols.model import RecursiveFullOnRawModel, T_nn_model, RawModel, FullModelOnRaw, FullModelOnFull
from a0_new.protocols.game import A0Game
from a0_new.protocols.player import FullModelPlayer

from multiprocessing import Process, Queue, Array, Event, Value
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from multiprocessing.synchronize import Event as EventType

import pickle
import dill

import numpy as np
from numpy.typing import NDArray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This class exists specifically for type hinting
    from multiprocessing.queues import Queue

from a0_new.experience_buffer import ExperienceBuffer, ExperienceData
from a0_new.play import play, GameData
from a0_new.utils.model import get_full_on_raw_from_player
from a0_new.train.self_play.dynamic_batching import gamedata_to_experiencedata

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)

def _play_process(
        game: A0Game[Any, Any],
        serialized_player: bytes,
        state_counter: SynchronizedArray[int],
        result_queue: Queue[tuple[GameData, list[ExperienceData]]],
        num_active_workers: Synchronized[int],
        shutdown_event: EventType,
        pid: int
    ) -> None:
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="play_process_" + str(pid)
    )
    logger.info(f"Play process {pid} started")

    # deserialize the player
    player: FullModelPlayer[RecursiveFullOnRawModel[Any], Any, Any] = dill.loads(serialized_player)

    # get the full model on raw from the player,
    # to use its methods for processing gamedata into experience data
    player_for_model = get_full_on_raw_from_player(player)

    while True:
        # play a game
        gamedata = play(
            game,
            [player, player],
            turn_limit=config.turn_limit,
            stop_signal=shutdown_event
        )

        # don't proceed if shutdown event is set
        # we don't want games that didn't finish because of the shutdown signal
        if shutdown_event.is_set():
            logger.info(f"Play process {pid} received shutdown signal, stopping...")
            break

        # update the state counter
        logger.info(f"Play process {pid} finished a game, processing results...")
        num_states = len(gamedata.turn_data)
        with state_counter.get_lock():
            state_counter[pid] += num_states
        # get experience data from gamedata
        experience_data_list = gamedata_to_experiencedata(gamedata, player_for_model)
        logger.info(f"Play process {pid} generated {len(experience_data_list)} experience data entries")

        # send the results back to the main process
        # blocking
        logger.info(f"Play process {pid} sending results to main process...")
        result_queue.put((gamedata, experience_data_list))
    
    with num_active_workers.get_lock():
        num_active_workers.value -= 1
    logger.info(f"Play process {pid} shutting down, active workers remaining: {num_active_workers.value}")

def self_play(
        game: A0Game[Any, Any],
        player: FullModelPlayer[RecursiveFullOnRawModel[T_nn_model], Any, Any],
        experience_buffer: ExperienceBuffer,
        iteration: int
    ) -> None:
    # gamedata list
    gamedata_list: list[GameData] = []

    # serialize the player and the model it is using
    logger.info("Serializing player + model for play processes...")
    serialized_player = dill.dumps(player)

    # get the number of clients to use, create the queue and shared variables
    logger.info(f"Setting up multiprocessing IPC with {config.num_workers} workers...")
    num_workers = config.num_workers
    result_queue: Queue[tuple[GameData, list[ExperienceData]]] = Queue(maxsize=num_workers)
    num_active_workers: Synchronized[int] = Value('i', 0)
    state_counter: SynchronizedArray[int] = Array('i', [0] * num_workers)
    shutdown_event = Event()

    # start the play processes,
    # which will send game results back to the main process
    logger.info("Starting play processes...")
    processes: list[Process] = []
    for i in range(num_workers):
        logger.info(f"Setting up and starting play process {i}...")
        # start the play process with the serialized player
        with num_active_workers.get_lock():
            num_active_workers.value += 1
        p = Process(
            target=_play_process,
            args=(game, serialized_player, state_counter, result_queue, num_active_workers, shutdown_event, i)
        )
        p.start()
        processes.append(p)
    
    # process results and coordinate shutdown
    logger.info("Main process entering result processing loop...")
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
                logger.info(f"Total states played {total_states} reached the training sample target {config.training_samples}, sending shutdown signal to play processes...")
                shutdown_event.set()
            # log state counts
            state_str = f"Num States Played ({total_states}/{config.training_samples}): "
            for _, count in enumerate(state_counter):
                state_str += f"{count} | "
            logger.info(state_str)
            # break the loop if shutdown event is set
            if shutdown_event.is_set():
                break
        
    logger.info("Main process finished result processing loop, waiting for play processes to shut down...")
    
    # wait for all play processes to finish
    for p in processes:
        # note: join() is blocking,
        # and waits for the processes in order
        # can't decrement num_active_workers here.
        p.join()
        logger.info(f"Play process {p.pid} has shut down.")

    # save the game data to disk
    logger.info("Saving game data to disk...")
    if config.training_dir:
        with open(config.training_dir + f"gamedata_{iteration + 1}.pkl", 'wb') as f:
            pickle.dump(gamedata_list, f)
