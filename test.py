import concurrent.futures
from concurrent.futures import Future, wait, FIRST_COMPLETED
import time

import os

num_cores = os.cpu_count() or 4

def _play():
    time.sleep(2)
    return 10

def self_play() -> None:
    total = 0
    # use self-play with the model and mcts to generate training data
    # run each game in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures: list[Future[int]] = []
        # start a game for each core
        for _ in range(num_cores):
            game_data_future = executor.submit(
                _play,
            )
            futures.append(game_data_future)
        
        # when a game finishes,
        while True:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                result = future.result()
                total += result
                print(f"Total: {total}")

                game_data_future = executor.submit(
                _play,
                )
                futures.append(game_data_future)

             # if the training set is full, cancel all currently running games
            if total >= 100:
                print("Training set is full, cancelling all games")
                # for future in futures:
                #     future.cancel()
                break

            # start a new game
            
        
        # for game_data_future in concurrent.futures.as_completed(futures):
        #     print("game finished, starting a new game")
        #     # get the game data
        #     game_data = game_data_future.result()
        #     total += game_data

        #     print(f"Total: {total}")

        #      # if the training set is full, cancel all currently running games
        #     if total >= 100:
        #         print("Training set is full, cancelling all games")
        #         for future in futures:
        #             future.cancel()
        #         break

        #     # start a new game
        #     game_data_future = executor.submit(
        #         _play,
        #     )
        #     futures.append(game_data_future)

            

           

self_play()