import pickle
import os
import time

import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)

from ChineseCheckers import Game, Board, Move, Point, Player, Tile

def generate_next_states(game: Game):
    moves = game.start_turn()

    states = []
    for move in moves:
        new_game = Game()
        new_game.set_game(game)
        logging.debug(new_game.board.board_view())
        logging.debug(move)
        new_game.end_turn(move)
        states.append(new_game)
    
    return states

SAVE_FILE = "dfs_progress.pkl"
SAVE_INTERVAL = 100

def load_progress():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "rb") as f:
            return pickle.load(f)
    return {
        "stack": [Game(4, 6)],
        "steps": 0,
    }

def save_progress(progress):
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(progress, f)
    print(f"[{time.strftime('%X')}] Progress saved. Steps: {progress['steps']}")

def dfs():
    progress = load_progress()
    stack = progress["stack"]
    steps = progress["steps"]

    try:
        while stack:
            current = stack.pop()
            steps += 1

            #print(f"Step {steps}")

            if current.end:
                #print(f"Reached terminal state.")
                continue

            for next_state in generate_next_states(current):
                stack.append(next_state)

            if steps % SAVE_INTERVAL == 0:
                progress.update({
                    "stack": stack,
                    "steps": steps,
                })
                save_progress(progress)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        progress.update({
            "stack": stack,
            "steps": steps,
        })
        save_progress(progress)

if __name__ == "__main__":
    dfs()
