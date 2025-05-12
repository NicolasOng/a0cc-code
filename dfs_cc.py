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
SAVE_INTERVAL = 1000

def load_progress():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "rb") as f:
            return pickle.load(f)
    return {
        "stack": [Game(4, 6)],
        #"stack": [Game(7, 1)],
        "visited": set(),
        "steps": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }

def save_progress(progress):
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(progress, f)
    print(f"[{time.strftime('%X')}] Progress saved. Steps: {progress['steps']}, Stack: {len(progress['stack'])} Wins: {progress['wins']}, Losses: {progress['losses']}, Draws: {progress['draws']}")

def dfs():
    progress = load_progress()
    stack = progress["stack"]
    visited = progress["visited"]
    steps = progress["steps"]
    wins = progress["wins"]
    losses = progress["losses"]
    draws = progress["draws"]

    try:
        while stack:
            current = stack.pop()

            if current in visited:
                continue
            visited.add(current)
            steps += 1

            #print(f"Step {steps}")

            if current.end:
                #print(f"Reached terminal state.")
                if current.winner is None:
                    draws += 1
                elif current.winner == Player.PLAYER_X:
                    wins += 1
                else:
                    losses += 1
                continue

            for next_state in generate_next_states(current):
                if next_state not in visited:
                    stack.append(next_state)

            if steps % SAVE_INTERVAL == 0:
                progress.update({
                    "stack": stack,
                    "visited": visited,
                    "steps": steps,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                })
                save_progress(progress)
        save_progress(progress)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        progress.update({
            "stack": stack,
            "visited": visited,
            "steps": steps,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        })
        save_progress(progress)

def read_progress():
    with open(SAVE_FILE, "rb") as f:
        progress = pickle.load(f)
    print(f"Steps: {progress['steps']}, Stack: {len(progress['stack'])}, Visited: {len(progress['visited'])}, Wins: {progress['wins']}, Losses: {progress['losses']}, Draws: {progress['draws']}")

if __name__ == "__main__":
    dfs()
    read_progress()
