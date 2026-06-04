from a0.players.human import HumanPlayer
from a0.players.mcts_rollout import MCTSRolloutPlayer
from a0.players.random import RandomPlayer

from a0.eval.player import make_bfs_baseline, make_baseline

from a0.game import play

from cc.core import Game

from config import config

def main():
    human_player = HumanPlayer()
    #human_player = RandomPlayer()

    # DBEval MCTS Rollout player: uses the exact single-agent BFS distance-to-goal
    # (DB) evaluator. Loads bfs_db/bfs_<board_size>_<num_pieces>.npy from config.input_dir.
    other_player: MCTSRolloutPlayer = make_bfs_baseline()
    # print the MCTS search tree after each of this player's moves
    other_player.print_tree = True
    #other_player = RandomPlayer()

    #human_player = other_player

    results = play(
        Game(
            config.board_size,
            config.num_pieces,
            no_reverse_moves=not config.backwards_moves,
            no_illegal_moves=not config.illegal_moves,
            no_side_moves=not config.sideways_moves
        ),
        players=[
            human_player,
            other_player
        ],
        turn_limit=80
    )

    # print the results
    print(f"Game ended: {results.ended}, Winner: {results.winner}, Turns: {len(results.turn_data)}")
    if results.final_board:
        print(f"Final board:\n{results.final_board.board_view()}")

    return results.winner is not None, results.final_board

def main2():
    not_draws = 0
    for i in range(100):
        print(f"Starting game {i+1}")
        not_draw, final_board = main()
        if not_draw:
            not_draws += 1
            if final_board:
                print(f"Final board state:\n{final_board.board_view()}")
        print("-" * 40)

    print(f"Out of 100 games, {not_draws} were not draws.")

if __name__ == "__main__":
    main()
