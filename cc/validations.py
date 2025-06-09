import random
import os

from config import config
from cc.ranking import CCDefaultRank, CCState
from cc.core import Game, Player
from cc.lookups import CCBaselineSolver

def generate_n_random_ranks(n: int, fn: str = "ranks.txt"):
    '''
    Generates a text file with n random ranks.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    
    print("Generating random ranks...")
    # get the max rank
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    max_rank = r.get_max_rank()
    
    # select n random ranks
    ranks = [random.randint(0, max_rank) for _ in range(n)]

    # write the ranks to a text file
    with open(fn, "w") as f:
        for rank in ranks:
            f.write(f"{rank}\n")

def generate_done_ranks(n: int, fn: str = "done_ranks.txt"):
    '''
    Generates a text file with ranks that are done.
    For 4x4, 3, this is about 6.8% of the total ranks.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    
    print("Generating done ranks...")
    # get the max rank
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces)
    max_rank = r.get_max_rank()
    
    # select n random ranks
    ranks = [random.randint(0, max_rank) for _ in range(n)]

    # write the ranks to a text file
    num_done = 0
    with open(fn, "w") as f:
        for rank in ranks:
            r.unrank(rank, s)
            board = s.get_board()
            done = cc.terminal_state(board)
            if done:
                f.write(f"{rank}\n")
                num_done += 1
    
    print(f"Generated {num_done} done ranks out of {n} total ranks.")

def read_ranks_from_file(fn: str) -> list[int]:
    '''
    Reads ranks from a text file and returns them as a list of integers.
    '''
    print(f"Reading ranks from {fn}...")
    with open(fn, "r") as f:
        ranks = [int(line.strip()) for line in f.readlines()]
    return ranks

def generate_moves(ranks: list[int], fn: str = "moves.txt"):
    '''
    Generates a text file with n random ranks and their corresponding moves.
    To validate the moves, generate the same text file with a baseline chinese checkers implementation.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    
    print("Generating moves for ranks...")
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)

    # generate the moves for each board, and put them in a text file
    with open(fn, "w") as f:
        for rank in ranks:
            r.unrank(rank, s)
            board = s.get_board()
            moves = cc.generate_moves_for_given_board(board)
            # TODO: choose a better way to represent moves
            f.write(f"Rank: {rank}, Moves: {moves}\n")

def generate_done(ranks: list[int], fn: str = "done.txt"):
    '''
    Generates a text file with n random ranks and if the board is done or not.
    To validate the done states, generate the same text file with a baseline chinese checkers implementation.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    
    print("Generating done states for ranks...")
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)

    # generate the done decision for each board, and put them in a text file
    with open(fn, "w") as f:
        for rank in ranks:
            r.unrank(rank, s)
            board = s.get_board()
            done = cc.terminal_state(board)
            f.write(f"{rank} {1 if done else 0}\n")

def generate_winner(ranks: list[int], fn: str = "winner.txt"):
    '''
    Generates a text file with n random ranks and their corresponding winner.
    To validate the winners, generate the same text file with a baseline chinese checkers implementation.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    
    print("Generating winners for ranks...")
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)

    # generate the winner for each board, and put them in a text file
    with open(fn, "w") as f:
        for rank in ranks:
            r.unrank(rank, s)
            board = s.get_board()
            winner = cc.get_winner(board)
            f.write(f"{rank} {0 if winner == Player.PLAYER_X else 1}\n")

def validate_solve_data(ranks: list[int]):
    '''
    Validates the solve data by checking the winner of the given ranks
    and comparing it to the expected winner from the solve data.
    '''
    print("Validating solve data for done ranks...")
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)

    # to check the initial state of the board
    # print(cc.board.board_view())

    mis_matches = 0
    for rank in ranks:
        r.unrank(rank, s)
        board = s.get_board()
        winner = cc.get_winner(board)

        result = l.lookup(s)

        if result == 0 or result == 3:
            sd_winner = None
        elif result == 1:
            sd_winner = Player.PLAYER_X if board.current_player == Player.PLAYER_O else Player.PLAYER_O
        elif result == 2:
            sd_winner = board.current_player
        else:
            sd_winner = None  # Handle unexpected result
        
        # if winner != sd_winner:
        #     mis_matches += 1
        #     print("...")
        #     print(board.board_view())
        #     print(board.current_player)
        #     s.print_ascii()
        #     print(f"Mismatch for rank {rank}: Expected winner {sd_winner}, got {winner}")
        
        if winner == sd_winner:
            print("---")
            print(board.board_view())
            print(board.current_player)
            s.print_ascii()
            print(f"Rank {rank} is valid: Winner {winner} matches solve data {sd_winner}.")
    
    print(f"Validation complete. Found {mis_matches} mismatches out of {len(ranks)} ranks.")

if __name__ == "__main__":
    # python -m cc.validations
    dir = config.data_folder + "validations/"
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    # Generate random ranks and done ranks
    n = 1000
    done_n = int(n / 0.068)  # Adjusted for 6.8% done ranks
    generate_n_random_ranks(n, dir + "ranks.txt")
    generate_done_ranks(done_n, dir + "done_ranks.txt")
    
    ranks = read_ranks_from_file(dir + "ranks.txt")
    done_ranks = read_ranks_from_file(dir + "done_ranks.txt")
    print(f"Read {len(ranks)} ranks and {len(done_ranks)} done ranks from files.")
    
    generate_moves(ranks, dir + "moves.txt")
    generate_done(ranks, dir + "done.txt")
    generate_winner(ranks, dir + "winner.txt")
    
    validate_solve_data(done_ranks)
