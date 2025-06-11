import random
import os
from typing import Optional, Callable, Any

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
    Generates a text file with n ranks that are done.
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

    # write the ranks to a text file
    num_done = 0
    with open(fn, "w") as f:
        while num_done < n:
            rank = random.randint(0, max_rank)
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

def generate_done_full(fn: str = "done_full.txt"):
    '''
    Generates a text file with all the done ranks.
    Ranks not in the file are considered not done.
    To validate the done states, generate the same text file with a baseline chinese checkers implementation.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    print("Generating full done states for all ranks...")
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces)
    max_rank = r.get_max_rank()
    with open(fn, "w") as f:
        for rank in range(max_rank + 1):
            r.unrank(rank, s)
            board = s.get_board()
            done = cc.terminal_state(board)
            if done: f.write(f"{rank} {1 if done else 0}\n")

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
            if winner == Player.PLAYER_X:
                pw = 0
            elif winner == Player.PLAYER_O:
                pw = 1
            else:
                pw = -1

            f.write(f"{rank} {pw}\n")

def generate_winner_full(fn: str = "winner_full.txt"):
    '''
    Generates a text file with all the winners for all ranks.
    Ranks not done (so no winner) are not in the file.
    To validate the winners, generate the same text file with a baseline chinese checkers implementation.
    '''
    if os.path.exists(fn):
        print(f"{fn} already exists, skipping generation.")
        return
    print("Generating full winners for all ranks...")
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces)
    max_rank = r.get_max_rank()
    with open(fn, "w") as f:
        for rank in range(max_rank + 1):
            r.unrank(rank, s)
            board = s.get_board()
            winner = cc.get_winner(board)
            if winner == Player.PLAYER_X:
                pw = 0
            elif winner == Player.PLAYER_O:
                pw = 1
            else:
                continue
            f.write(f"{rank} {pw}\n")

def compare_lines(l1: str, l2: str, pkg: tuple[CCState, CCDefaultRank, CCBaselineSolver, Game], values_type: str) -> bool:
    '''
    Compares two lines of the format "rank, values" and prints the differences.
    The values are expected to be lists of integers of any size and order (for moves).
    Assumes l1 is the baseline and l2 is the new implementation.
    '''
    rank1, values1 = l1.split(" ", 1)
    rank2, values2 = l2.split(" ", 1)
    assert rank1 == rank2, f"Ranks do not match: {rank1} != {rank2}"
    rank = int(rank1)
    values1 = sorted(list(set(map(int, values1.strip().split()))))
    values2 = sorted(list(set(map(int, values2.strip().split()))))
    (s, r, l, cc) = pkg
    if values1 != values2:
        r.unrank(rank, s)
        board = s.get_board()
        print("---")
        print(board.board_view())
        s.print_ascii()
        if values_type == "done":
            done1 = values1[0]
            done2 = values2[0]
            print(f"{rank}: expected {'done' if done1 == 1 else 'not done'} ({done1}), got {'done' if done2 == 1 else 'not done'} ({done2})")
        elif values_type == "winner":
            winner_dict = {0: Player.PLAYER_X, 1: Player.PLAYER_O, -1: None}
            winner1 = winner_dict[values1[0]]
            winner2 = winner_dict[values2[0]]
            print(f"{rank}: expected winner {winner1} ({values1[0]}), got {winner2} ({values2[0]})")
        #print(f"Values differ for rank {rank}: values1 = {values1}, values2 = {values2}")
        return True
    return False

def parse_line(line: Optional[str], line_no: int, no_input: bool) -> tuple[int | float, Optional[list[int]]]:
    '''
    Parses a line from the file and returns the input and output.
    If line is None, returns (None, None).
    '''
    # EOF line ('')
    if not line: return float('inf'), None
    # the rest of the lines ('...\n')
    if no_input:
        # if no_input, input is the line number
        # output is the sorted list of integers on the line
        return line_no, sorted(list(set(map(int, line.strip().split()))))
    else:
        # if input is present, split the line into input and output
        parts = line.strip().split(" ", 1)
        return int(parts[0]), sorted(list(set(map(int, parts[1].strip().split()))))

def validate_files(fn1: str, fn2: str, same_ranks: bool, only_values: bool, values_type: str) -> None:
    '''
    Validates two files by checking if they have the same content.
    fn1 is the baseline file and fn2 is the new implementation file.

    The files are expected to be in the format "input output" on each line,
    or if no_input is True, just "output" on each line (input = line number).

    The inputs are expected to be sorted in ascending order.
    (they should be comparable with the < and == operators)
    '''
    if not os.path.exists(fn1) or not os.path.exists(fn2):
        print(f"One of the files {fn1} or {fn2} does not exist.")
        return
    
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)
    pkg = (s, r, l, cc)
    
    num_diff = 0
    with open(fn1, "r") as f1, open(fn2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            diff = compare_lines(line1.strip(), line2.strip(), pkg, values_type)
            if diff: num_diff += 1
            if diff and not same_ranks and not only_values:
                break
        
        # Check for extra lines in either file
        longer, shorter, longer_fn = (lines1, lines2, fn1) if len(lines1) > len(lines2) else (lines2, lines1, fn2)
        if len(longer) > len(shorter):
            for i in range(len(shorter), len(longer)):
                print(f"Extra line in {longer_fn} at {i+1}: {longer[i].strip()}")
                num_diff += 1
    print(f"Validation complete. Found {num_diff} differences between {fn1} and {fn2}.")

def stream_validate(fn1: str, fn2: str, no_input: bool, parse_line: Callable[[Optional[str]], tuple[Any, Any]], compare_line: Callable[[Any, Any, Any], None]) -> None:
    # Stream compare two files line by line, handling missing lines and mismatches.
    # open both files
    with open(fn1, 'r') as f1, open(fn2, 'r') as f2:
        # read the first line from both files
        line1 = f1.readline()
        line2 = f2.readline()
        line_no1 = 0
        line_no2 = 0

        # loop until both files are exhausted
        while line1 or line2:
            # get the input/output from the line in file 1 and file 2
            # if EOF is reached, input should be float('inf'), so we always advance the other file
            input1, output1 = parse_line(line1)
            input2, output2 = parse_line(line2)

            # if the inputs are the same, check if the outputs are the same
            if input1 == input2:
                # log the results of the comparison
                compare_line((input1, output1), (input2, output2), pkg)
                # and advance both files to the next line
                line1 = f1.readline()
                line2 = f2.readline()
                line_no1 += 1
                line_no2 += 1
            # if the input of file 1 is less than the input of file 2, it means file 2 is missing this input
            elif input1 < input2:
                # log the input file 1 has that file 2 is missing
                compare_line((input1, output1), (None, None), pkg)
                # and advance file 1 to the next line
                line1 = f1.readline()
                line_no1 += 1
            # if the input of file 2 is less than the input of file 1, it means file 1 is missing this input
            else:
                # log the input file 2 has that file 1 is missing
                compare_line((None, None), (input2, output2), pkg)
                # and advance file 2 to the next line
                line2 = f2.readline()
                line_no2 += 1

def validate_solve_data_matches_winner(ranks: Optional[list[int]]):
    '''
    Validates the solve data by checking the winner of the given ranks
    and comparing it to the expected winner from the solve data.
    This is only for done ranks.

    If ranks is None, it will check all ranks.
    Otherwise, it will check only the given ranks.

    No file needs to be generated for this validation.
    '''
    print("Validating solve data for done ranks...")
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)

    # to check the initial state of the board
    # print(cc.board.board_view())

    mis_matches = 0
    num_ranks = len(ranks) if ranks else r.get_max_rank() + 1
    for i in range(num_ranks):
        rank = ranks[i] if ranks else i

        r.unrank(rank, s)
        board = s.get_board()

        if not cc.terminal_state(board):
            continue

        winner = cc.get_winner(board)
        result = l.lookup(s)

        if result == 0 or result == 3: # Draw or Illegal
            sd_winner = None
        elif result == 1: # Loss
            sd_winner = Player.PLAYER_O
        elif result == 2: # Win
            sd_winner = Player.PLAYER_X
        else:
            sd_winner = None  # Handle unexpected result
        
        if winner != sd_winner:
            mis_matches += 1
            print("...")
            print(board.board_view())
            print(board.current_player)
            s.print_ascii()
            print(f"{rank}: Winner {winner} does not match solve data {sd_winner} ({result}).")
    
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
    generate_winner(done_ranks, dir + "winner.txt")

    validate_files(
        dir + "done_baseline.txt",
        dir + "done.txt",
        same_ranks=True,
        only_values=False,
        values_type="done"
    )

    # validate_files(
    #     dir + "winner.txt",
    #     dir + "winner_baseline.txt",
    #     same_ranks=True,
    #     only_values=False,
    #     values_type="winner"
    # )
    
    validate_solve_data_matches_winner(done_ranks)
