import random
import os
from typing import Optional, Callable, Any

from tqdm import tqdm

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

def generate_n_done_ranks(n: int, fn: str = "done_ranks.txt"):
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
            done = cc.done(board)
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

def generate_files(fn: str, ranks: Optional[list[int]], mode: str, num_lines: int, start: int=0) -> None:
    '''
    generates a file with the given ranks and mode. If the number of lines exceeds num_lines, creates new files with incrementing suffixes.
    '''
    print(f"Generating {mode} for ranks...")

    # objects for state data
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)

    # file handling objects
    base_fn, ext = os.path.splitext(fn)
    file_idx = 0
    line_count = 0
    f = None
    def open_new_file(idx: int):
        return open(f"{base_fn}_{idx}{ext}", "w")

    # if ranks is None, use all ranks
    num_ranks = len(ranks) if ranks else r.get_max_rank()
    print(num_ranks, "ranks to generate.")

    f = open_new_file(file_idx)
    try:
        # loop through the ranks and write the desired information to the file(s)
        for i in tqdm(range(start, num_ranks)):
            # if the line count exceeds num_lines, create a new file
            if line_count >= num_lines:
                f.close()
                file_idx += 1
                f = open_new_file(file_idx)
                line_count = 0
            
            # unrank the rank and the board
            rank = ranks[i] if ranks else i
            r.unrank(rank, s)
            board = s.get_board()

            # depending on the mode, write the desired information to the file
            if mode == "done":
                done = cc.done(board)
                f.write(f"{rank} {1 if done else 0}\n")
                line_count += 1
            
            elif mode == "winner":
                winner = cc.winner(board)
                if winner == Player.PLAYER_X:
                    pw = 0
                elif winner == Player.PLAYER_O:
                    pw = 1
                else:
                    pw = -1
                f.write(f"{rank} {pw}\n")
                line_count += 1
            
            elif mode == "illegal":
                illegal = not cc.legal(board)
                f.write(f"{rank} {1 if illegal else 0}\n")
                line_count += 1
            
            elif mode == "moves":
                moves = cc.generate_moves_for_given_board(board)
                # TODO: choose a better way to represent moves
                f.write(f"Rank: {rank}, Moves: {moves}\n")
                line_count += 1
            
            elif mode == "done_full":
                done = cc.done(board)
                if done:
                    f.write(f"{rank}\n")
                    line_count += 1
            
            elif mode == "winner_full":
                winner = cc.winner(board)
                if winner == Player.PLAYER_X:
                    pw = 0
                elif winner == Player.PLAYER_O:
                    pw = 1
                else:
                    pw = -1
                if pw != -1:
                    f.write(f"{rank} {pw}\n")
                    line_count += 1
            
            elif mode == "solvedata":
                f.write(s.get_ascii_compact() + f"{l.lookup(s)}\n")
    finally:
        if f:
            f.close()

def parse_line(line: Optional[str], line_no: Optional[int], output: Optional[int]) -> tuple[int | float, Optional[list[int]]]:
    '''
    Parses a line from the file and returns the input and output.
    If line is None, returns (None, None).
    '''
    # EOF line ('')
    if not line: return float('inf'), None
    # the rest of the lines ('...\n')
    if line_no is not None:
        # if line number is given, input is the line number
        # output is the sorted list of integers on the line
        return line_no, [output] if output is not None else sorted(list(set(map(int, line.strip().split()))))
    else:
        # if input is present, split the line into input and output
        parts = line.strip().split(" ", 1)
        return int(parts[0]), [output] if output is not None else sorted(list(set(map(int, parts[1].strip().split()))))

def print_rank_info(rank: int, s: CCState, r: CCDefaultRank, l: CCBaselineSolver, cc: Game) -> None:
    '''
    Prints the rank information for the given rank.
    This includes the board view, ASCII representation, the done state, the winner, and the outcome.
    '''
    # get the info
    r.unrank(rank, s)
    board = s.get_board()
    done = cc.done(board)
    winner = cc.winner(board)
    result = l.lookup(s)
    if result == 0 or result == 3: # Draw or Illegal
        r_winner = None
    elif result == 1: # Loss
        r_winner = Player.PLAYER_O
    elif result == 2: # Win
        r_winner = Player.PLAYER_X
    else:
        raise ValueError(f"Unexpected result: {result}")
    illegal = not cc.legal(board)
    # print the info
    print(board.board_view())
    print(f"Current Player: {board.current_player}")
    s.print_ascii()
    print(f"Rank: {rank}, Done: {done}, Winner: {winner}, Result: {r_winner} ({result}), Illegal: {illegal}")

def compare_line(line1: tuple[Optional[int], Optional[list[int]]],
                 line2: tuple[Optional[int], Optional[list[int]]],
                 pkg: tuple[CCState, CCDefaultRank, CCBaselineSolver, Game, str]) -> bool:
    '''
    Compares two lines of the format (input, output) and prints the differences.
    Assumes line1 is the baseline and line2 is the new implementation.
    '''
    input1, output1 = line1
    input2, output2 = line2
    (s, r, l, cc, mode) = pkg

    if input1 is None and input2 is None:
        raise ValueError("Both inputs are None, this should not happen.")
    elif input1 is None:
        print(f"Input/Output {input2}/{output2} is missing in the baseline file.")
        return True
    elif input2 is None:
        print(f"Input/Output {input1}/{output1} is missing in the new implementation file.")
        return True
    elif input1 == input2:
        if output1 != output2:
            print(f"Inputs match but outputs differ: {input1}/{output1} vs {input2}/{output2}")
            if mode == "done":
                rank = input1
                done1 = output1[0] if output1 else None
                done2 = output2[0] if output2 else None
                print_rank_info(rank, s, r, l, cc)
                print(f"Expected {'done' if done1 == 1 else 'not done'} ({done1}), got {'done' if done2 == 1 else 'not done'} ({done2})")
                print("---")
            elif mode == "solvedata":
                rank = input1
                print_rank_info(rank, s, r, l, cc)
                print(f"Expected solve data output: {output1}, got: {output2}")
                print("---")
            elif mode == "winner":
                rank = input1
                winner1 = output1[0] if output1 else None
                winner2 = output2[0] if output2 else None
                print_rank_info(rank, s, r, l, cc)
                print(f"Expected winner: {winner1}, got: {winner2}")
                print("---")
            elif mode == "illegal":
                rank = input1
                illegal1 = output1[0] if output1 else None
                illegal2 = output2[0] if output2 else None
                print_rank_info(rank, s, r, l, cc)
                print(f"Expected {'illegal' if illegal1 == 1 else 'not illegal'} ({illegal1}), got {'illegal' if illegal2 == 1 else 'not illegal'} ({illegal2})")
                print("---")
            return True
        else:
            return False  # No differences found
    else:
        raise ValueError(f"Inputs do not match: {input1}, {input2} (this should not happen)")

def stream_validate(fn1: str, fn2: str, mode: str, no_input: bool, output: Optional[int], parse_line: Callable[[Optional[str], Any, Any], tuple[Any, Any]], compare_line: Callable[[Any, Any, Any], bool]) -> None:
    '''
    Validates two files by checking if they have the same content.
    fn1 is the baseline file and fn2 is the new implementation file.

    The files are expected to be in the format "input output" on each line,
    - if no_input is True, just "output" on each line (input = line number).
    - if output is given, it is the output of the line (there is no output)

    The inputs are expected to be sorted in ascending order.
    (they should be comparable with the < and == operators)
    '''
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)
    cc = Game(config.board_size, config.num_pieces)
    pkg = (s, r, l, cc, mode)
    # Stream compare two files line by line, handling missing lines and mismatches.
    # open both files
    with open(fn1, 'r') as f1, open(fn2, 'r') as f2:
        # read the first line from both files
        line1 = f1.readline()
        line2 = f2.readline()
        line_no1 = 0
        line_no2 = 0

        # loop until both files are exhausted
        differences = 0
        total_lines = 0
        while line1 or line2:
            total_lines += 1
            # get the input/output from the line in file 1 and file 2
            # if EOF is reached, input should be float('inf'), so we always advance the other file
            input1, output1 = parse_line(line1, line_no1 if no_input else None, output)
            input2, output2 = parse_line(line2, line_no2 if no_input else None, output)

            # if the inputs are the same, check if the outputs are the same
            if input1 == input2:
                # log the results of the comparison
                diff = compare_line((input1, output1), (input2, output2), pkg)
                # and advance both files to the next line
                line1 = f1.readline()
                line2 = f2.readline()
                line_no1 += 1
                line_no2 += 1
            # if the input of file 1 is less than the input of file 2, it means file 2 is missing this input
            elif input1 < input2:
                # log the input file 1 has that file 2 is missing
                diff = compare_line((input1, output1), (None, None), pkg)
                # and advance file 1 to the next line
                line1 = f1.readline()
                line_no1 += 1
            # if the input of file 2 is less than the input of file 1, it means file 1 is missing this input
            else:
                # log the input file 2 has that file 1 is missing
                diff = compare_line((None, None), (input2, output2), pkg)
                # and advance file 2 to the next line
                line2 = f2.readline()
                line_no2 += 1
            # if there was a difference, increment the differences counter
            if diff: differences += 1
    print(f"Stream validation complete. Found {differences}/{total_lines} differences between {fn1} and {fn2}.")

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
    num_ranks = len(ranks) if ranks else r.get_max_rank()
    print(num_ranks, "ranks to validate.")
    for i in tqdm(range(num_ranks)):
        rank = ranks[i] if ranks else i

        try:
            r.unrank(rank, s)
        except Exception as e:
            print(f"Error unranking {rank}: {e}")
            continue
        board = s.get_board()

        if not cc.done(board):
            continue

        winner = cc.winner(board)
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
    
    print(f"Validation complete. Found {mis_matches} mismatches out of {len(ranks) if ranks is not None else num_ranks} ranks.")

def generate_small_validation_files(dir: str, n: int):
    '''
    Generates small validation files for the given directory and number of ranks.
    Delete any existing files in the directory before generating new ones.
    '''
    generate_n_random_ranks(n, dir + "ranks.txt")
    generate_n_done_ranks(n, dir + "ranks_done.txt")

    ranks = read_ranks_from_file(dir + "ranks.txt")
    done_ranks = read_ranks_from_file(dir + "ranks_done.txt")

    generate_files(dir + "done.txt", ranks, "done", n + 1)
    generate_files(dir + "winner.txt", done_ranks, "winner", n + 1)
    generate_files(dir + "illegal.txt", ranks, "illegal", n + 1)
    #generate_files(dir + "moves.txt", ranks, "moves", n + 1)

def validate_small_files(dir: str):
    '''
    Validates the small files in the given directory.
    '''
    print("Validating small files...")
    stream_validate(
        dir + "done_baseline.txt",
        dir + "done_0.txt",
        "done",
        no_input=False,
        output=None,
        parse_line=parse_line,
        compare_line=compare_line
    )

    stream_validate(
        dir + "winner_baseline.txt",
        dir + "winner_0.txt",
        "winner",
        no_input=False,
        output=None,
        parse_line=parse_line,
        compare_line=compare_line
    )

    stream_validate(
        dir + "illegal_baseline.txt",
        dir + "illegal_0.txt",
        "illegal",
        no_input=False,
        output=None,
        parse_line=parse_line,
        compare_line=compare_line
    )

def generate_large_validation_files(dir: str):
    generate_files(dir + "solvedata.txt", None, "solvedata", 10**18)

def validate_large_files(dir: str):
    stream_validate(
        dir + "solvedata_baseline.txt",
        dir + "solvedata_0.txt",
        "solvedata",
        no_input=True,
        output=None,
        parse_line=parse_line,
        compare_line=compare_line
    )

    validate_solve_data_matches_winner(None)

if __name__ == "__main__":
    # python -m cc.validations
    dir = config.data_folder + "validations/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # generate and validate small files
    # (generate files with your baseline implementation first)
    n = 1000
    generate_small_validation_files(dir, n)
    validate_small_files(dir)
