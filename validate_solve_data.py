from solve_file_loader import read_8byte_array, get_value_2, Outcome
from cc_ranking import CCLocalRank12, CCState, CCDefaultRank
import numpy.typing as npt
import numpy as np

def file_loading_test():
    entries, mem = read_8byte_array("solve-data/CC-SOLVE-BASELINE-49-3.dat")
    print(f"Entries: {entries}, Memory Size: {len(mem)}")

    a = get_value_2(mem, 0)
    b = get_value_2(mem, 1)
    print(f"Value at index 0: {a}, Value at index 1: {b}")

    c = get_value_2(mem, 559352639)
    print(f"Value at index 559352639: {c}")

def baseline_lookup(mem: npt.NDArray[np.uint64], s: CCState, r: CCDefaultRank) -> int:
    rank = r.rank(s)
    outcome = get_value_2(mem, rank)
    return outcome

def baseline_solver_test_function():
    num_spots = 49
    num_players = 2
    num_pieces = 3
    l = CCLocalRank12(num_spots, num_players, num_pieces)
    r = CCDefaultRank(num_spots, num_players, num_pieces)
    s = CCState(num_spots, num_pieces, num_players)

    _, mem = read_8byte_array("solve-data/CC-SOLVE-BASELINE-49-3.dat")

    print(f"Max rank: {l.get_max_rank()}")
    s.print_ascii()

    iterations = l.get_max_rank()
    iterations = 100000
    for x in range(iterations):
        l.unrank(x, s)
        #print(f"Rank {x}:", end='')
        #s.print_ascii()
        if baseline_lookup(mem, s, r) == Outcome.ILLEGAL.value:
            print(f"Illegal state found at rank {x}")
            s.print_ascii()
            exit(0)

baseline_solver_test_function()