from solve_file_loader import SolveData, Outcome
from cc_ranking import CCLocalRank12, CCState, CCDefaultRank
from cc_lookups import CCBaselineSolver

def file_loading_test():
    entries, mem = read_8byte_array("solve-data/CC-SOLVE-BASELINE-49-3.dat")
    print(f"Entries: {entries}, Memory Size: {len(mem)}")

    a = get_value_2(mem, 0)
    b = get_value_2(mem, 1)
    print(f"Value at index 0: {a}, Value at index 1: {b}")

    c = get_value_2(mem, 559352639)
    print(f"Value at index 559352639: {c}")

def baseline_solver_test_function():
    num_spots = 49
    num_players = 2
    num_pieces = 3
    d = CCBaselineSolver("solve-data/CC-SOLVE-BASELINE-49-3.dat", num_spots, num_players, num_pieces)
    l = CCLocalRank12(num_spots, num_players, num_pieces)
    s = CCState(num_spots, num_pieces, num_players)

    # set to (0, 10), (10, 20), ...
    n = 10000000
    i_start = 25 * n
    i_end = min(60 * n, l.get_max_rank())
    for x in range(i_start, i_end):
        l.unrank(x, s)
        s.print_ascii_compact()
        print(d.lookup(s))

baseline_solver_test_function()