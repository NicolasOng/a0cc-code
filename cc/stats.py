import random

from tqdm import tqdm

from config import config
from cc.ranking import CCDefaultRank, CCState
from cc.core import Game, Player
from cc.lookups import CCBaselineSolver

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def num_children_per_state(num_ranks: int | None):
    '''
    Generates a dictionary with the number of children per state.
    The keys are the states and the values are the number of children.
    '''
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces, False, False, False)

    num_children_list: list[tuple[int, int]] = []

    # decide how many ranks to check (if None specified, check all ranks)
    max_rank = r.get_max_rank()
    n = max_rank if num_ranks is None else num_ranks

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to check (if checking all ranks, use the index as the rank)
        cur_rank = i if num_ranks is None else random.randint(0, max_rank)
        
        # unrank the current rank to get the state
        r.unrank(cur_rank, s)
        # get the board from the state
        board = s.get_board()
        # get the number of moves for the board
        num_moves = len(cc.generate_moves_for_given_board(board))

        # append the rank and the number of moves to the list
        num_children_list.append((cur_rank, num_moves))
    
    # convert the list to a numpy array of shape (n, 2)
    num_children_array = np.array(num_children_list, dtype=int)
    # save the array to a npy file
    np.save(config.stats_dir + 'num_children.npy', num_children_array)
    # exract the number of children from the array (n,)
    num_children = num_children_array[:, 1]
    # count occurrences of each number of children
    # both are (n,)
    unique, counts = np.unique(num_children, return_counts=True)
    normalized_counts = counts / n  # normalize counts to get percentages
    # save this to a npz file
    np.savez(config.stats_dir + 'num_children_unique_counts.npz', unique=unique, counts=normalized_counts)

def graph_num_children():
    '''
    Generates a graph of the number of children per state.
    '''
    # load the unique counts of number of children from the npz file
    unique_counts = np.load(config.stats_dir + 'num_children_unique_counts.npz')
    unique = unique_counts['unique']
    counts = unique_counts['counts']

    # Plot as bar chart
    plt.bar(unique, counts)
    plt.xlabel('Number of Children')
    plt.ylabel('Percent')
    plt.title('Percentage of States with Given Number of Children')
    plt.xticks(unique)  # Ensure ticks match the integer values
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(config.plot_dir + 'num_children_per_state.png')
    plt.clf()

def do_binning_for_pwc(percent_winning_children_array: npt.NDArray[np.float64], num_bins: int, n: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # counts: number of states in each bin (n,)
    # bin_edges: edge of each bin. eg first bin's edges are bin_edges[0], bin_edges[1] (n+1,)
    counts, bin_edges = np.histogram(percent_winning_children_array, bins=num_bins, range=(0, 1))
    normalized_counts = counts / n  # normalize counts to get percentages
    # get bin centers (n,)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, normalized_counts, bin_edges, bin_centers

def percent_winning_childen_per_state(num_ranks: int | None):
    '''
    Generates a dictionary with the number of children per state.
    The keys are the states and the values are the number of children.
    '''
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)

    rank_list: list[int] = []
    percent_winning_children_list: list[float] = []

    # decide how many ranks to check (if None specified, check all ranks)
    max_rank = r.get_max_rank()
    n = max_rank if num_ranks is None else num_ranks

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to check (if checking all ranks, use the index as the rank)
        cur_rank = i if num_ranks is None else random.randint(0, max_rank)
        
        # unrank the current rank to get the state
        r.unrank(cur_rank, s)
        # get the board from the state
        board = s.get_board()
        # get the moves for the board
        moves = cc.generate_moves_for_given_board(board)
        # for each move, check if it results in a "winning" state for the current player
        current_player = board.current_player
        num_winners = 0
        for move in moves:
            # apply the move to the state
            board.apply_move(move)
            # get the winner of the state based on the solve data
            winner = l.get_winner(board)
            # if the winner is the current player, increment the count
            if winner == current_player:
                num_winners += 1
            # un-apply the move
            board.undo_move(move)
        percent_winners = num_winners / len(moves) if len(moves) > 0 else 0.0

        # append the rank and the percent of winning moves to the lists
        rank_list.append(cur_rank)
        percent_winning_children_list.append(percent_winners)
    
    # convert the lists to numpy arrays of shape (n,)
    percent_winning_children_array = np.array(percent_winning_children_list, dtype=float)
    rank_array = np.array(rank_list, dtype=int)
    # save the array to a npy file
    np.savez(config.stats_dir + 'percent_winning_children.npz', ranks=rank_array, percent_winning_children=percent_winning_children_array)
    
    # create bins for the percent winning children data
    _, normalized_counts, bin_edges, bin_centers = do_binning_for_pwc(percent_winning_children_array, num_bins=100, n=n)
    # save this data to a npz file
    np.savez(config.stats_dir + 'percent_winning_children_histogram_100.npz', counts=normalized_counts, bin_edges=bin_edges, bin_centers=bin_centers)

    # again with 10 bins
    _, normalized_counts, bin_edges, bin_centers = do_binning_for_pwc(percent_winning_children_array, num_bins=10, n=n)
    np.savez(config.stats_dir + 'percent_winning_children_histogram_10.npz', counts=normalized_counts, bin_edges=bin_edges, bin_centers=bin_centers)

def graph_percent_winning_children(bins: int):
    '''
    Generates a graph of the percent winning children per state.
    '''
    # load the histogram data from the npz file
    histogram_data = np.load(config.stats_dir + f'percent_winning_children_histogram_{bins}.npz')
    counts = histogram_data['counts']
    bin_centers = histogram_data['bin_centers']

    # Plot as bar chart
    plt.bar(bin_centers, counts, width=1 / bins, edgecolor='black')
    plt.xlabel('Percent of Winning Children')
    plt.ylabel('Percent')
    plt.title(f'Percentage of States with Given Percent of Winning Children (bins={bins})')
    plt.xticks(np.arange(0, 1.1, 0.1))  # Set x-ticks to be from 0 to 1 with step 0.1
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(config.plot_dir + f'percent_winning_children_per_state_{bins}.png')
    plt.clf()

def count_states_with_all_children_winning():
    percent_winning_children_data = np.load(config.stats_dir + 'percent_winning_children.npz')
    percent_winning_children_array = percent_winning_children_data['percent_winning_children']
    # get the percentage of states where all children are winning
    num_states_with_all_children_winning = np.sum(percent_winning_children_array == 1.0)
    num_states_with_all_children_winning /= len(percent_winning_children_array)
    print(f'Percent of states with all children winning: {num_states_with_all_children_winning}')
    # do the same for the number of states with no winning children
    num_states_with_no_winning_children = np.sum(percent_winning_children_array == 0.0)
    num_states_with_no_winning_children /= len(percent_winning_children_array)
    print(f'Percent of states with no winning children: {num_states_with_no_winning_children}')

def main():
    '''
    Main function to run the script.
    Generates the number of children per state and graphs it.
    '''
    count_children = False
    count_percent_winning_children = False
    if count_children:
        # Generate the number of children per state
        num_children_per_state(num_ranks=None)  # Set to None for all ranks or specify a number
        # Generate the graph of the number of children per state
        graph_num_children()
    if count_percent_winning_children:
        # Generate the percent winning children per state
        percent_winning_childen_per_state(num_ranks=None)
        # Generate the graph of the percent winning children per state
        # with 10 and 100 bins
        graph_percent_winning_children(100)
        graph_percent_winning_children(10)
        count_states_with_all_children_winning()
    

if __name__ == "__main__":
    main()
