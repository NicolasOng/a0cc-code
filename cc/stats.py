import random

from tqdm import tqdm

from config import config
from cc.ranking import CCDefaultRank, CCState
from cc.core import Game, Player
from cc.lookups import CCBaselineSolver
from cc.ground_truth import GroundTruth

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pickle

def num_children_per_state(num_ranks: int | None):
    '''
    Generates an np array with the number of children per state.
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

def do_binning_for_pwc(percent_winning_children_array: npt.NDArray[np.float64], num_bins: int, n: int, fn: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # counts: number of states in each bin (n,)
    # bin_edges: edge of each bin. eg first bin's edges are bin_edges[0], bin_edges[1] (n+1,)
    counts, bin_edges = np.histogram(percent_winning_children_array, bins=num_bins, range=(0, 1))
    normalized_counts = counts / n  # normalize counts to get percentages
    # get bin centers (n,)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # save this data to a npz file
    np.savez(config.stats_dir + fn,
             counts=counts,
             normalized_counts=normalized_counts,
             bin_edges=bin_edges,
             bin_centers=bin_centers)

    return counts, normalized_counts, bin_edges, bin_centers

def percent_winning_childen_per_state(num_ranks: int | None):
    '''
    Generates an np array with the percent of winning children per state
    from the perspective of the current player.
    '''
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    gt = GroundTruth()

    # full list of ranks and percent winning children
    rank_list: list[int] = []
    percent_winning_children_list: list[float] = []

    # subset of ranks
    winning_ranks: list[int] = []
    pwc_on_winning_ranks: list[float] = []
    losing_ranks: list[int] = []
    pwc_on_losing_ranks: list[float] = []

    # decide how many ranks to check (if None specified, check all ranks)
    max_rank = gt.get_max_rank()
    n = max_rank if num_ranks is None else num_ranks

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to check (if checking all ranks, use the index as the rank)
        cur_rank = i if num_ranks is None else random.randint(0, max_rank - 1)
        
        # unrank the current rank to get the state
        board = gt.unrank(cur_rank)

        # get the winner of the state based on the solve data
        board_winner = gt.get_winner(board)

        # ignore illegal and draw states
        if board_winner is None:
            continue

        # get the moves for the board
        moves = cc.generate_moves_for_given_board(board)

        # for each move, check if it results in a "winning" state for the current player
        current_player = board.current_player
        num_winners = 0
        num_moves = 0
        for move in moves:
            # apply the move to the state
            board.apply_move(move)
            # get the winner of the state based on the solve data
            move_winner = gt.get_winner(board)
            # ignore illegal and draw moves
            if move_winner is None:
                board.undo_move(move)
                continue
            # if the winner is the current player, increment the count
            # if move_winner == Player.PLAYER_X: # (to do everything from Player X's perspective)
            if move_winner == current_player:
                num_winners += 1
            # un-apply the move
            board.undo_move(move)
            num_moves += 1
        percent_winners = num_winners / num_moves if num_moves > 0 else 0.0

        # append the rank and the percent of winning moves to the lists
        rank_list.append(cur_rank)
        percent_winning_children_list.append(percent_winners)

        # do pwc the winning and losing ranks
        # this ignores children of draws and illegal states
        other_player = Player.PLAYER_O if current_player == Player.PLAYER_X else Player.PLAYER_X
        if board_winner == current_player:
            winning_ranks.append(cur_rank)
            pwc_on_winning_ranks.append(percent_winners)
        elif board_winner == other_player:
            losing_ranks.append(cur_rank)
            pwc_on_losing_ranks.append(percent_winners)
    
    # convert the lists to numpy arrays of shape (n,)
    percent_winning_children_array = np.array(percent_winning_children_list, dtype=float)
    rank_array = np.array(rank_list, dtype=int)
    winning_ranks_array = np.array(winning_ranks, dtype=int)
    pwc_on_winning_ranks_array = np.array(pwc_on_winning_ranks, dtype=float)
    losing_ranks_array = np.array(losing_ranks, dtype=int)
    pwc_on_losing_ranks_array = np.array(pwc_on_losing_ranks, dtype=float)
    # save the array to a npz file
    np.savez(config.stats_dir + 'percent_winning_children.npz',
            ranks=rank_array,
            percent_winning_children=percent_winning_children_array,
            winning_ranks=winning_ranks_array,
            pwc_on_winning_ranks=pwc_on_winning_ranks_array,
            losing_ranks=losing_ranks_array,
            pwc_on_losing_ranks=pwc_on_losing_ranks_array
            )
    
    # create bins for the percent winning children data
    do_binning_for_pwc(percent_winning_children_array, num_bins=100, n=n, fn='percent_winning_children_histogram_100.npz')

    # again with 10 bins
    do_binning_for_pwc(percent_winning_children_array, num_bins=10, n=n, fn='percent_winning_children_histogram_10.npz')

    # again for the winning and losing ranks
    do_binning_for_pwc(pwc_on_winning_ranks_array, num_bins=100, n=len(winning_ranks), fn='pwc_on_winning_ranks_histogram_100.npz')
    do_binning_for_pwc(pwc_on_losing_ranks_array, num_bins=100, n=len(losing_ranks), fn='pwc_on_losing_ranks_histogram_100.npz')

def graph_percent_winning_children(fn: str, bins: int):
    '''
    Generates a graph of the percent winning children per state.
    '''
    # load the histogram data from the npz file
    histogram_data = np.load(config.stats_dir + f'{fn}.npz')
    counts = histogram_data['normalized_counts']
    bin_centers = histogram_data['bin_centers']

    # Plot as bar chart
    plt.bar(bin_centers, counts, width=1 / bins, edgecolor='black')
    plt.xlabel('Percent of Winning Children')
    plt.ylabel('Percent')
    plt.title(f'Percentage of States with Given Percent of Winning Children (bins={bins})')
    plt.xticks(np.arange(0, 1.1, 0.1))  # Set x-ticks to be from 0 to 1 with step 0.1
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(config.plot_dir + f'{fn}.png')
    plt.clf()

def count_states_with_all_children_winning():
    percent_winning_children_data = np.load(config.stats_dir + 'percent_winning_children.npz')
    percent_winning_children_array = percent_winning_children_data['percent_winning_children']
    # get the percentage of states where all children are winning
    num_states_with_all_children_winning = np.sum(percent_winning_children_array == 1.0)
    percent_states_with_all_children_winning = num_states_with_all_children_winning / len(percent_winning_children_array)
    print(f'Number of states with all children winning: {num_states_with_all_children_winning} ({percent_states_with_all_children_winning:.2%})')
    # do the same for the number of states with no winning children
    num_states_with_no_winning_children = np.sum(percent_winning_children_array == 0.0)
    percent_states_with_no_winning_children = num_states_with_no_winning_children / len(percent_winning_children_array)
    print(f'Number of states with no winning children: {num_states_with_no_winning_children} ({percent_states_with_no_winning_children:.2%})')
    # save the results to a pkl file
    results = {
        'num_states_with_all_children_winning': num_states_with_all_children_winning,
        'num_states_with_no_winning_children': num_states_with_no_winning_children,
        'total_states': len(percent_winning_children_array),
        'percent_states_with_all_children_winning': percent_states_with_all_children_winning,
        'percent_states_with_no_winning_children': percent_states_with_no_winning_children
    }
    with open(config.stats_dir + 'states_with_all_children_winning.pkl', 'wb') as f:
        pickle.dump(results, f)

def count_num_winning_states(num_ranks: int | None = None):
    '''
    Prints the and saves the percentage of winning states for each player.
    '''
    r = CCDefaultRank(config.num_spots, config.num_players, config.num_pieces)
    s = CCState(config.num_spots, config.num_pieces, config.num_players)
    cc = Game(config.board_size, config.num_pieces, False, False, False)
    l = CCBaselineSolver(config.solve_data, config.num_spots, config.num_players, config.num_pieces)

    px_wins = 0
    po_wins = 0
    draws = 0
    illegal = 0
    total_states = 0

    px_win_on_px_turn = 0
    po_win_on_po_turn = 0

    px_turns = 0
    po_turns = 0

    # decide how many ranks to check (if None specified, check all ranks)
    max_rank = r.get_max_rank()
    n = max_rank if num_ranks is None else num_ranks

    total_states = n

    # for the amount of ranks specified,
    for i in tqdm(range(n)):
        # get the rank to check (if checking all ranks, use the index as the rank)
        cur_rank = i if num_ranks is None else random.randint(0, max_rank)
        
        # unrank the current rank to get the state
        r.unrank(cur_rank, s)
        # get the board from the state
        board = s.get_board()
        # get the winner of the state based on the solve data
        winner = l.get_winner(board)
        # increment the counts
        if board.current_player == Player.PLAYER_X:
            px_turns += 1
        elif board.current_player == Player.PLAYER_O:
            po_turns += 1
        if winner == Player.PLAYER_X:
            px_wins += 1
            if board.current_player == Player.PLAYER_X:
                px_win_on_px_turn += 1
        elif winner == Player.PLAYER_O:
            po_wins += 1
            if board.current_player == Player.PLAYER_O:
                po_win_on_po_turn += 1
        elif winner == None:
            # no winner, check if it's a draw or illegal state
            sd = l.board_lookup(board)
            if sd == 0:
                draws += 1
            elif sd == 3:
                illegal += 1
    
    # calculate the percentages
    px_win_percentage = px_wins / total_states
    po_win_percentage = po_wins / total_states
    px_win_on_px_turn_percentage = px_win_on_px_turn / total_states
    po_win_on_po_turn_percentage = po_win_on_po_turn / total_states
    px_turn_percentage = px_turns / total_states
    po_turn_percentage = po_turns / total_states
    draw_percentage = draws / total_states
    illegal_percentage = illegal / total_states

    # print the results
    print(f'Player X wins: {px_wins} ({px_win_percentage:.2%})')
    print(f'Player O wins: {po_wins} ({po_win_percentage:.2%})')
    print(f'Draws: {draws} ({draw_percentage:.2%})')
    print(f'Illegal states: {illegal} ({illegal_percentage:.2%})')
    print(f'Player X wins on Player X turn: {px_win_on_px_turn} ({px_win_on_px_turn_percentage:.2%})')
    print(f'Player O wins on Player O turn: {po_win_on_po_turn} ({po_win_on_po_turn_percentage:.2%})')
    print(f'Player X losses on Player X turn: {px_turns - px_win_on_px_turn} ({(px_turns - px_win_on_px_turn)/total_states:.2%})')
    print(f'Player O losses on Player O turn: {po_turns - po_win_on_po_turn} ({(po_turns - po_win_on_po_turn)/total_states:.2%})')
    print(f'Player X turns: {px_turns} ({px_turn_percentage:.2%})')
    print(f'Player O turns: {po_turns} ({po_turn_percentage:.2%})')
    print(f'Total states: {total_states}')

    # save the results to a pkl file
    results = {
        'px_wins': px_wins,
        'po_wins': po_wins,
        'draws': draws,
        'illegal': illegal,
        'total_states': total_states,
        'px_win_percentage': px_win_percentage,
        'po_win_percentage': po_win_percentage,
        'draw_percentage': draw_percentage,
        'illegal_percentage': illegal_percentage,
        'px_win_on_px_turn': px_win_on_px_turn,
        'po_win_on_po_turn': po_win_on_po_turn,
        'px_turns': px_turns,
        'po_turns': po_turns,
        'px_win_on_px_turn_percentage': px_win_on_px_turn_percentage,
        'po_win_on_po_turn_percentage': po_win_on_po_turn_percentage,
        'px_turn_percentage': px_turn_percentage,
        'po_turn_percentage': po_turn_percentage
    }

    with open(config.stats_dir + 'num_winning_states.pkl', 'wb') as f:
        pickle.dump(results, f)

def main():
    '''
    Main function to run the script.
    Generates the number of children per state and graphs it.
    '''
    count_children = False
    count_percent_winning_children = True
    count_winning_states = False

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
        graph_percent_winning_children('percent_winning_children_histogram_100', 100)
        graph_percent_winning_children('percent_winning_children_histogram_10', 10)
        # also for the winning and losing ranks
        graph_percent_winning_children('pwc_on_winning_ranks_histogram_100', 100)
        graph_percent_winning_children('pwc_on_losing_ranks_histogram_100', 100)
        # also count the states with all children winning/losing
        count_states_with_all_children_winning()
    if count_winning_states:
        # Count the number of winning states
        count_num_winning_states()
        

if __name__ == "__main__":
    main()
